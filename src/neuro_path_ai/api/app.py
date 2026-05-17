from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from neuro_path_ai.data.feature_processing import normalize_clinical
from neuro_path_ai.data.ingestion import (
    GridMeta,
    classify_sequence,
    load_image_grid_to_volume,
    volume_to_grid_image,
)
from neuro_path_ai.models.explainability import GradCAM3D, export_cam_to_nifti, find_default_target_layer
from neuro_path_ai.models.fusion import FusionModel
from neuro_path_ai.models.imaging import Simple3DCNN, build_resnet18_backbone

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional dependency
    nib = None

app = FastAPI(title="NeuroPath AI Inference API")

MODEL_CHECKPOINT = os.getenv("NEUROPATH_MODEL", "artifacts/checkpoints/best_model.pt")
PRETRAINED_BACKBONE = os.getenv("NEUROPATH_BACKBONE", "resnet18")
TABULAR_DIM = int(os.getenv("NEUROPATH_TABULAR_DIM", "14"))

_model: FusionModel | None = None
_target_layer: torch.nn.Module | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> FusionModel:
    if PRETRAINED_BACKBONE == "resnet18":
        # Do not attempt to load MedicalNet pretrained weights by default
        backbone = build_resnet18_backbone(in_channels=1, feature_dim=512, pretrained=False)
        imaging_dim = 512
    else:
        backbone = Simple3DCNN()
        imaging_dim = 128
    model = FusionModel(
        imaging_backbone=backbone,
        tabular_dim=TABULAR_DIM,
        imaging_dim=imaging_dim,
        num_classes=2,
    )
    checkpoint = Path(MODEL_CHECKPOINT)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=_device))
    return model


@app.on_event("startup")
def _startup() -> None:
    global _model, _target_layer
    _model = _load_model().to(_device)
    _model.eval()
    _target_layer = find_default_target_layer(_model.imaging_backbone)


@app.post("/predict")
async def predict(
    nifti_file: UploadFile = File(...),
    clinical_json: str = Form(...),
) -> dict[str, Any]:
    # Add a debug try/except to return trace details for local testing
    if nib is None:
        raise HTTPException(status_code=500, detail="nibabel is required for NIfTI handling")

    try:
        payload = json.loads(clinical_json)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="clinical_json must be a JSON object")

        clinical_vector = payload.get("clinical_features", [])
        if not isinstance(clinical_vector, list):
            raise HTTPException(status_code=400, detail="clinical_features must be a list")

        data = await nifti_file.read()
        tmp_path = Path("artifacts/uploads") / nifti_file.filename
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(data)

        volume, grid_meta, warning = _load_volume(tmp_path)
        clinical, tabular_warning = _prepare_tabular_features(clinical_vector)

        if _model is None or _target_layer is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        volume = volume.to(_device)
        clinical = clinical.to(_device)
        with torch.no_grad():
            logits = _model(volume, clinical)
            probs = torch.softmax(logits, dim=1)[:, 1]
            prediction = int((probs >= 0.5).item())

        response: dict[str, Any] = {
            "probability": float(probs.item()),
            "prediction": prediction,
            "warning": _merge_warnings(warning, tabular_warning),
        }

        # Attempt to generate heatmap, but do not fail the request if explainability fails.
        try:
            cam = GradCAM3D(_model, _target_layer)
            heatmap = cam.generate_upsampled(
                volume,
                target_shape=volume.shape[-3:],
                extra_features=clinical,
                threshold=0.6,
            )

            if grid_meta is None:
                heatmap_path = Path("artifacts/heatmaps") / f"{tmp_path.stem}_cam.nii.gz"
                heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                export_cam_to_nifti(heatmap, str(tmp_path), str(heatmap_path))
                response["heatmap_path"] = str(heatmap_path)
            else:
                heatmap_image = _heatmap_to_grid_overlay(heatmap, volume, grid_meta)
                grid_path = Path("artifacts/heatmaps") / f"{tmp_path.stem}_cam.png"
                grid_path.parent.mkdir(parents=True, exist_ok=True)
                _save_grid_image(grid_path, heatmap_image)
                response["heatmap_path"] = str(grid_path)
        except Exception as e:
            # attach a warning but return the main prediction
            response["warning"] = _merge_warnings(response.get("warning"), f"Heatmap generation failed: {e}")

        return response
    except HTTPException:
        # Re-raise HTTPExceptions unchanged
        raise
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        # Return the traceback in the HTTPException detail for debugging
        raise HTTPException(status_code=500, detail={"error": str(exc), "trace": tb})


def _load_volume(file_path: Path) -> tuple[torch.Tensor, GridMeta | None, str | None]:
    ext = file_path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        volume, meta = load_image_grid_to_volume(file_path, target_depth=64)
        sequence = classify_sequence(volume)
        volume = _normalize_for_sequence(volume, sequence)
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        warning = (
            "Analysis complete, but for clinical-grade precision (95%+), "
            "please upload the original DICOM/NIfTI data."
        )
        return tensor, meta, warning

    volume = nib.load(str(file_path)).get_fdata().astype("float32")
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    return tensor, None, None


def _prepare_tabular_features(clinical_vector: list[float]) -> tuple[torch.Tensor, str | None]:
    if not clinical_vector:
        raise HTTPException(status_code=400, detail="clinical_features cannot be empty")

    clinical = normalize_clinical(clinical_vector)
    current_dim = int(clinical.shape[0])
    if current_dim == TABULAR_DIM:
        return clinical.unsqueeze(0), None

    if current_dim < TABULAR_DIM:
        padded = torch.zeros((TABULAR_DIM,), dtype=clinical.dtype)
        padded[:current_dim] = clinical
        warning = (
            f"Received {current_dim} tabular features; padded to {TABULAR_DIM} "
            "with zeros to match the trained model."
        )
        return padded.unsqueeze(0), warning

    warning = f"Received {current_dim} tabular features; truncated to {TABULAR_DIM} to match the trained model."
    return clinical[:TABULAR_DIM].unsqueeze(0), warning


def _merge_warnings(*warnings: str | None) -> str | None:
    parts = [part.strip() for part in warnings if part and part.strip()]
    if not parts:
        return None
    return " ".join(parts)


def _normalize_for_sequence(volume: np.ndarray, sequence: str) -> np.ndarray:
    if sequence == "t1":
        return (volume - volume.mean()) / (volume.std() + 1e-6)
    if sequence == "t2":
        min_val = volume.min()
        max_val = volume.max()
        return (volume - min_val) / (max_val - min_val + 1e-6)
    return (volume - volume.mean()) / (volume.std() + 1e-6)


def _heatmap_to_grid_overlay(cam: torch.Tensor, volume: torch.Tensor, meta: GridMeta) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="opencv-python is required for JPG heatmaps") from exc

    cam_np = cam.detach().cpu().numpy()
    if cam_np.ndim == 4:
        cam_np = cam_np[0]
    cam_np = (cam_np * 255).clip(0, 255).astype("uint8")
    cam_grid = volume_to_grid_image(cam_np, meta)
    cam_grid[cam_grid < 30] = 0

    vol_np = volume.detach().cpu().numpy()
    if vol_np.ndim == 5:
        vol_np = vol_np[0, 0]
    vol_grid = volume_to_grid_image(_normalize_to_uint8(vol_np), meta)

    heatmap_color = cv2.applyColorMap(cam_grid, cv2.COLORMAP_JET)
    base_gray = cv2.cvtColor(vol_grid, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base_gray, 0.55, heatmap_color, 0.45, 0)


def _normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    min_val = float(volume.min())
    max_val = float(volume.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(volume, dtype="uint8")
    scaled = (volume - min_val) / (max_val - min_val)
    return (scaled * 255).clip(0, 255).astype("uint8")


def _save_grid_image(path: Path, image: np.ndarray) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="opencv-python is required for JPG heatmaps") from exc
    cv2.imwrite(str(path), image)
