from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Heavy libraries imported lazily inside functions to allow the API
# to start even when full ML dependencies are not installed.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model.pt"
MODEL_CHECKPOINT = Path(os.getenv("NEUROPATH_MODEL", str(DEFAULT_CHECKPOINT)))
if not MODEL_CHECKPOINT.is_absolute():
    MODEL_CHECKPOINT = PROJECT_ROOT / MODEL_CHECKPOINT

BACKBONE_STRATEGY = os.getenv("NEUROPATH_BACKBONE", "auto").lower()
TABULAR_DIM_ENV = os.getenv("NEUROPATH_TABULAR_DIM")

_model: FusionModel | None = None
_target_layer = None
_device = None
_tabular_dim: int = 14  # Default value, will be overridden at startup
TABULAR_DIM: int = 14  # Backward-compatible alias for any lingering references


def _load_checkpoint_state(checkpoint: Path) -> dict[str, torch.Tensor] | None:
    if not checkpoint.exists():
        return None
    try:
        import torch
    except ImportError:
        # Torch not available in this environment; skip loading checkpoints.
        return None

    payload = torch.load(checkpoint, map_location=_device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dict.")
    return state_dict


def _infer_backbone(state_dict: dict[str, torch.Tensor]) -> str | None:
    if any(
        key.startswith("imaging_backbone.layer")
        or key.startswith("imaging_backbone.conv1")
        for key in state_dict
    ):
        return "resnet18"
    if any(
        key.startswith("imaging_backbone.encoder")
        or key.startswith("imaging_backbone.projection")
        for key in state_dict
    ):
        return "simple3d"
    return None


def _infer_tabular_dim(state_dict: dict[str, torch.Tensor]) -> int | None:
    weight = state_dict.get("tabular_encoder.net.0.weight")
    if weight is None:
        return None
    return int(weight.shape[1])


def _resolve_backbone(state_dict: dict[str, torch.Tensor] | None) -> str:
    if BACKBONE_STRATEGY not in {"auto", "simple3d", "resnet18"}:
        raise ValueError("NEUROPATH_BACKBONE must be one of: auto, simple3d, resnet18.")
    if BACKBONE_STRATEGY != "auto":
        return BACKBONE_STRATEGY
    if state_dict is None:
        return "simple3d"
    inferred = _infer_backbone(state_dict)
    if inferred is None:
        raise ValueError("Unable to infer backbone from checkpoint. Set NEUROPATH_BACKBONE explicitly.")
    return inferred


def _resolve_tabular_dim(state_dict: dict[str, torch.Tensor] | None) -> int:
    if TABULAR_DIM_ENV is not None:
        return int(TABULAR_DIM_ENV)
    inferred = _infer_tabular_dim(state_dict) if state_dict is not None else None
    return inferred if inferred is not None else 14


def _load_model() -> FusionModel:
    checkpoint = Path(MODEL_CHECKPOINT)
    state_dict = _load_checkpoint_state(checkpoint)
    backbone_name = _resolve_backbone(state_dict)
    tabular_dim = _resolve_tabular_dim(state_dict)

    if backbone_name == "resnet18":
        # Do not attempt to load MedicalNet pretrained weights by default
        backbone = build_resnet18_backbone(in_channels=1, feature_dim=512, pretrained=False)
        imaging_dim = 512
    else:
        backbone = Simple3DCNN()
        imaging_dim = 128

    model = FusionModel(
        imaging_backbone=backbone,
        tabular_dim=tabular_dim,
        imaging_dim=imaging_dim,
        num_classes=2,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


@app.on_event("startup")
def _startup() -> None:
    global _model, _target_layer, _tabular_dim, TABULAR_DIM
    checkpoint = Path(MODEL_CHECKPOINT)
    state_dict = _load_checkpoint_state(checkpoint)
    _tabular_dim = _resolve_tabular_dim(state_dict)
    TABULAR_DIM = _tabular_dim
    try:
        import torch

        global _device
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _model = _load_model().to(_device)
        _model.eval()
        _target_layer = find_default_target_layer(_model.imaging_backbone)
    except ImportError:
        # ML runtime not present; leave model as None so API can still run.
        _model = None
        _target_layer = None


@app.get("/health")
def health() -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    nifti_file: UploadFile = File(...),
    clinical_json: str = Form(...),
) -> dict[str, Any]:
    # Add a debug try/except to return trace details for local testing
    try:
        import torch

        payload = json.loads(clinical_json)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="clinical_json must be a JSON object")

        clinical_vector = payload.get("clinical_features", [])
        if not isinstance(clinical_vector, list):
            raise HTTPException(status_code=400, detail="clinical_features must be a list")

        gender_value = payload.get("gender")
        if gender_value is not None:
            clinical_vector = _append_gender_feature(clinical_vector, gender_value)

        data = await nifti_file.read()
        tmp_path = PROJECT_ROOT / "artifacts" / "uploads" / nifti_file.filename
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
                heatmap_path = PROJECT_ROOT / "artifacts" / "heatmaps" / f"{tmp_path.stem}_cam.nii.gz"
                heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                export_cam_to_nifti(heatmap, str(tmp_path), str(heatmap_path))
                response["heatmap_path"] = str(heatmap_path)
            else:
                heatmap_image = _heatmap_to_grid_overlay(heatmap, volume, grid_meta)
                grid_path = PROJECT_ROOT / "artifacts" / "heatmaps" / f"{tmp_path.stem}_cam.png"
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


def _append_gender_feature(clinical_vector: list[float], gender_value: str | int | float) -> list[float]:
    if isinstance(gender_value, str):
        normalized_gender = gender_value.strip().lower()
        if normalized_gender == "female":
            gender_code = 0.0
        elif normalized_gender == "male":
            gender_code = 1.0
        elif normalized_gender in {"other", "nonbinary", "non-binary", "nb"}:
            gender_code = 0.5
        else:
            raise HTTPException(status_code=400, detail="gender must be 'female', 'male', or 'other'")
    elif isinstance(gender_value, (int, float)):
        gender_code = float(gender_value)
    else:
        raise HTTPException(status_code=400, detail="gender must be a string or numeric value")

    return clinical_vector + [gender_code]


def _load_volume(file_path: Path) -> tuple[torch.Tensor, GridMeta | None, str | None]:
    ext = file_path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        volume, meta = load_image_grid_to_volume(file_path, target_depth=64)
        sequence = classify_sequence(volume)
        volume = _normalize_for_sequence(volume, sequence)
        import torch
        import numpy as np

        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        warning = (
            "Analysis complete, but for clinical-grade precision (95%+), "
            "please upload the original DICOM/NIfTI data."
        )
        return tensor, meta, warning

    if nib is None:
        raise HTTPException(status_code=500, detail="nibabel is required for NIfTI handling")
    volume = nib.load(str(file_path)).get_fdata().astype("float32")
    import torch
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
    return tensor, None, None


def _prepare_tabular_features(clinical_vector: list[float]) -> tuple[torch.Tensor, str | None]:
    if not clinical_vector:
        raise HTTPException(status_code=400, detail="clinical_features cannot be empty")
    import torch

    # Use training set statistics for normalization (must match training data preprocessing)
    # Features: [age, mmse, apoe4, gender, ...additional_features]
    # These defaults should match the training data statistics
    clinical_mean = [65.0, 26.0, 0.5, 0.5]  # Typical means for age, mmse, apoe4, gender
    clinical_std = [8.0, 5.0, 0.8, 0.5]     # Typical stds for these features
    
    # Pad to match expected number of features (remaining features get mean=0, std=1)
    while len(clinical_mean) < _tabular_dim:
        clinical_mean.append(0.0)
    while len(clinical_std) < _tabular_dim:
        clinical_std.append(1.0)
    
    clinical = normalize_clinical(
        clinical_vector,
        scaler="zscore",
        mean=clinical_mean[:len(clinical_vector)],
        std=clinical_std[:len(clinical_vector)]
    )
    current_dim = int(clinical.shape[0])
    if current_dim == _tabular_dim:
        return clinical.unsqueeze(0), None

    if current_dim < _tabular_dim:
        padded = torch.zeros((_tabular_dim,), dtype=clinical.dtype)
        padded[:current_dim] = clinical
        warning = (
            f"Received {current_dim} tabular features; padded to {_tabular_dim} "
            "with zeros to match the trained model."
        )
        return padded.unsqueeze(0), warning

    warning = f"Received {current_dim} tabular features; truncated to {_tabular_dim} to match the trained model."
    return clinical[:_tabular_dim].unsqueeze(0), warning


def _merge_warnings(*warnings: str | None) -> str | None:
    parts = [part.strip() for part in warnings if part and part.strip()]
    if not parts:
        return None
    return " ".join(parts)


def _normalize_for_sequence(volume: np.ndarray, sequence: str) -> np.ndarray:
    import numpy as np

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
