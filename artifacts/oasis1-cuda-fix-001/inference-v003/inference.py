"""Safe OASIS-1 v003 subject-level inference.

This module never discovers data, trains, fine-tunes, or accesses a dataset.  It
accepts exactly nine caller-supplied slices for one subject and refuses an
untrusted checkpoint before constructing the model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

EXPECTED_CHECKPOINT_NAME = "neuropath-oasis1-cnn-v003.pt"
EXPECTED_CHECKPOINT_SHA256 = "349ccc59ec3ba8dc834fdca7f426eb50fc8d1e5be38cb8d0bd58ab7a30c73e60"
THRESHOLD = 0.5
SLICE_COUNT = 9
PREPROCESSING = {"size": 128, "mode": "L", "scale": 255.0, "mean": 0.5, "std": 0.5}
CLASS_MAPPING = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 1, "ModerateDemented": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert(PREPROCESSING["mode"]).resize(
        (PREPROCESSING["size"], PREPROCESSING["size"]), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32).copy() / PREPROCESSING["scale"]
    return torch.from_numpy((array - PREPROCESSING["mean"]) / PREPROCESSING["std"]).unsqueeze(0)


class DeterministicGridPool2d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (16, 16):
            raise ValueError("v003 requires a 16x16 feature grid from a 128x128 input")
        return F.avg_pool2d(x, kernel_size=4, stride=4)


class ImageClassifier2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for cin, cout in ((1, 16), (16, 32), (32, 64)):
            layers.extend((nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)))
        self.features = nn.Sequential(*layers, DeterministicGridPool2d())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(1024, 2))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image))


def load_model(checkpoint: str | Path, *, device: str | torch.device = "cpu",
               expected_sha256: str = EXPECTED_CHECKPOINT_SHA256) -> ImageClassifier2D:
    """Load only the named, hash-pinned v003 checkpoint."""
    path = Path(checkpoint)
    if path.name != EXPECTED_CHECKPOINT_NAME:
        raise ValueError(f"Refusing checkpoint name {path.name!r}; expected {EXPECTED_CHECKPOINT_NAME!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    # The exploratory v003 writer stores protocol/binding fields rather than
    # the generic image2d metadata; accept either exact v003 representation.
    if payload.get("architecture") not in (None, "image2d_v1"):
        raise ValueError("Checkpoint architecture metadata is not v003 image2d_v1")
    if payload.get("preprocessing") not in (None, PREPROCESSING) or payload.get("class_mapping") not in (None, CLASS_MAPPING):
        raise ValueError("Checkpoint preprocessing or class mapping metadata mismatch")
    if payload.get("variant") != "horizontal_flip" or payload.get("epoch") != 18:
        raise ValueError("Checkpoint is not the selected horizontal_flip epoch 18 checkpoint")
    protocol = payload.get("protocol", {})
    if protocol and (protocol.get("threshold") != THRESHOLD or protocol.get("aggregation") not in
                     ("mean positive slice probability per patient", "arithmetic mean of slice positive probabilities per subject")):
        raise ValueError("Checkpoint protocol does not match frozen v003 aggregation")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state or any(not torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError("Checkpoint has no finite model_state_dict")
    model = ImageClassifier2D().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_subject(model: ImageClassifier2D, images: Sequence[Image.Image], *, subject_id: str = "subject") -> dict:
    """Return the frozen nine-slice mean probability and thresholded class."""
    if len(images) != SLICE_COUNT:
        raise ValueError(f"Expected exactly {SLICE_COUNT} slices for {subject_id}; received {len(images)}")
    batch = torch.stack([preprocess_image(image) for image in images])
    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1)[:, 1].cpu().tolist()
    score = float(np.mean(probabilities))
    return {"subject_id": subject_id, "slice_count": SLICE_COUNT,
            "slice_positive_probabilities": [float(p) for p in probabilities],
            "cdr_score": score, "prediction": int(score >= THRESHOLD),
            "threshold": THRESHOLD, "scope": "exploratory non-lateralized CDR classification only"}


def predict_subject_paths(model: ImageClassifier2D, paths: Iterable[str | Path], *, subject_id: str = "subject") -> dict:
    paths = list(paths)
    if len(paths) != SLICE_COUNT:
        raise ValueError(f"Expected exactly {SLICE_COUNT} image paths; received {len(paths)}")
    with_images = []
    for path in paths:
        with Image.open(path) as image:
            with_images.append(image.copy())
    return predict_subject(model, with_images, subject_id=subject_id)
