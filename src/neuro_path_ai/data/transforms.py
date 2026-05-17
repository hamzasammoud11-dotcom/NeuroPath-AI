from __future__ import annotations

from typing import Callable

import torch

try:
    from monai.transforms import Compose, RandFlip, RandRotate, RandZoom
except ImportError:  # pragma: no cover - optional dependency
    Compose = None
    RandFlip = None
    RandRotate = None
    RandZoom = None


def build_train_transforms() -> Callable[[torch.Tensor], torch.Tensor]:
    if Compose is None:
        raise ImportError("monai is required for augmentation transforms")
    return Compose(
        [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        ]
    )


def build_eval_transforms() -> Callable[[torch.Tensor], torch.Tensor]:
    def _identity(volume: torch.Tensor) -> torch.Tensor:
        return volume

    return _identity


def white_stripe_normalization(
    volume: torch.Tensor,
    lower_pct: float = 0.8,
    upper_pct: float = 0.95,
) -> torch.Tensor:
    data = volume.float()
    flat = data.flatten()
    flat = flat[flat > 0]
    if flat.numel() == 0:
        return data

    lower = torch.quantile(flat, lower_pct)
    upper = torch.quantile(flat, upper_pct)
    stripe = flat[(flat >= lower) & (flat <= upper)]
    if stripe.numel() == 0:
        return data
    scale = stripe.mean()
    if scale <= 0:
        return data
    return data / (scale + 1e-6)
