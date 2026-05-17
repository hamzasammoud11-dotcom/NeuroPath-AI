from __future__ import annotations

import torch
from torch import nn

try:
    from monai.networks.nets import resnet18
except ImportError:  # pragma: no cover - optional dependency
    resnet18 = None


class Simple3DCNN(nn.Module):
    """Baseline 3D CNN for MRI volumes."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.projection = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x).flatten(1)
        return self.projection(features)


def build_resnet18_backbone(
    in_channels: int = 1,
    feature_dim: int = 512,
    pretrained: bool = True,
) -> nn.Module:
    if resnet18 is None:
        raise ImportError("monai is required for ResNet18 backbones")
    try:
        # MONAI's MedicalNet pretrained weights require n_input_channels=1 and feed_forward=False
        return resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=feature_dim,
            pretrained=pretrained,
            feed_forward=False,
        )
    except TypeError:
        # Older MONAI versions may not accept the same kwargs; fall back to basic call
        return resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=feature_dim,
        )
