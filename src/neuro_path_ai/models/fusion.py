from __future__ import annotations

import torch
from torch import nn


class MultimodalFusion(nn.Module):
    """Late-fusion head combining imaging, genetics, and clinical features."""

    def __init__(self, imaging_dim: int, genetics_dim: int, clinical_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(imaging_dim + genetics_dim + clinical_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)

    def forward(
        self,
        imaging_features: torch.Tensor,
        genetics_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([imaging_features, genetics_features, clinical_features], dim=1)
        hidden = self.fusion(fused)
        return self.classifier(hidden)


class TabularEncoder(nn.Module):
    """MLP encoder for genetics + clinical features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionHead(nn.Module):
    """Aggregator that balances imaging and tabular branches before classification."""

    def __init__(
        self,
        imaging_dim: int,
        tabular_dim: int,
        fusion_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.imaging_proj = nn.Sequential(
            nn.Linear(imaging_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, imaging_features: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        imaging = self.imaging_proj(imaging_features)
        tabular = self.tabular_proj(tabular_features)
        fused = torch.cat([imaging, tabular], dim=1)
        hidden = self.fusion(fused)
        return self.classifier(hidden)


class LateFusionModel(nn.Module):
    """Backbone + tabular encoder + fusion head for end-to-end training."""

    def __init__(
        self,
        imaging_backbone: nn.Module,
        tabular_dim: int,
        imaging_dim: int,
        tabular_hidden_dim: int = 64,
        tabular_out_dim: int = 128,
        fusion_dim: int = 256,
        fusion_hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.imaging_backbone = imaging_backbone
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim,
            hidden_dim=tabular_hidden_dim,
            output_dim=tabular_out_dim,
            dropout=dropout,
        )
        self.head = LateFusionHead(
            imaging_dim=imaging_dim,
            tabular_dim=tabular_out_dim,
            fusion_dim=fusion_dim,
            hidden_dim=fusion_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, volumes: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        imaging_features = self.imaging_backbone(volumes)
        tabular_encoded = self.tabular_encoder(tabular_features)
        return self.head(imaging_features, tabular_encoded)


class FusionModel(LateFusionModel):
    """Alias for the late-fusion model used in training pipelines."""
