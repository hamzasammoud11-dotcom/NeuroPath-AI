from __future__ import annotations

import torch
from torch import nn


class ProgressionGRU(nn.Module):
    """GRU-based head for longitudinal cognitive trajectories."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(x)
        return self.head(outputs[:, -1])
