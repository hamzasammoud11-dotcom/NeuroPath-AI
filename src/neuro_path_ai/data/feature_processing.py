from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def normalize_volume(volume: torch.Tensor, method: str = "zscore") -> torch.Tensor:
    data = volume.float()
    if method == "zscore":
        mean = data.mean()
        std = data.std()
        return (data - mean) / (std + 1e-6)
    if method == "minmax":
        min_val = data.min()
        max_val = data.max()
        return (data - min_val) / (max_val - min_val + 1e-6)
    if method in {"none", ""}:
        return data
    raise ValueError(f"Unknown normalization method: {method}")


def encode_genetics(
    raw_vector: Iterable[object],
    encoding: str = "one_hot",
    categories: list[int] | None = None,
) -> torch.Tensor:
    values = list(raw_vector)
    if encoding == "raw":
        return torch.tensor([float(v) for v in values], dtype=torch.float32)
    if encoding == "one_hot":
        if categories is None:
            categories = sorted({int(v) for v in values}) or [0]
        encoded: list[float] = []
        for value in values:
            vector = [1.0 if int(value) == cat else 0.0 for cat in categories]
            encoded.extend(vector)
        return torch.tensor(encoded, dtype=torch.float32)
    raise ValueError(f"Unknown genetics encoding: {encoding}")


def normalize_clinical(
    raw_vector: Iterable[float],
    scaler: str = "zscore",
    mean: list[float] | None = None,
    std: list[float] | None = None,
    min_values: list[float] | None = None,
    max_values: list[float] | None = None,
) -> torch.Tensor:
    values = np.array(list(raw_vector), dtype=np.float32)
    if values.size == 0:
        return torch.zeros((0,), dtype=torch.float32)

    if scaler == "zscore":
        if mean is None:
            mean = values.mean(axis=0, keepdims=False).tolist()
        if std is None:
            std = values.std(axis=0, keepdims=False).tolist()
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        normalized = (values - mean_arr) / (std_arr + 1e-6)
        return torch.tensor(normalized, dtype=torch.float32)

    if scaler == "minmax":
        if min_values is None:
            min_values = values.min(axis=0, keepdims=False).tolist()
        if max_values is None:
            max_values = values.max(axis=0, keepdims=False).tolist()
        min_arr = np.array(min_values, dtype=np.float32)
        max_arr = np.array(max_values, dtype=np.float32)
        normalized = (values - min_arr) / (max_arr - min_arr + 1e-6)
        return torch.tensor(normalized, dtype=torch.float32)

    if scaler in {"none", ""}:
        return torch.tensor(values, dtype=torch.float32)

    raise ValueError(f"Unknown clinical scaler: {scaler}")
