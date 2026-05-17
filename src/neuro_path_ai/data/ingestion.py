from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    from scipy.ndimage import zoom
except ImportError:  # pragma: no cover - optional dependency
    zoom = None


@dataclass(frozen=True)
class GridMeta:
    rows: int
    cols: int
    slice_height: int
    slice_width: int


def load_image_grid_to_volume(image_path: Path, target_depth: int | None = None) -> tuple[np.ndarray, GridMeta]:
    if cv2 is None:
        raise ImportError("opencv-python is required for grid ingestion")

    image = _read_grayscale(image_path)
    rows, cols, slice_h, slice_w = _detect_grid(image)
    slices = _extract_slices(image, rows, cols, slice_h, slice_w)
    volume = np.stack(slices, axis=0).astype(np.float32)

    if target_depth is not None and target_depth > 0:
        volume = _resample_depth(volume, target_depth)

    return volume, GridMeta(rows=rows, cols=cols, slice_height=slice_h, slice_width=slice_w)


def volume_to_grid_image(volume: np.ndarray, meta: GridMeta) -> np.ndarray:
    depth = volume.shape[0]
    expected = meta.rows * meta.cols
    if depth < expected:
        padding = np.zeros((expected - depth, meta.slice_height, meta.slice_width), dtype=volume.dtype)
        volume = np.concatenate([volume, padding], axis=0)
    volume = volume[:expected]
    rows = []
    idx = 0
    for _ in range(meta.rows):
        row_slices = []
        for _ in range(meta.cols):
            row_slices.append(volume[idx])
            idx += 1
        rows.append(np.concatenate(row_slices, axis=1))
    return np.concatenate(rows, axis=0)


def classify_sequence(volume: np.ndarray) -> str:
    mean_val = float(volume.mean())
    std_val = float(volume.std())
    if std_val < 10:
        return "t1"
    if mean_val > 80:
        return "t2"
    return "flair"


def _read_grayscale(image_path: Path) -> np.ndarray:
    if image_path.suffix.lower() in {".tif", ".tiff"} and Image is not None:
        img = Image.open(image_path).convert("L")
        return np.array(img)
    # cv2.imread can fail on Windows for Unicode paths; imdecode+fromfile handles those paths reliably.
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) if encoded.size > 0 else None
    if image is None and Image is not None:
        try:
            img = Image.open(image_path).convert("L")
            return np.array(img)
        except Exception:
            pass
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def _detect_grid(image: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape
    edges = cv2.Canny(image, 30, 100)
    row_strength = edges.sum(axis=1)
    col_strength = edges.sum(axis=0)

    row_lines = _find_grid_lines(row_strength)
    col_lines = _find_grid_lines(col_strength)

    if len(row_lines) < 2 or len(col_lines) < 2:
        rows, cols, slice_h, slice_w = _fallback_grid(image)
        return rows, cols, slice_h, slice_w

    rows = len(row_lines) - 1
    cols = len(col_lines) - 1
    slice_h = int(np.median(np.diff(row_lines)))
    slice_w = int(np.median(np.diff(col_lines)))
    return rows, cols, slice_h, slice_w


def _find_grid_lines(strength: np.ndarray, min_gap: int = 4) -> list[int]:
    threshold = strength.mean() + strength.std()
    candidates = np.where(strength > threshold)[0]
    if candidates.size == 0:
        return []
    lines = [int(candidates[0])]
    for idx in candidates[1:]:
        if idx - lines[-1] >= min_gap:
            lines.append(int(idx))
    if lines[0] != 0:
        lines = [0] + lines
    if lines[-1] != len(strength) - 1:
        lines.append(len(strength) - 1)
    return lines


def _fallback_grid(image: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape
    best_rows = 1
    best_cols = 1
    best_cell = min(height, width)
    for rows in range(2, 10):
        if height % rows != 0:
            continue
        cell_h = height // rows
        cols = width // cell_h if cell_h > 0 else 1
        if cols < 2 or width % cols != 0:
            continue
        cell_w = width // cols
        if abs(cell_h - cell_w) < abs(best_cell - cell_w):
            best_rows, best_cols = rows, cols
            best_cell = cell_w
    return best_rows, best_cols, height // best_rows, width // best_cols


def _extract_slices(
    image: np.ndarray,
    rows: int,
    cols: int,
    slice_h: int,
    slice_w: int,
) -> list[np.ndarray]:
    slices: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * slice_h
            x0 = c * slice_w
            slices.append(image[y0 : y0 + slice_h, x0 : x0 + slice_w])
    return slices


def _resample_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    if zoom is None:
        raise ImportError("scipy is required for spline interpolation")
    if volume.shape[0] == target_depth:
        return volume
    factor = target_depth / volume.shape[0]
    return zoom(volume, (factor, 1.0, 1.0), order=3)
