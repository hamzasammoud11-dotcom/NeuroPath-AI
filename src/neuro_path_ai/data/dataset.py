from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from neuro_path_ai.data.feature_processing import (
    encode_genetics,
    normalize_clinical,
    normalize_volume,
)
from neuro_path_ai.data.transforms import white_stripe_normalization

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional dependency
    nib = None

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional dependency
    sitk = None


@dataclass(frozen=True)
class SampleRecord:
    subject_id: str
    mri_path: str
    genetics_vector: list[Any]
    clinical_vector: list[float]
    label: int


@dataclass(frozen=True)
class FeatureConfig:
    genetics_encoding: str = "one_hot"
    genetics_categories: list[int] | None = None
    clinical_scaler: str = "zscore"
    clinical_mean: list[float] | None = None
    clinical_std: list[float] | None = None
    clinical_min: list[float] | None = None
    clinical_max: list[float] | None = None


class NiftiDataset(Dataset):
    """Dataset loader for MRI volumes plus non-imaging features."""

    def __init__(
        self,
        records: list[SampleRecord],
        root: Path | None = None,
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalization: str = "zscore",
        feature_config: FeatureConfig | None = None,
        return_subject_id: bool = False,
    ) -> None:
        self.records = records
        self.root = root or Path(".")
        self.image_transform = image_transform
        self.normalization = normalization
        self.feature_config = feature_config or FeatureConfig()
        self.return_subject_id = return_subject_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, str
    ]:
        record = self.records[idx]
        volume = self._load_volume(record.mri_path)
        volume = self._normalize_volume(volume)
        if self.image_transform is not None:
            volume = self.image_transform(volume)

        genetics = encode_genetics(
            record.genetics_vector,
            encoding=self.feature_config.genetics_encoding,
            categories=self.feature_config.genetics_categories,
        )
        clinical = normalize_clinical(
            record.clinical_vector,
            scaler=self.feature_config.clinical_scaler,
            mean=self.feature_config.clinical_mean,
            std=self.feature_config.clinical_std,
            min_values=self.feature_config.clinical_min,
            max_values=self.feature_config.clinical_max,
        )
        features = torch.cat([genetics, clinical], dim=0)
        label = torch.tensor(record.label, dtype=torch.long)
        if self.return_subject_id:
            return volume, features, label, record.subject_id
        return volume, features, label

    def _load_volume(self, mri_path: str) -> torch.Tensor:
        path = self.root / mri_path
        if path.is_dir():
            return self._load_dicom_series(path)
        if path.suffix in {".nii", ".gz"} or path.name.endswith(".nii.gz"):
            return self._load_nifti(path)
        if path.suffix.lower() == ".dcm":
            return self._load_dicom_series(path.parent)
        raise ValueError(f"Unsupported MRI path: {path}")

    def _load_nifti(self, path: Path) -> torch.Tensor:
        if nib is None:
            raise ImportError("nibabel is required to load NIfTI volumes")
        volume = nib.load(str(path)).get_fdata().astype(np.float32)
        volume = np.expand_dims(volume, axis=0)
        return torch.from_numpy(volume)

    def _normalize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        if self.normalization == "white_stripe":
            return white_stripe_normalization(volume)
        return normalize_volume(volume, method=self.normalization)

    def _load_dicom_series(self, path: Path) -> torch.Tensor:
        if sitk is None:
            raise ImportError("SimpleITK is required to load DICOM series")
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(str(path))
        if not series:
            raise ValueError(f"No DICOM series found in {path}")
        reader.SetFileNames(series)
        image = reader.Execute()
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        volume = np.expand_dims(volume, axis=0)
        return torch.from_numpy(volume)


class SyntheticNiftiDataset(Dataset):
    """Synthetic fallback for smoke testing the training loop."""

    def __init__(
        self,
        records: list[SampleRecord],
        volume_shape: tuple[int, int, int] = (96, 96, 96),
        return_subject_id: bool = False,
    ) -> None:
        self.records = records
        self.volume_shape = volume_shape
        self.return_subject_id = return_subject_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, str
    ]:
        record = self.records[idx]
        volume = torch.rand((1, *self.volume_shape), dtype=torch.float32)
        genetics = encode_genetics(record.genetics_vector)
        clinical = normalize_clinical(record.clinical_vector)
        features = torch.cat([genetics, clinical], dim=0)
        label = torch.tensor(record.label, dtype=torch.long)
        if self.return_subject_id:
            return volume, features, label, record.subject_id
        return volume, features, label


def load_manifest(csv_path: Path) -> list[SampleRecord]:
    table = pd.read_csv(csv_path)
    records: list[SampleRecord] = []
    for _, row in table.iterrows():
        records.append(
            SampleRecord(
                subject_id=str(row["subject_id"]),
                mri_path=str(row["mri_path"]),
                genetics_vector=_parse_vector(row["genetics_vector"]),
                clinical_vector=_parse_vector(row["clinical_vector"]),
                label=int(row["label"]),
            )
        )
    return records


def load_adni_merge_records(
    adni_merge_csv: Path,
    imaging_index_csv: Path,
    diagnosis_map: dict[str, int] | None = None,
    genetics_field: str = "APOE4",
    clinical_fields: Iterable[str] = ("MMSE",),
    visit_field: str = "VISCODE",
) -> list[SampleRecord]:
    diagnosis_map = diagnosis_map or {"CN": 0, "MCI": 1, "AD": 2}
    adni_table = pd.read_csv(adni_merge_csv)
    imaging_table = pd.read_csv(imaging_index_csv)

    adni_table[visit_field] = adni_table[visit_field].astype(str).str.lower()
    imaging_table[visit_field] = imaging_table[visit_field].astype(str).str.lower()

    merged = adni_table.merge(
        imaging_table,
        left_on=["PTID", visit_field],
        right_on=["subject_id", visit_field],
        how="inner",
    )

    genetics_mean = _mean_impute_value(adni_table.get(genetics_field))
    clinical_means = {
        field: _mean_impute_value(adni_table.get(field)) for field in clinical_fields
    }

    records: list[SampleRecord] = []
    for _, row in merged.iterrows():
        diagnosis = str(row.get("DX", ""))
        if diagnosis not in diagnosis_map:
            continue

        genetics_raw = row.get(genetics_field)
        genetics_value, genetics_missing = _impute_with_flag(genetics_raw, genetics_mean)

        clinical_vector: list[float] = []
        for field in clinical_fields:
            value, missing = _impute_with_flag(row.get(field), clinical_means[field])
            clinical_vector.append(value)
            clinical_vector.append(float(missing))

        records.append(
            SampleRecord(
                subject_id=str(row.get("PTID")),
                mri_path=str(row.get("mri_path")),
                genetics_vector=[genetics_value, float(genetics_missing)],
                clinical_vector=clinical_vector,
                label=diagnosis_map[diagnosis],
            )
        )
    return records


def _parse_vector(raw_value: Any) -> list[float]:
    if isinstance(raw_value, str):
        return [float(v) for v in raw_value.split()] if raw_value else []
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        return [float(v) for v in raw_value]
    return [float(raw_value)]


def _mean_impute_value(series: pd.Series | None) -> float:
    if series is None:
        return 0.0
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().any():
        return float(values.mean(skipna=True))
    return 0.0


def _impute_with_flag(value: Any, fallback: float) -> tuple[float, bool]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback, True
    if np.isnan(numeric):
        return fallback, True
    return numeric, False
