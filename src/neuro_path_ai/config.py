from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_root: Path
    raw_root: Path
    processed_root: Path
    checkpoints_root: Path
    logs_root: Path


def default_paths(project_root: Path | None = None) -> Paths:
    root = project_root or Path(__file__).resolve().parents[2]
    data_root = root / "data"
    return Paths(
        project_root=root,
        data_root=data_root,
        raw_root=data_root / "raw",
        processed_root=data_root / "processed",
        checkpoints_root=root / "artifacts" / "checkpoints",
        logs_root=root / "artifacts" / "logs",
    )
