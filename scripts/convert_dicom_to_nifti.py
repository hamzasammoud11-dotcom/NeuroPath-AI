from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def convert_dicom_to_nifti(dicom_root: Path, output_root: Path, recurse: bool = True) -> None:
    dicom_root = dicom_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if recurse:
        dicom_dirs = [p for p in dicom_root.rglob("*") if p.is_dir()]
    else:
        dicom_dirs = [dicom_root]

    for dicom_dir in dicom_dirs:
        if not any(dicom_dir.iterdir()):
            continue
        subject_name = dicom_dir.name
        target_dir = output_root / subject_name
        target_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "dcm2niix",
            "-z",
            "y",
            "-o",
            str(target_dir),
            str(dicom_dir),
        ]
        subprocess.run(command, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DICOM folders to NIfTI using dcm2niix.")
    parser.add_argument("--dicom-root", required=True, help="Root folder containing DICOM series.")
    parser.add_argument("--output-root", required=True, help="Folder to write NIfTI outputs.")
    parser.add_argument("--no-recurse", action="store_true", help="Only process the root folder.")
    args = parser.parse_args()

    convert_dicom_to_nifti(
        Path(args.dicom_root),
        Path(args.output_root),
        recurse=not args.no_recurse,
    )


if __name__ == "__main__":
    main()
