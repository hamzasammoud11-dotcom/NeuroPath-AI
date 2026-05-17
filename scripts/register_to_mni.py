from __future__ import annotations

import argparse
from pathlib import Path

import ants


def register_to_mni(moving_path: Path, template_path: Path, output_path: Path) -> None:
    moving = ants.image_read(str(moving_path))
    template = ants.image_read(str(template_path))
    registration = ants.registration(fixed=template, moving=moving, type_of_transform="Affine")
    warped = registration["warpedmovout"]
    ants.image_write(warped, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Register MRI volume to MNI template.")
    parser.add_argument("--moving", required=True, help="Path to moving NIfTI volume.")
    parser.add_argument("--template", required=True, help="Path to MNI template NIfTI volume.")
    parser.add_argument("--output", required=True, help="Path to write registered volume.")
    args = parser.parse_args()

    register_to_mni(Path(args.moving), Path(args.template), Path(args.output))


if __name__ == "__main__":
    main()
