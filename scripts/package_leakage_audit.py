"""Create a NEW audit-only code ZIP. No models, data or training scripts included."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "artifacts/cloud/neuropath-leakage-audit.zip")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Archive already exists; choose a new --output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = ["scripts/leakage_audit.py", "requirements-audit.txt", "src/neuro_path_ai/__init__.py",
             "src/neuro_path_ai/models/image2d.py", "docs/LEAKAGE_AUDIT.md",
             "scripts/leakage_review.py", "scripts/leakage_review_template.html", "docs/LEAKAGE_REVIEW.md"]
    with ZipFile(args.output, "x", ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, name)
    print(args.output)


if __name__ == "__main__":
    main()
