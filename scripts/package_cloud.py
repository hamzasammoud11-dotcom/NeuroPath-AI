"""Package current local training code, including uncommitted changes, for upload."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
destination = root / "artifacts/cloud/neuropath-code.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
paths = [root / "scripts/train_images.py", root / "requirements-training.txt"]
paths.extend((root / "src/neuro_path_ai").rglob("*.py"))
with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
    for path in sorted(paths):
        archive.write(path, path.relative_to(root))
print(destination)
