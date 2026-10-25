"""Train a binary 2D baseline using originals; augment training images only."""
import argparse
from collections import Counter
import csv
import json
import hashlib
import os
import platform
import subprocess
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, brier_score_loss
from torch.utils.data import DataLoader
from neuro_path_ai.data.images2d import discover_originals, split_originals, ImageDataset2D
from neuro_path_ai.models.image2d import CLASS_MAPPING, PREPROCESSING, ImageClassifier2D, load_image_checkpoint


def save_checkpoint(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]])


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))


def training_signature(splits, args):
    # Paths can change when a cloud runtime downloads the same images again.
    rows = {name: [{k: row[k] for k in ("sha256", "label", "source_class", "patient_id")}
                   for row in records] for name, records in splits.items()}
    config = dict(seed=args.seed, batch_size=args.batch_size, lr=args.lr,
                  patience=args.patience, preprocessing=PREPROCESSING, class_mapping=CLASS_MAPPING,
                  patient_id_source=args.patient_id_source, splits=rows)
    root = Path(__file__).resolve().parents[1]
    config["code_hashes"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in ("scripts/train_images.py", "src/neuro_path_ai/data/images2d.py",
                                          "src/neuro_path_ai/models/image2d.py")}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for images, target in loader:
            scores.extend(model(images.to(device)).softmax(1)[:, 1].cpu().tolist())
            labels.extend(target.tolist())
    predictions = np.asarray(scores) >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return dict(balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
                sensitivity=float(tp / max(tp + fn, 1)), specificity=float(tn / max(tn + fp, 1)),
                f1=float(f1_score(labels, predictions, zero_division=0)),
                roc_auc=float(roc_auc_score(labels, scores)), brier_score=float(brier_score_loss(labels, scores)),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]]), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="OriginalDataset or its parent")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/image2d"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--patient-manifest", type=Path, help="CSV with path,patient_id; paths relative to OriginalDataset")
    parser.add_argument("--patient-id-source", help="Provenance of patient mapping (source metadata or export reference)")
    parser.add_argument("--allow-image-split", action="store_true", help="Explicit exploratory fallback when patient IDs are unavailable")
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/last_model.pt at the last completed epoch")
    parser.add_argument("--require-cuda", action="store_true", help="Refuse CPU training for a cloud GPU run")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, patience and lr must be positive")
    if args.patient_manifest and not (args.patient_id_source or "").strip():
        parser.error("--patient-id-source is required with --patient-manifest")
    if args.patient_id_source and not args.patient_manifest:
        parser.error("--patient-id-source requires --patient-manifest")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("A CUDA GPU is required; select a GPU runtime")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    records, duplicates = discover_originals(args.data_dir, args.patient_manifest)
    splits = split_originals(records, args.seed, allow_image_split=args.allow_image_split)
    patient_split = args.patient_manifest is not None
    signature = training_signature(splits, args)
    if args.resume:
        if not (args.output_dir / "last_model.pt").is_file() or not (args.output_dir / "best_model.pt").is_file():
            parser.error("Resume requires last_model.pt and best_model.pt in --output-dir")
        resume = torch.load(args.output_dir / "last_model.pt", map_location="cpu", weights_only=True)
        if resume.get("training_signature") != signature:
            parser.error("Resume dataset, patient mapping, splits or training settings differ from the saved run")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        resume = None
    manifest = dict(seed=args.seed, source="https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset",
                    split_unit="patient" if patient_split else "unique original image",
                    patient_separation_verified=patient_split, patient_id_source=args.patient_id_source,
                    duplicates_removed=duplicates, preprocessing=PREPROCESSING,
                    class_mapping=CLASS_MAPPING, splits=splits)
    if not args.resume:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({name: dict(Counter(r["source_class"] for r in rows)) for name, rows in splits.items()}, indent=2))
    print("Patient IDs are disjoint according to the supplied mapping." if patient_split else
          "Patient separation is NOT verified; near-duplicate images may remain.")
    if args.prepare_only:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       numpy=np.__version__, cuda=torch.version.cuda,
                       device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    if args.resume:
        previous_environment = json.loads((args.output_dir / "environment.json").read_text(encoding="utf-8"))
        if previous_environment != environment:
            print("Warning: runtime environment changed; exact numerical reproducibility is not guaranteed.")
    environment_file = "resume_environment.json" if args.resume else "environment.json"
    (args.output_dir / environment_file).write_text(json.dumps(environment, indent=2), encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    freeze_file = "resume_requirements.txt" if args.resume else "requirements-frozen.txt"
    (args.output_dir / freeze_file).write_text(freeze.stdout, encoding="utf-8")
    loaders = {name: DataLoader(ImageDataset2D(rows, augment=name == "train"),
                                batch_size=args.batch_size, shuffle=name == "train") for name, rows in splits.items()}
    model = ImageClassifier2D().to(device)
    counts = Counter(r["label"] for r in splits["train"])
    weights = torch.tensor([len(splits["train"]) / (2 * counts[i]) for i in range(2)], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, stale, history = -1.0, 0, []
    checkpoint = args.output_dir / "best_model.pt"
    start_epoch = 1
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        best, stale, history = resume["best_score"], resume["stale_epochs"], resume["history"]
        start_epoch = resume["epoch"] + 1
        restore_rng(resume["rng_state"])
        print(f"Resuming after epoch {start_epoch - 1}")
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        total_loss, count = 0.0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            count += labels.size(0)
        metrics, _ = evaluate(model, loaders["validation"], device)
        history.append(dict(epoch=epoch, train_loss=total_loss / count, validation=metrics))
        print(json.dumps(history[-1]))
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if metrics["balanced_accuracy"] > best:
            best, stale = metrics["balanced_accuracy"], 0
            save_checkpoint(dict(architecture="image2d_v1", model_state_dict=model.state_dict(),
                            preprocessing=PREPROCESSING, class_mapping=CLASS_MAPPING, epoch=epoch,
                            validation_metrics=metrics, seed=args.seed,
                            split_unit=manifest["split_unit"], patient_id_source=args.patient_id_source,
                            patient_separation_verified=patient_split), checkpoint)
        else:
            stale += 1
        save_checkpoint(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                             epoch=epoch, best_score=best, stale_epochs=stale, history=history,
                             training_signature=signature, rng_state=rng_state()), args.output_dir / "last_model.pt")
        if stale >= args.patience:
            break
    model = load_image_checkpoint(checkpoint, device)
    metrics, scores = evaluate(model, loaders["test"], device)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "probability", "prediction"])
        for record, score in zip(splits["test"], scores):
            writer.writerow([record["path"], record["label"], score, int(score >= 0.5)])
    print("Held-out test:", json.dumps(metrics))


if __name__ == "__main__":
    main()
