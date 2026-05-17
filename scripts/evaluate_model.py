from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from neuro_path_ai.data.dataset import NiftiDataset, SampleRecord, load_adni_merge_records
from neuro_path_ai.evaluation.metrics import compute_binary_metrics
from neuro_path_ai.models.fusion import FusionModel
from neuro_path_ai.models.imaging import Simple3DCNN, build_resnet18_backbone


def _load_records(args: argparse.Namespace) -> list[SampleRecord]:
    if args.adni_merge and args.imaging_index:
        return load_adni_merge_records(Path(args.adni_merge), Path(args.imaging_index))
    raise SystemExit("Provide --adni-merge and --imaging-index for evaluation.")


def _build_model(tabular_dim: int, args: argparse.Namespace) -> FusionModel:
    if args.pretrained_backbone:
        backbone = build_resnet18_backbone(in_channels=1, feature_dim=512, pretrained=True)
        imaging_dim = 512
    else:
        backbone = Simple3DCNN()
        imaging_dim = 128
    return FusionModel(
        imaging_backbone=backbone,
        tabular_dim=tabular_dim,
        imaging_dim=imaging_dim,
        num_classes=2,
    )


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained NeuroPath AI model.")
    parser.add_argument("--adni-merge", type=str, required=True)
    parser.add_argument("--imaging-index", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--output-csv", type=str, default="artifacts/reports/clinical_report.csv")
    parser.add_argument("--mc-dropout-runs", type=int, default=0)
    args = parser.parse_args()

    records = _load_records(args)
    if not records:
        raise SystemExit("No records found for evaluation.")

    dataset = NiftiDataset(records, root=Path(args.data_root), return_subject_id=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    tabular_dim = len(records[0].genetics_vector) + len(records[0].clinical_vector)
    model = _build_model(tabular_dim, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_scores: list[float] = []
    all_uncertainty: list[float] = []
    all_subjects: list[str] = []

    with torch.no_grad():
        for batch in data_loader:
            mri, features, labels, subject_ids = batch
            mri = mri.to(device)
            features = features.to(device)
            labels = labels.to(device)

            if args.mc_dropout_runs and args.mc_dropout_runs > 1:
                probs, uncertainty = _mc_dropout_predict(
                    model, mri, features, runs=args.mc_dropout_runs
                )
            else:
                logits = model(mri, features)
                probs = torch.softmax(logits, dim=1)[:, 1]
                uncertainty = torch.zeros_like(probs)

            preds = (probs >= 0.5).long()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())
            all_uncertainty.extend(uncertainty.cpu().tolist())
            all_subjects.extend(list(subject_ids))

    metrics = compute_binary_metrics(all_labels, all_preds, all_scores)
    matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=False)

    print("Confusion Matrix:\n", matrix)
    print("Classification Report:\n", report)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_clinical_report(output_path, all_subjects, all_preds, all_scores, all_uncertainty, metrics)


def _write_clinical_report(
    output_path: Path,
    subjects: list[str],
    preds: list[int],
    scores: list[float],
    uncertainty: list[float],
    metrics,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("patient_id,prediction,confidence,uncertainty,sensitivity,specificity\n")
        for subject, pred, score, uncert in zip(subjects, preds, scores, uncertainty):
            handle.write(
                f"{subject},{pred},{score:.4f},{uncert:.4f},{metrics.sensitivity:.4f},{metrics.specificity:.4f}\n"
            )


def _enable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def _mc_dropout_predict(
    model: torch.nn.Module,
    mri: torch.Tensor,
    features: torch.Tensor,
    runs: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    _enable_dropout(model)
    scores: list[torch.Tensor] = []
    for _ in range(runs):
        logits = model(mri, features)
        probs = torch.softmax(logits, dim=1)[:, 1]
        scores.append(probs)
    stacked = torch.stack(scores, dim=0)
    mean = stacked.mean(dim=0)
    var = stacked.var(dim=0, unbiased=False)
    return mean, var


if __name__ == "__main__":
    evaluate()
