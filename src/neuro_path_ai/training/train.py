from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn
from torch.utils.data import DataLoader

from pathlib import Path

from neuro_path_ai.data.dataset import SampleRecord, SyntheticNiftiDataset
from neuro_path_ai.evaluation.metrics import BinaryMetrics, compute_binary_metrics
from neuro_path_ai.models.fusion import FusionModel
from neuro_path_ai.models.imaging import Simple3DCNN, build_resnet18_backbone
from neuro_path_ai.utils.logging import setup_logger

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
except ImportError:  # pragma: no cover - optional dependency
    train_test_split = None
    StratifiedKFold = None


@dataclass
class TrainConfig:
    batch_size: int = 2
    steps: int = 5
    epochs: int = 3
    learning_rate: float = 1e-3
    val_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42
    checkpoint_path: str = "artifacts/checkpoints/best_model.pt"


def _synthetic_records(count: int = 8) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for idx in range(count):
        records.append(
            SampleRecord(
                subject_id=f"SYN-{idx:03d}",
                mri_path="synthetic.nii.gz",
                genetics_vector=[0.1] * 8,
                clinical_vector=[0.2] * 6,
                label=idx % 2,
            )
        )
    return records


def train_loop(
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: int,
    epochs: int,
    device: torch.device,
    log_wandb: bool,
    checkpoint_path: Path,
) -> BinaryMetrics:
    loss_fn = nn.CrossEntropyLoss()
    logger = setup_logger()

    best_sensitivity = -1.0

    last_val_metrics = BinaryMetrics(0.0, 0.0, None, 0.0, 0, 0, 0, 0)

    for epoch in range(1, epochs + 1):
        model.train()
        all_labels: list[int] = []
        all_preds: list[int] = []
        all_scores: list[float] = []
        last_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            if step > steps:
                break
            mri, features, labels = batch
            mri = mri.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(mri, features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            preds = torch.argmax(logits.detach(), dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())

        train_metrics = compute_binary_metrics(all_labels, all_preds, all_scores)
        val_metrics = evaluate(val_loader, model, device)
        last_val_metrics = val_metrics
        if val_metrics.sensitivity > best_sensitivity:
            best_sensitivity = val_metrics.sensitivity
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        logger.info(
            "epoch=%s loss=%.4f train_sens=%.4f train_spec=%.4f val_sens=%.4f val_spec=%.4f",
            epoch,
            last_loss,
            train_metrics.sensitivity,
            train_metrics.specificity,
            val_metrics.sensitivity,
            val_metrics.specificity,
        )

        if log_wandb and wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": last_loss,
                    "train/sensitivity": train_metrics.sensitivity,
                    "train/specificity": train_metrics.specificity,
                    "train/f1": train_metrics.f1,
                    "train/auc_roc": train_metrics.auc_roc,
                    "val/sensitivity": val_metrics.sensitivity,
                    "val/specificity": val_metrics.specificity,
                    "val/f1": val_metrics.f1,
                    "val/auc_roc": val_metrics.auc_roc,
                }
            )

    return last_val_metrics


def evaluate(
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model: nn.Module,
    device: torch.device,
) -> BinaryMetrics:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_scores: list[float] = []
    with torch.no_grad():
        for mri, features, labels in data_loader:
            mri = mri.to(device)
            features = features.to(device)
            labels = labels.to(device)
            logits = model(mri, features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())
    return compute_binary_metrics(all_labels, all_preds, all_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NeuroPath AI brain stack.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data to smoke test the pipeline.")
    parser.add_argument("--synthetic-count", type=int, default=32, help="Number of synthetic samples to generate when using --synthetic.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="neuro-path-ai")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use a pretrained ResNet18 backbone.")
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/checkpoints/best_model.pt")
    parser.add_argument("--k-folds", type=int, default=0, help="Run stratified K-fold cross-validation.")
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
    )

    records = _synthetic_records(count=args.synthetic_count) if args.synthetic else []
    if not records:
        raise SystemExit("No data records provided. Use --synthetic for a smoke test.")

    labels = [record.label for record in records]
    if train_test_split is None:
        raise ImportError("scikit-learn is required for stratified splits")

    if args.wandb and wandb is not None:
        wandb.init(project=args.wandb_project)

    if args.k_folds and args.k_folds > 1:
        _run_kfold(records, labels, args, config)
        return

    train_records, temp_records = train_test_split(
        records,
        test_size=config.val_split + config.test_split,
        stratify=labels,
        random_state=config.seed,
    )
    temp_labels = [record.label for record in temp_records]
    val_size = config.val_split / (config.val_split + config.test_split)
    val_records, test_records = train_test_split(
        temp_records,
        test_size=1 - val_size,
        stratify=temp_labels,
        random_state=config.seed,
    )

    train_loader = DataLoader(SyntheticNiftiDataset(train_records), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(SyntheticNiftiDataset(val_records), batch_size=config.batch_size, shuffle=False)
    _ = test_records

    tabular_dim = _infer_tabular_dim(train_records)
    model = _build_model(tabular_dim, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loop(
        train_loader,
        val_loader,
        model,
        optimizer,
        config.steps,
        config.epochs,
        device,
        args.wandb,
        Path(config.checkpoint_path),
    )


def _build_model(tabular_dim: int, args: argparse.Namespace) -> FusionModel:
    if args.pretrained_backbone:
        imaging_model = build_resnet18_backbone(in_channels=1, feature_dim=512, pretrained=True)
        imaging_dim = 512
    else:
        imaging_model = Simple3DCNN()
        imaging_dim = 128
    return FusionModel(
        imaging_backbone=imaging_model,
        tabular_dim=tabular_dim,
        imaging_dim=imaging_dim,
        num_classes=2,
    )


def _infer_tabular_dim(records: list[SampleRecord]) -> int:
    sample = records[0]
    return int(len(sample.genetics_vector) + len(sample.clinical_vector))


def _run_kfold(
    records: list[SampleRecord],
    labels: list[int],
    args: argparse.Namespace,
    config: TrainConfig,
) -> None:
    if StratifiedKFold is None:
        raise ImportError("scikit-learn is required for stratified K-fold")

    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tabular_dim = _infer_tabular_dim(records)
    kfold = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=config.seed)

    fold_metrics: list[BinaryMetrics] = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(records, labels), start=1):
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        train_loader = DataLoader(SyntheticNiftiDataset(train_records), batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(SyntheticNiftiDataset(val_records), batch_size=config.batch_size, shuffle=False)

        model = _build_model(tabular_dim, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        fold_checkpoint = Path(config.checkpoint_path).with_name(
            f"best_model_fold_{fold}.pt"
        )
        metrics = train_loop(
            train_loader,
            val_loader,
            model,
            optimizer,
            config.steps,
            config.epochs,
            device,
            args.wandb,
            fold_checkpoint,
        )
        fold_metrics.append(metrics)
        logger.info(
            "fold=%s val_sens=%.4f val_spec=%.4f val_f1=%.4f",
            fold,
            metrics.sensitivity,
            metrics.specificity,
            metrics.f1,
        )

    mean_sens = sum(m.sensitivity for m in fold_metrics) / len(fold_metrics)
    mean_spec = sum(m.specificity for m in fold_metrics) / len(fold_metrics)
    mean_f1 = sum(m.f1 for m in fold_metrics) / len(fold_metrics)

    logger.info("kfold_mean sens=%.4f spec=%.4f f1=%.4f", mean_sens, mean_spec, mean_f1)
    if args.wandb and wandb is not None:
        wandb.log(
            {
                "kfold/mean_sensitivity": mean_sens,
                "kfold/mean_specificity": mean_spec,
                "kfold/mean_f1": mean_f1,
            }
        )


if __name__ == "__main__":
    main()
