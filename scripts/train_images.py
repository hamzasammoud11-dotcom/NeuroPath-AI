from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from neuro_path_ai.models.imaging import Simple3DCNN
from neuro_path_ai.models.fusion import FusionModel
from neuro_path_ai.evaluation.metrics import compute_binary_metrics


class AugmentedImageDataset(Dataset):
    """Loads 2D images from class subfolders and creates pseudo-3D volumes."""

    def __init__(
        self,
        root: Path,
        classes: dict[str, int],
        depth: int = 32,
        size: tuple[int, int] = (96, 96),
    ) -> None:
        self.root = Path(root)
        self.classes = classes
        self.depth = depth
        self.size = size
        self.samples: list[tuple[Path, int]] = []
        for cname, label in classes.items():
            folder = self.root / cname
            if not folder.exists():
                continue
            for p in folder.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").resize(self.size)
        arr = np.array(img, dtype=np.float32) / 255.0
        # create pseudo-3D by stacking the 2D slice along depth
        volume = np.stack([arr] * self.depth, axis=0)  # (D, H, W)
        volume = np.expand_dims(volume, axis=0)  # (C=1, D, H, W)
        vol_t = torch.from_numpy(volume)
        # synthetic tabular features: genetics (8) + clinical (6)
        genetics = torch.zeros(8, dtype=torch.float32)
        clinical = torch.zeros(6, dtype=torch.float32)
        features = torch.cat([genetics, clinical], dim=0)
        label_t = torch.tensor(label, dtype=torch.long)
        return vol_t, features, label_t


def train(
    data_dir: Path,
    batch_size: int,
    epochs: int,
    steps: int,
    lr: float,
    depth: int,
):
    classes = {
        "NonDemented": 0,
        "VeryMildDemented": 1,
        "MildDemented": 1,
        "ModerateDemented": 1,
    }
    ds = AugmentedImageDataset(data_dir, classes, depth=depth)
    if len(ds) == 0:
        raise SystemExit(f"No images found in {data_dir}")

    indices = list(range(len(ds)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=[ds.samples[i][1] for i in indices])

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=batch_size, shuffle=False)

    tabular_dim = 8 + 6
    imaging_backbone = Simple3DCNN(in_channels=1, feature_dim=128)
    model = FusionModel(imaging_backbone=imaging_backbone, tabular_dim=tabular_dim, imaging_dim=128, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_path = Path("artifacts/checkpoints/best_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_sensitivity = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        all_labels = []
        all_preds = []
        all_scores = []
        last_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            if step > steps:
                break
            vols, features, labels = batch
            vols = vols.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(vols, features)
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

        # Save checkpoint when validation sensitivity improves
        if val_metrics.sensitivity > best_sensitivity:
            best_sensitivity = val_metrics.sensitivity
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            print(f"Saved new best model (sensitivity={best_sensitivity:.4f}) to {checkpoint_path}")

        print(
            f"epoch={epoch} loss={last_loss:.4f} train_sens={train_metrics.sensitivity:.4f} train_spec={train_metrics.specificity:.4f} val_sens={val_metrics.sensitivity:.4f} val_spec={val_metrics.specificity:.4f}"
        )

    # final save (also save latest)
    latest_path = checkpoint_path.with_name("latest_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "epoch": epochs}, latest_path)
    print(f"Saved latest model to {latest_path}")


def evaluate(data_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], model: nn.Module, device: torch.device):
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []
    with torch.no_grad():
        for vols, features, labels in data_loader:
            vols = vols.to(device)
            features = features.to(device)
            labels = labels.to(device)
            logits = model(vols, features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())
    return compute_binary_metrics(all_labels, all_preds, all_scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("AugmentedAlzheimerDataset"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--depth", type=int, default=32, help="Pseudo-3D depth (stacked slices)")
    args = parser.parse_args()

    train(args.data_dir, args.batch_size, args.epochs, args.steps, args.lr, args.depth)


if __name__ == "__main__":
    main()
