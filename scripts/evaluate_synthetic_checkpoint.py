#!/usr/bin/env python3
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch.utils.data import DataLoader
from neuro_path_ai.data.dataset import SampleRecord, SyntheticNiftiDataset
from neuro_path_ai.models.fusion import FusionModel
from neuro_path_ai.models.imaging import Simple3DCNN
from neuro_path_ai.evaluation.metrics import compute_binary_metrics

checkpoint_path = ROOT / 'artifacts' / 'checkpoints' / 'best_model.pt'
ckpt = torch.load(checkpoint_path, map_location='cpu')
state_dict = ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt)

records = [
    SampleRecord(
        subject_id=f'SYN-{i:03d}',
        mri_path='synthetic.nii.gz',
        genetics_vector=[0.1] * 8,
        clinical_vector=[0.2] * 6,
        label=i % 2,
    )
    for i in range(100)
]

dataset = SyntheticNiftiDataset(records)
loader = DataLoader(dataset, batch_size=8, shuffle=False)
model = FusionModel(imaging_backbone=Simple3DCNN(), tabular_dim=dataset[0][1].shape[0], imaging_dim=128, num_classes=2)
model.load_state_dict(state_dict)
model.eval()

all_labels = []
all_preds = []
all_scores = []
with torch.no_grad():
    for mri, features, labels in loader:
        logits = model(mri, features)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= 0.5).long()
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_scores.extend(probs.tolist())

metrics = compute_binary_metrics(all_labels, all_preds, all_scores)
accuracy = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / len(all_labels)
print(f'n_samples={len(all_labels)}')
print(f'accuracy={accuracy:.4f}')
print(f'sensitivity={metrics.sensitivity:.4f}')
print(f'specificity={metrics.specificity:.4f}')
print(f'auc_roc={metrics.auc_roc:.4f}')
