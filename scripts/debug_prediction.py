#!/usr/bin/env python3
"""Debug script to test the full prediction pipeline."""

import sys
import json
import tempfile
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import torch
import nibabel as nib

from neuro_path_ai.api.app import _load_model, _prepare_tabular_features, _load_volume
from neuro_path_ai.models.imaging import Simple3DCNN

# Load the model
print("Loading model...")
try:
    model = _load_model()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Model loaded on {device}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Create test volumes
print("\nCreating test NIfTI volumes...")
test_cases = [
    {"name": "random_1", "seed": 42},
    {"name": "random_2", "seed": 43},
    {"name": "ones", "data": np.ones((64, 64, 64), dtype=np.float32)},
]

results = []

for test_case in test_cases:
    # Create volume
    if "seed" in test_case:
        np.random.seed(test_case["seed"])
        volume_data = np.random.rand(64, 64, 64).astype(np.float32)
    else:
        volume_data = test_case["data"]
    
    # Save as NIfTI
    img = nib.Nifti1Image(volume_data, affine=np.eye(4))
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        tmp_path = Path(f.name)
        img.to_filename(str(tmp_path))
    
    # Load volume
    try:
        volume, _, _ = _load_volume(tmp_path)
        volume = volume.to(device)
    except Exception as e:
        print(f"✗ Failed to load volume: {e}")
        tmp_path.unlink()
        continue
    
    # Test with different clinical features
    clinical_cases = [
        [65.0, 26.0, 0.0, 0.0],  # Default
        [70.0, 26.0, 0.0, 0.0],  # Different age
        [50.0, 10.0, 2.0, 1.0],  # Very different
    ]
    
    for clinical_vector in clinical_cases:
        try:
            clinical, _ = _prepare_tabular_features(clinical_vector)
            clinical = clinical.to(device)
            
            with torch.no_grad():
                logits = model(volume, clinical)
                probs = torch.softmax(logits, dim=1)[:, 1]
                prob_val = float(probs.item())
            
            result = {
                "volume": test_case["name"],
                "clinical": clinical_vector,
                "probability": prob_val,
                "probability_pct": prob_val * 100,
                "logits": logits[0].cpu().tolist(),
            }
            results.append(result)
            
            print(f"{test_case['name']:15} + {str(clinical_vector):40} -> {prob_val:.6f} ({prob_val*100:.1f}%)")
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
    
    tmp_path.unlink()

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if results:
    probs = [r["probability"] for r in results]
    prob_pcts = [r["probability_pct"] for r in results]
    
    print(f"Min probability:     {min(probs):.6f} ({min(prob_pcts):.1f}%)")
    print(f"Max probability:     {max(probs):.6f} ({max(prob_pcts):.1f}%)")
    print(f"Mean probability:    {np.mean(probs):.6f} ({np.mean(prob_pcts):.1f}%)")
    print(f"Std probability:     {np.std(probs):.6f}")
    
    # Check for constant output
    unique_probs = set(round(p, 6) for p in probs)
    print(f"Unique probabilities: {len(unique_probs)}")
    
    if len(unique_probs) == 1:
        print("\n⚠️  WARNING: All outputs are identical! This is the bug.")
        print(f"   Constant value: {probs[0]:.6f} ({probs[0]*100:.1f}%)")
    else:
        print("\n✓ Probabilities are varying correctly")
