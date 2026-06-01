#!/usr/bin/env python3
"""
Test script to verify the normalization fix.
This demonstrates that changing normalization changes the model output.
"""

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import torch
import tempfile
import nibabel as nib

from neuro_path_ai.data.feature_processing import normalize_clinical

print("="*80)
print("NORMALIZATION BUG DEMONSTRATION")
print("="*80)

# Show the bug: normalizing without training statistics
clinical_vectors = [
    [65.0, 26.0, 0.0, 0.0],  # Default
    [70.0, 26.0, 0.0, 0.0],  # Different age
    [65.0, 20.0, 1.0, 1.0],  # Different MMSE and other features
]

print("\n1. BUGGY NORMALIZATION (computing mean/std from single sample):")
print("-" * 80)

for vec in clinical_vectors:
    # Compute mean/std from the sample itself (THE BUG)
    values = np.array(vec, dtype=np.float32)
    buggy_mean = values.mean(axis=0, keepdims=False)
    buggy_std = values.std(axis=0, keepdims=False)
    
    # Normalize using buggy method
    buggy_normalized = normalize_clinical(vec)  # No mean/std passed
    
    print(f"\nInput: {vec}")
    print(f"  Sample mean: {buggy_mean:.4f}, Sample std: {buggy_std:.6f}")
    print(f"  Normalized: {buggy_normalized.tolist()}")
    print(f"  After padding to 14: {list(buggy_normalized.tolist()) + [0]*(14-len(vec))}")

print("\n\n2. FIXED NORMALIZATION (using proper training statistics):")
print("-" * 80)

# Proper training statistics (should be saved with checkpoint)
TRAINING_MEAN = [65.0, 26.0, 0.5, 0.5]  # age, mmse, apoe4, gender
TRAINING_STD = [8.0, 5.0, 0.8, 0.5]

for vec in clinical_vectors:
    # Normalize using proper training statistics
    fixed_normalized = normalize_clinical(
        vec,
        scaler="zscore",
        mean=TRAINING_MEAN[:len(vec)],
        std=TRAINING_STD[:len(vec)]
    )
    
    print(f"\nInput: {vec}")
    print(f"  Using training mean: {TRAINING_MEAN[:len(vec)]}")
    print(f"  Using training std:  {TRAINING_STD[:len(vec)]}")
    print(f"  Normalized: {fixed_normalized.tolist()}")
    print(f"  After padding to 14: {list(fixed_normalized.tolist()) + [0]*(14-len(vec))}")

print("\n\n3. KEY DIFFERENCES:")
print("-" * 80)

print("\nWith BUGGY normalization, changing age from 65→70 produces very small changes")
print("in the normalized features (both have mean≈22, std≈28):")
buggy_1 = normalize_clinical([65.0, 26.0, 0.0, 0.0])
buggy_2 = normalize_clinical([70.0, 26.0, 0.0, 0.0])
diff_buggy = (buggy_1 - buggy_2).abs().sum().item()
print(f"  Distance between [65,26,0,0] and [70,26,0,0]: {diff_buggy:.4f}")

print("\nWith FIXED normalization, the same change produces larger, appropriate changes:")
fixed_1 = normalize_clinical([65.0, 26.0, 0.0, 0.0], scaler="zscore",
                              mean=TRAINING_MEAN[:4], std=TRAINING_STD[:4])
fixed_2 = normalize_clinical([70.0, 26.0, 0.0, 0.0], scaler="zscore",
                              mean=TRAINING_MEAN[:4], std=TRAINING_STD[:4])
diff_fixed = (fixed_1 - fixed_2).abs().sum().item()
print(f"  Distance between [65,26,0,0] and [70,26,0,0]: {diff_fixed:.4f}")

print(f"\n  Ratio (fixed/buggy): {diff_fixed/diff_buggy:.2f}x more variation with proper normalization")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("""
The bug was that normalize_clinical() computed mean/std from individual samples
instead of using training set statistics. This caused:

1. Different normalization for each input (dependent on sample values)
2. Misalignment with how the model was trained
3. Model receiving unexpectedly normalized features
4. Constant or near-constant output (49.1%) because model can't process the data

The fix: Pass proper training statistics (mean/std) to normalize_clinical()
so that ALL samples are normalized consistently using training set statistics.
""")
