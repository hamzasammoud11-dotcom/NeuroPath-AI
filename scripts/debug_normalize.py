#!/usr/bin/env python3
"""Debug script to understand normalize_clinical behavior."""

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
from neuro_path_ai.data.feature_processing import normalize_clinical

# Test 1: Default inputs
print("Test 1: Default inputs (age=65, mmse=26, apoe4=0, gender=0)")
clinical_vector_1 = [65.0, 26.0, 0.0, 0.0]
normalized_1 = normalize_clinical(clinical_vector_1)
print(f"  Input:      {clinical_vector_1}")
print(f"  Normalized: {normalized_1.tolist()}")
print(f"  Mean: {np.mean(clinical_vector_1)}, Std: {np.std(clinical_vector_1)}")

# Test 2: Different age
print("\nTest 2: Different age (age=70, mmse=26, apoe4=0, gender=0)")
clinical_vector_2 = [70.0, 26.0, 0.0, 0.0]
normalized_2 = normalize_clinical(clinical_vector_2)
print(f"  Input:      {clinical_vector_2}")
print(f"  Normalized: {normalized_2.tolist()}")
print(f"  Mean: {np.mean(clinical_vector_2)}, Std: {np.std(clinical_vector_2)}")

# Test 3: Very different inputs
print("\nTest 3: Very different inputs (age=50, mmse=10, apoe4=2, gender=1)")
clinical_vector_3 = [50.0, 10.0, 2.0, 1.0]
normalized_3 = normalize_clinical(clinical_vector_3)
print(f"  Input:      {clinical_vector_3}")
print(f"  Normalized: {normalized_3.tolist()}")
print(f"  Mean: {np.mean(clinical_vector_3)}, Std: {np.std(clinical_vector_3)}")

# Test 4: Check if normalization varies enough
print("\nTest 4: Comparing normalized vectors")
print(f"  Vector 1 vs 2 (age diff=5): {np.allclose(normalized_1.numpy(), normalized_2.numpy())}")
print(f"  Vector 1 vs 3 (major diff):  {np.allclose(normalized_1.numpy(), normalized_3.numpy())}")
print(f"  Distance 1-2: {np.linalg.norm(normalized_1.numpy() - normalized_2.numpy()):.4f}")
print(f"  Distance 1-3: {np.linalg.norm(normalized_1.numpy() - normalized_3.numpy()):.4f}")
