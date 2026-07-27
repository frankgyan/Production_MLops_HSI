# debug_data.py
"""
Quick script to check for NaN values in your preprocessed data
"""

import numpy as np
from pathlib import Path

artifact_dir = Path("artifacts/preprocessed")

print("="*60)
print("CHECKING DATA FOR NaN VALUES")
print("="*60)

# Load data
X_train = np.load(artifact_dir / 'X_train.npy')
y_train = np.load(artifact_dir / 'y_train.npy')
X_val = np.load(artifact_dir / 'X_val.npy')
y_val = np.load(artifact_dir / 'y_val.npy')
X_test = np.load(artifact_dir / 'X_test.npy')
y_test = np.load(artifact_dir / 'y_test.npy')

print(f"\n📊 Data shapes:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_val:   {X_val.shape}")
print(f"   y_val:   {y_val.shape}")
print(f"   X_test:  {X_test.shape}")
print(f"   y_test:  {y_test.shape}")

# Check for NaN
print(f"\n🔍 Checking for NaN values:")
nan_count_train = np.isnan(X_train).sum()
nan_count_val = np.isnan(X_val).sum()
nan_count_test = np.isnan(X_test).sum()
print(f"   X_train NaN count: {nan_count_train}")
print(f"   X_val NaN count:   {nan_count_val}")
print(f"   X_test NaN count:  {nan_count_test}")

# Check for Inf
print(f"\n🔍 Checking for Inf values:")
inf_count_train = np.isinf(X_train).sum()
inf_count_val = np.isinf(X_val).sum()
inf_count_test = np.isinf(X_test).sum()
print(f"   X_train Inf count: {inf_count_train}")
print(f"   X_val Inf count:   {inf_count_val}")
print(f"   X_test Inf count:  {inf_count_test}")

# Check statistics
print(f"\n📊 Data statistics:")
print(f"   X_train min: {X_train.min():.6f}")
print(f"   X_train max: {X_train.max():.6f}")
print(f"   X_train mean: {X_train.mean():.6f}")
print(f"   X_train std:  {X_train.std():.6f}")

print(f"\n   X_val min: {X_val.min():.6f}")
print(f"   X_val max: {X_val.max():.6f}")
print(f"   X_val mean: {X_val.mean():.6f}")
print(f"   X_val std:  {X_val.std():.6f}")

print(f"\n   X_test min: {X_test.min():.6f}")
print(f"   X_test max: {X_test.max():.6f}")
print(f"   X_test mean: {X_test.mean():.6f}")
print(f"   X_test std:  {X_test.std():.6f}")

# Check class distribution
print(f"\n📊 Class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(unique, counts):
    class_names = {0: 'HNHP', 1: 'HNLP', 2: 'LNHP', 3: 'LNLP'}
    print(f"   Class {cls} ({class_names[cls]}): {count} samples")

print(f"\n✅ Data check complete!")