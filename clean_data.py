# clean_data.py

# Quick script to clean NaN values from existing data
# Save this at: C:\Frank\New_folder\Quinoa_dataset-hyper\clean_data.py


import numpy as np
from pathlib import Path

artifact_dir = Path("artifacts/preprocessed")

print("="*60)
print("CLEANING NaN VALUES FROM DATA")
print("="*60)
print(f"Looking for data in: {artifact_dir.absolute()}")

# Check if directory exists
if not artifact_dir.exists():
    print(f"\n❌ ERROR: Directory not found!")
    print(f"   Expected: {artifact_dir.absolute()}")
    print(f"   Make sure you've run Stage 1 and Stage 2 first.")
    exit(1)

# Load data
try:
    X_train = np.load(artifact_dir / 'X_train.npy')
    y_train = np.load(artifact_dir / 'y_train.npy')
    X_val = np.load(artifact_dir / 'X_val.npy')
    y_val = np.load(artifact_dir / 'y_val.npy')
    X_test = np.load(artifact_dir / 'X_test.npy')
    y_test = np.load(artifact_dir / 'y_test.npy')
except FileNotFoundError as e:
    print(f"\n❌ ERROR: {e}")
    print("   Make sure you've run Stage 1 and Stage 2 first.")
    exit(1)

print(f"\n📊 Before cleaning:")
print(f"   X_train shape: {X_train.shape}")
print(f"   X_train NaN count: {np.isnan(X_train).sum()}")
print(f"   X_val NaN count:   {np.isnan(X_val).sum()}")
print(f"   X_test NaN count:  {np.isnan(X_test).sum()}")

# Replace NaN with 0
X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

# Clip extreme values to prevent exploding gradients
X_train = np.clip(X_train, -10, 10)
X_val = np.clip(X_val, -10, 10)
X_test = np.clip(X_test, -10, 10)

print(f"\n📊 After cleaning:")
print(f"   X_train NaN count: {np.isnan(X_train).sum()}")
print(f"   X_val NaN count:   {np.isnan(X_val).sum()}")
print(f"   X_test NaN count:  {np.isnan(X_test).sum()}")

# Overwrite original files with cleaned data
np.save(artifact_dir / 'X_train.npy', X_train)
np.save(artifact_dir / 'X_val.npy', X_val)
np.save(artifact_dir / 'X_test.npy', X_test)

# Also save as cleaned versions (backup)
np.save(artifact_dir / 'X_train_cleaned.npy', X_train)
np.save(artifact_dir / 'X_val_cleaned.npy', X_val)
np.save(artifact_dir / 'X_test_cleaned.npy', X_test)

print(f"\n✅ Data cleaned and saved!")
print(f"   Overwritten: X_train.npy, X_val.npy, X_test.npy")
print(f"   Backup saved: X_train_cleaned.npy, X_val_cleaned.npy, X_test_cleaned.npy")

print(f"\n📊 Statistics after cleaning:")
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

print(f"\n✅ Data cleaning complete!")
print(f"\n💡 Next step: Run training")
print(f"   python stages/stage3_train_simple.py --config config/production.yaml")