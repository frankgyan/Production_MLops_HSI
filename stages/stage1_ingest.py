# stages/stage1_ingest.py
"""
STAGE 1: Data Ingestion & Versioning
Handles nested directory structure with .tiff files in subfolders.
"""

import os
import json
import yaml
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tifffile as tiff

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_all_hsi_files(data_dir: Path, extensions=None):
    """
    Recursively find all HSI files in nested directories.
    
    Args:
        data_dir: Root directory to search
        extensions: List of extensions to look for (e.g., ['.tiff', '.tif', '.npy'])
    
    Returns:
        Dictionary mapping filename (without extension) to full path
    """
    if extensions is None:
        extensions = ['.tiff', '.tif', '.npy']
    
    file_index = {}
    
    print(f"🔍 Recursively scanning: {data_dir}")
    
    for root, dirs, files in os.walk(data_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            
            # Check if file has a supported extension
            if any(file_path.suffix.lower() == ext for ext in extensions):
                # Get base name without extension
                base_name = file_path.stem
                
                # Store the path (use first occurrence if duplicates exist)
                if base_name not in file_index:
                    file_index[base_name] = str(file_path)
                    # Only print first 10 to keep output clean
                    if len(file_index) <= 10:
                        print(f"   Found: {file_path.relative_to(data_dir)}")
                else:
                    print(f"   ⚠️  Duplicate found: {base_name} (skipping)")
    
    print(f"✅ Found {len(file_index)} unique HSI files")
    return file_index

def find_label_file(data_dir: Path, label_filename: str) -> Path:
    """
    Recursively search for the label file in the directory tree.
    """
    print(f"🔍 Searching for label file: {label_filename}")
    
    # Search recursively
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == label_filename:
                label_path = Path(root) / file
                print(f"✅ Found label file: {label_path}")
                return label_path
    
    # Try to find any Excel or CSV file if the exact name isn't found
    print(f"⚠️  Exact label file '{label_filename}' not found. Searching for any Excel/CSV...")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.xlsx', '.xls', '.csv')):
                label_path = Path(root) / file
                print(f"✅ Found alternative label file: {label_path}")
                return label_path
    
    raise FileNotFoundError(f"No label file found in {data_dir}")

def load_hsi_file(file_path: str) -> np.ndarray:
    """
    Load HSI file from various formats.
    Handles .tiff, .tif, .npy, .npz formats.
    """
    file_path = Path(file_path)
    
    if file_path.suffix in ['.tiff', '.tif']:
        # Load with tifffile
        try:
            hsi = tiff.imread(str(file_path)).astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load TIFF file {file_path}: {e}")
    
    elif file_path.suffix == '.npy':
        hsi = np.load(file_path).astype(np.float32)
    
    elif file_path.suffix == '.npz':
        data = np.load(file_path)
        hsi = data[data.files[0]].astype(np.float32)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return hsi

def normalize_hsi_shape(hsi: np.ndarray, filename: str) -> np.ndarray:
    """
    Normalize HSI cube to (15, 15, 155) shape.
    Handles: (155, 15, 15) -> (15, 15, 155)
    Handles: (15, 15, 68) -> interpolation to 155 bands
    """
    # Case 1: Shape is (155, 15, 15) -> transpose
    if hsi.shape == (155, 15, 15):
        print(f"   Transposing {filename}: (155,15,15) -> (15,15,155)")
        return np.moveaxis(hsi, 0, -1)
    
    # Case 2: Shape is already (15, 15, 155)
    elif hsi.shape == (15, 15, 155):
        return hsi
    
    # Case 3: Shape is (15, 15, 68) -> interpolate to 155
    elif hsi.shape == (15, 15, 68):
        print(f"   Interpolating {filename}: 68 -> 155 bands")
        try:
            from scipy.interpolate import interp1d
            
            h, w, bands = hsi.shape
            original_indices = np.linspace(0, 1, bands)
            target_indices = np.linspace(0, 1, 155)
            
            interpolated = np.zeros((h, w, 155))
            for i in range(h):
                for j in range(w):
                    f = interp1d(original_indices, hsi[i, j, :],
                                 kind='linear', fill_value='extrapolate')
                    interpolated[i, j, :] = f(target_indices)
            return interpolated
        except ImportError:
            print(f"   WARNING: scipy not installed. Cannot interpolate {filename}")
            raise
    
    # Case 4: Unknown shape
    else:
        raise ValueError(f"Unexpected shape {hsi.shape} for {filename}")

def create_dataset_artifact(config):
    """
    Load all HSI files and labels, save as a compressed .npz file.
    Handles nested directory structures recursively.
    """
    # Get the data directory
    data_dir = Path(config['stages']['stage1_ingest']['raw_data_source'])
    label_filename = config['stages']['stage1_ingest']['label_filename']
    file_extension = config['stages']['stage1_ingest'].get('file_extension', '.tiff')
    
    print(f"\n📂 Data Directory: {data_dir}")
    print(f"   Label file: {label_filename}")
    print(f"   File extension: {file_extension}")
    
    # Find label file
    label_file = find_label_file(data_dir, label_filename)
    
    # Load labels
    print(f"📋 Loading labels from: {label_file}")
    if label_file.suffix in ['.xlsx', '.xls']:
        df_labels = pd.read_excel(label_file, engine='openpyxl')
    else:
        df_labels = pd.read_csv(label_file)
    
    # Rename columns (assume first two are File_Name and Label)
    if df_labels.shape[1] >= 2:
        df_labels.columns = ['File_Name', 'Label'] + list(df_labels.columns[2:])
    else:
        raise ValueError(f"Label file must have at least 2 columns. Found {df_labels.shape[1]}")
    
    print(f"   Found {len(df_labels)} entries in label file")
    
    # Find all HSI files recursively
    extensions = ['.tiff', '.tif', '.npy'] if file_extension == '.tiff' else [file_extension, '.tiff', '.tif', '.npy']
    file_index = find_all_hsi_files(data_dir, extensions)
    
    # Load each image
    images = []
    labels = []
    skipped = 0
    found_files = 0
    
    print(f"\n📥 Loading images (matching label file entries)...")
    
    for idx, row in df_labels.iterrows():
        filename = str(row['File_Name']).strip()
        label = int(row['Label']) - 1  # 1-4 -> 0-3
        
        if filename in file_index:
            file_path = file_index[filename]
            found_files += 1
            print(f"   [{idx+1}/{len(df_labels)}] Loading: {filename} (label {label})")
        else:
            print(f"   ⚠️  File not found: {filename}")
            skipped += 1
            continue
        
        try:
            # Load the HSI file
            hsi = load_hsi_file(file_path)
            
            # Normalize shape to (15, 15, 155)
            hsi = normalize_hsi_shape(hsi, filename)
            
            # Final verification
            if hsi.shape != (15, 15, 155):
                print(f"   ⚠️  Skipping {filename}: unexpected shape {hsi.shape}")
                skipped += 1
                continue
            
            images.append(hsi)
            labels.append(label)
            
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
            skipped += 1
            continue
    
    if not images:
        raise RuntimeError(f"No valid images found in {data_dir}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Version handling
    version = config['stages']['stage1_ingest'].get('dataset_version')
    if not version:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as compressed .npz
    artifact_dir = Path(config['stages']['stage1_ingest']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = artifact_dir / f"dataset_v{version}.npz"
    np.savez_compressed(output_path, images=images, labels=labels)
    
    # ============================================================
    # FIXED: Save metadata with string keys for JSON compatibility
    # ============================================================
    class_counts = dict(zip(*np.unique(labels, return_counts=True)))
    # Convert integer keys to strings
    class_counts_str = {str(k): int(v) for k, v in class_counts.items()}
    
    # Create human-readable class names
    class_names = {0: 'HNHP', 1: 'HNLP', 2: 'LNHP', 3: 'LNLP'}
    class_distribution = {class_names[int(k)]: v for k, v in class_counts_str.items()}
    
    metadata = {
        "version": version,
        "num_samples": int(len(images)),
        "class_distribution": class_distribution,  # Human-readable: {"HNHP": 50, ...}
        "class_counts": class_counts_str,          # String keys: {"0": 50, ...}
        "shape": [15, 15, 155],                    # List instead of tuple
        "source_data_dir": str(data_dir),
        "label_file": str(label_file),
        "total_labeled_files": int(len(df_labels)),
        "found_files": int(found_files),
        "skipped_files": int(skipped),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(artifact_dir / f"metadata_v{version}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Write latest version file
    with open(artifact_dir / "latest_version.txt", 'w') as f:
        f.write(version)
    
    print(f"\n✅ DATASET CREATION COMPLETE!")
    print(f"   Version: {version}")
    print(f"   Total samples loaded: {len(images)}")
    print(f"   Class distribution: {class_distribution}")
    print(f"   Found: {found_files}/{len(df_labels)} files from label list")
    print(f"   Skipped: {skipped} files")
    print(f"   Saved to: {output_path}")
    
    return version

def run_stage1(config_path: str):
    """Main orchestrator for Stage 1."""
    config = load_config(config_path)
    
    print("="*60)
    print("STAGE 1: Data Ingestion & Versioning")
    print("="*60)
    
    # Get the data directory
    data_dir = config['stages']['stage1_ingest']['raw_data_source']
    print(f"\n📍 Data Directory: {data_dir}")
    
    # Validate the directory exists
    if not Path(data_dir).exists():
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        print("   Please update the 'raw_data_source' in config.yaml")
        return False
    
    # Count files to give user confidence
    try:
        # Count .tiff files in all subdirectories
        tiff_files = list(Path(data_dir).rglob("*.tiff")) + list(Path(data_dir).rglob("*.tif"))
        print(f"   Found {len(tiff_files)} .tiff/.tif files in directory tree")
        
        # Count subdirectories
        subdirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
        print(f"   Found {len(subdirs)} subdirectories")
        
        if subdirs:
            print(f"   First few subdirectories:")
            for d in subdirs[:5]:
                print(f"     - {d.name}/")
    
    except Exception as e:
        print(f"   ⚠️  Could not scan directory: {e}")
    
    # Create dataset artifact
    print("\n📦 Creating dataset artifact...")
    try:
        version = create_dataset_artifact(config)
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print(f"✅ STAGE 1 COMPLETE! Dataset version: {version}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Data Ingestion & Versioning")
    parser.add_argument('--config', default='config/production.yaml', help='Path to YAML config file')
    parser.add_argument('--data_dir', help='Override data directory (optional)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Allow command-line override of data directory
    if args.data_dir:
        config['stages']['stage1_ingest']['raw_data_source'] = args.data_dir
        print(f"📌 Data directory overridden to: {args.data_dir}")
    
    run_stage1(args.config)