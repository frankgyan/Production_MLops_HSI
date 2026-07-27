# stages/stage2_preprocess.py
"""
STAGE 2: Advanced HSI Preprocessing with Vegetation Indices + PCA
"""

import os
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import signal
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# 1. VEGETATION INDICES FOR HSI
# ============================================================

def calculate_vegetation_indices(spectral_bands, band_indices):
    """
    Calculate common vegetation indices from HSI spectral bands.
    
    Args:
        spectral_bands: HSI cube (15, 15, 155)
        band_indices: Dict mapping band names to their index positions
    
    Returns:
        Dictionary of vegetation indices
    """
    # Extract specific bands (assuming 155-band HSI with ~400-1000nm range)
    # You'll need to adjust these based on your actual band wavelengths
    R = spectral_bands[:, :, band_indices.get('red', 50)]      # ~680nm
    NIR = spectral_bands[:, :, band_indices.get('nir', 100)]   # ~800nm
    G = spectral_bands[:, :, band_indices.get('green', 30)]    # ~550nm
    B = spectral_bands[:, :, band_indices.get('blue', 20)]     # ~450nm
    
    indices = {}
    
    # 1. NDVI (Normalized Difference Vegetation Index)
    # Range: -1 to 1, Healthy vegetation: 0.8-1.0
    ndvi = (NIR - R) / (NIR + R + 1e-8)
    indices['NDVI'] = ndvi
    
    # 2. NDWI (Normalized Difference Water Index)
    # Range: -1 to 1, Water content
    ndwi = (G - NIR) / (G + NIR + 1e-8)
    indices['NDWI'] = ndwi
    
    # 3. EVI (Enhanced Vegetation Index)
    # Range: -1 to 1, Better for dense vegetation
    evi = 2.5 * ((NIR - R) / (NIR + 6 * R - 7.5 * B + 1))
    indices['EVI'] = evi
    
    # 4. SAVI (Soil Adjusted Vegetation Index)
    # L=0.5 for moderate vegetation cover
    L = 0.5
    savi = ((NIR - R) / (NIR + R + L)) * (1 + L)
    indices['SAVI'] = savi
    
    # 5. GNDVI (Green NDVI)
    # More sensitive to chlorophyll content
    gndvi = (NIR - G) / (NIR + G + 1e-8)
    indices['GNDVI'] = gndvi
    
    # 6. Chlorophyll Index (CI)
    # Ratio of NIR to green
    ci = NIR / (G + 1e-8)
    indices['CI'] = ci
    
    # 7. Anthocyanin Reflectance Index (ARI)
    # Uses green/red ratio for anthocyanin
    ari = G / (R + 1e-8)
    indices['ARI'] = ari
    
    # 8. Pigment Specific Simple Ratio (PSSR)
    # Chlorophyll a and b
    pssr = NIR / (R + 1e-8)
    indices['PSSR'] = pssr
    
    # 9. Normalized Difference Nitrogen Index (NDNI)
    # Nitrogen content proxy
    # Using bands at ~1510nm and ~1680nm if available
    try:
        band1510 = spectral_bands[:, :, band_indices.get('1510nm', 130)]
        band1680 = spectral_bands[:, :, band_indices.get('1680nm', 145)]
        ndni = np.log(1 / band1510) - np.log(1 / band1680)
        indices['NDNI'] = ndni
    except:
        pass
    
    # 10. Red Edge Position (REP) approximation
    # Using bands around 700-740nm
    try:
        red_edge1 = spectral_bands[:, :, band_indices.get('red_edge1', 90)]
        red_edge2 = spectral_bands[:, :, band_indices.get('red_edge2', 95)]
        rep = (red_edge1 + red_edge2) / 2
        indices['REP'] = rep
    except:
        pass
    
    return indices


def extract_vegetation_indices_as_features(images, band_indices):
    """
    Extract vegetation indices as spatial features.
    
    Args:
        images: HSI cubes (n_samples, 15, 15, 155)
        band_indices: Dict mapping band names to indices
    
    Returns:
        Feature maps (n_samples, 15, 15, n_indices)
    """
    print(f"   Computing vegetation indices for {len(images)} samples...")
    
    n_samples = len(images)
    h, w = images.shape[1], images.shape[2]
    
    # Calculate indices for all samples
    all_indices = []
    
    for i, img in enumerate(images):
        indices = calculate_vegetation_indices(img, band_indices)
        
        # Stack all indices as feature maps
        index_stack = []
        for idx_name, idx_map in indices.items():
            index_stack.append(idx_map)
        
        # Shape: (15, 15, n_indices)
        index_features = np.stack(index_stack, axis=-1)
        all_indices.append(index_features)
        
        if (i + 1) % 500 == 0:
            print(f"   Processed {i+1}/{n_samples} samples")
    
    return np.array(all_indices)


# ============================================================
# 2. PREPROCESSING FUNCTIONS
# ============================================================

def snv_normalize(image):
    """Standard Normal Variate per pixel spectrum."""
    mean = np.mean(image, axis=2, keepdims=True)
    std = np.std(image, axis=2, keepdims=True) + 1e-8
    return (image - mean) / std


def savitzky_golay_filter(image, window_length=9, polyorder=3):
    """Savitzky-Golay smoothing."""
    h, w, bands = image.shape
    filtered = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            filtered[i, j, :] = signal.savgol_filter(
                image[i, j, :], 
                window_length=window_length, 
                polyorder=polyorder
            )
    return filtered


def pca_reduce_spectral(images, n_components=20):
    """PCA reduction across spectral dimension."""
    n_samples, h, w, bands = images.shape
    flat = images.reshape(-1, bands)
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(flat)
    
    new_shape = (n_samples, h, w, n_components)
    return reduced.reshape(new_shape), pca


# ============================================================
# 3. MAIN FUNCTION
# ============================================================

def run_stage2(config_path):
    config = load_config(config_path)
    
    print("="*70)
    print("STAGE 2: HSI Preprocessing with Vegetation Indices + PCA")
    print("="*70)
    
    # Read version
    stage1_artifact_dir = Path(config['stages']['stage1_ingest']['artifact_dir'])
    version_file = stage1_artifact_dir / "latest_version.txt"
    
    if not version_file.exists():
        print("❌ No version file found.")
        return False
    
    with open(version_file, 'r') as f:
        version = f.read().strip()
    
    print(f"\n📦 Loading dataset version: {version}")
    
    dataset_path = stage1_artifact_dir / f"dataset_v{version}.npz"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    data = np.load(dataset_path)
    images = data['images']
    labels = data['labels']
    
    print(f"   Loaded {len(images)} samples")
    print(f"   Original image shape: {images[0].shape}")
    
    # ============================================================
    # 3. BAND INDICES (adjust based on your sensor)
    # ============================================================
    
    # For a 155-band HSI (adjust these indices based on your actual wavelengths)
    # You may need to calibrate these with your specific band centers
    band_indices = {
        'blue': 20,      # ~450nm
        'green': 30,     # ~550nm
        'red': 50,       # ~680nm
        'nir': 100,      # ~800nm
        'red_edge1': 85, # ~700nm
        'red_edge2': 92, # ~730nm
        '1510nm': 130,   # ~1510nm (if available)
        '1680nm': 145,   # ~1680nm (if available)
    }
    
    # ============================================================
    # 4. APPLY PREPROCESSING
    # ============================================================
    
    preprocessing_config = config['stages']['stage2_preprocess']['preprocessing']
    
    print("\n" + "="*70)
    print("🔄 APPLYING PREPROCESSING")
    print("="*70)
    
    processed_images = images.copy()
    enabled_steps = []
    
    # Step 1: Savitzky-Golay Smoothing
    if preprocessing_config.get('savitzky_golay', {}).get('enabled', False):
        print("\n🔊 Step 1: Savitzky-Golay Smoothing...")
        sg_params = preprocessing_config['savitzky_golay']
        processed_images = np.array([
            savitzky_golay_filter(img, 
                                  window_length=sg_params.get('window_length', 9),
                                  polyorder=sg_params.get('polyorder', 3))
            for img in processed_images
        ])
        print(f"   ✓ Smoothing complete")
        enabled_steps.append("Savitzky-Golay")
    
    # Step 2: SNV Normalization
    if preprocessing_config.get('snv', {}).get('enabled', False):
        print("\n🔊 Step 2: SNV Normalization...")
        processed_images = np.array([
            snv_normalize(img)
            for img in processed_images
        ])
        print(f"   ✓ SNV normalization complete")
        enabled_steps.append("SNV")
    
    # Step 3: Vegetation Indices (NEW)
    if preprocessing_config.get('vegetation_indices', {}).get('enabled', False):
        print("\n🔊 Step 3: Computing Vegetation Indices...")
        vi_features = extract_vegetation_indices_as_features(processed_images, band_indices)
        print(f"   ✓ Vegetation indices computed: {vi_features.shape[3]} indices")
        enabled_steps.append("VegetationIndices")
    else:
        vi_features = None
    
    # Step 4: PCA Dimensionality Reduction
    if preprocessing_config.get('pca', {}).get('enabled', False):
        print("\n🔊 Step 4: PCA Dimensionality Reduction...")
        pca_params = preprocessing_config['pca']
        n_components = pca_params.get('n_components', 20)
        
        spectral_features, pca_model = pca_reduce_spectral(processed_images, n_components)
        
        # Save PCA model
        pca_dir = Path(config['stages']['stage2_preprocess']['artifact_dir'])
        pca_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(pca_model, pca_dir / 'pca_model.joblib')
        
        print(f"   ✓ PCA reduction: 155 -> {n_components} bands")
        enabled_steps.append(f"PCA({n_components})")
    else:
        spectral_features = processed_images
    
    # Step 5: Combine PCA + Vegetation Indices (NEW)
    if vi_features is not None and spectral_features is not None:
        print("\n🔊 Step 5: Combining PCA and Vegetation Indices...")
        combined_features = np.concatenate([spectral_features, vi_features], axis=-1)
        print(f"   ✓ Combined features shape: {combined_features.shape[1:]} (PCA + {vi_features.shape[3]} VI)")
        enabled_steps.append(f"Combined(PCA{combined_features.shape[3]-vi_features.shape[3]}+VI{vi_features.shape[3]})")
        final_features = combined_features
    elif vi_features is not None:
        final_features = vi_features
    else:
        final_features = spectral_features
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Final image shape: {final_features.shape[1:]}")
    print(f"   Enabled steps: {', '.join(enabled_steps)}")

    # ============================================================
    # 5. DATA SPLITTING
    # ============================================================
    
    test_size = config['stages']['stage2_preprocess']['test_size']
    val_size = config['stages']['stage2_preprocess']['val_size']
    random_state = config['stages']['stage2_preprocess']['random_state']
    
    print(f"\n📊 Splitting data:")
    print(f"   Test size: {test_size*100:.0f}%")
    print(f"   Validation size: {val_size*100:.0f}%")
    
    # First split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        final_features, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=random_state
    )
    
    # Save splits
    artifact_dir = Path(config['stages']['stage2_preprocess']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    split_paths = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    print("\n💾 Saving splits...")
    for name, data in split_paths.items():
        np.save(artifact_dir / f"{name}.npy", data)
        print(f"   Saved: {name}.npy ({len(data)} samples)")
    
    # Save metadata
    class_names = {0: 'HNHP', 1: 'HNLP', 2: 'LNHP', 3: 'LNLP'}
    
    split_metadata = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "final_image_shape": list(final_features.shape[1:]),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "n_vegetation_indices": vi_features.shape[3] if vi_features is not None else 0,
        "n_pca_components": n_components if preprocessing_config.get('pca', {}).get('enabled', False) else 155,
        "class_distribution": {
            "train": {class_names[int(k)]: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "val": {class_names[int(k)]: int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            "test": {class_names[int(k)]: int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
        },
        "preprocessing_steps": enabled_steps
    }
    
    with open(artifact_dir / "split_metadata.json", 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    print(f"\n" + "="*70)
    print(f"✅ STAGE 2 COMPLETE!")
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"   Final shape: {final_features.shape[1:]}")
    print(f"   Features: {final_features.shape[3]} (PCA + Vegetation Indices)")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml')
    args = parser.parse_args()
    
    run_stage2(args.config)