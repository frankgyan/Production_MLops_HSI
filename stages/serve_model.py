# serve_final.py
"""
Final API Server - Supports both .npy and .tiff files with automatic preprocessing
"""

import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from pathlib import Path
import io
import os
import tempfile

# ============================================================
# CONFIGURATION
# ============================================================

class_names = ['HNHP', 'HNLP', 'LNHP', 'LNLP']

# Try ALL possible model locations
MODEL_CANDIDATES = [
    Path("artifacts/deployment_v2/deployed_model.keras"),
    Path("artifacts/deployment/deployed_model.keras"),
    Path("artifacts/model/best_model.keras"),
]

# Try to load PCA model
PCA_PATH = Path("artifacts/preprocessed/pca_model.joblib")

app = FastAPI(
    title="Quinoa HSI Classification API",
    description="90% Accuracy Model - Supports .npy and .tiff files",
    version="2.0.0",
    docs_url="/docs"
)

model = None
model_path = None
model_accuracy = 0.9008
pca_model = None

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def snv_normalize(image):
    """Standard Normal Variate per pixel spectrum."""
    mean = np.mean(image, axis=2, keepdims=True)
    std = np.std(image, axis=2, keepdims=True) + 1e-8
    return (image - mean) / std

def preprocess_tiff(image, pca_model):
    """
    Preprocess raw TIFF: SNV + PCA
    """
    # Step 1: SNV Normalization
    image_snv = snv_normalize(image)
    
    # Step 2: Apply PCA
    h, w, bands = image_snv.shape
    flat = image_snv.reshape(-1, bands)
    reduced = pca_model.transform(flat)
    image_pca = reduced.reshape(h, w, -1)
    
    return image_pca

# ============================================================
# LOAD MODEL AND PCA
# ============================================================

@app.on_event("startup")
async def load_artifacts():
    global model, model_path, pca_model
    
    print("="*60)
    print("STARTING HSI API SERVER (V2 - 90% MODEL)")
    print("="*60)
    
    # Load PCA model for TIFF preprocessing
    if PCA_PATH.exists():
        try:
            import joblib
            pca_model = joblib.load(PCA_PATH)
            print(f"[SUCCESS] PCA model loaded! Components: {pca_model.n_components_}")
        except Exception as e:
            print(f"[WARNING] Failed to load PCA: {e}")
            pca_model = None
    else:
        print("[INFO] PCA model not found. TIFF files will not work.")
        pca_model = None
    
    # Load ML model
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            print(f"[INFO] Found model at: {candidate}")
            try:
                model = tf.keras.models.load_model(candidate)
                model_path = candidate
                print("[SUCCESS] Model loaded successfully!")
                print(f"[INFO] Input shape: {model.input_shape}")
                print("="*60)
                print("[SUCCESS] API Server Ready!")
                print(f"[INFO] Docs: http://localhost:8000/docs")
                print(f"[INFO] Health: http://localhost:8000/health")
                print("[INFO] Supported formats: .npy, .tiff, .tif")
                print("="*60)
                return
            except Exception as e:
                print(f"[ERROR] Failed to load from {candidate}: {e}")
                continue
    
    print("[ERROR] No model found!")
    model = None

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "Quinoa HSI Classification API",
        "version": "2.0.0",
        "status": "running" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": str(model_path) if model_path else None,
        "accuracy": model_accuracy if model is not None else 0,
        "supported_formats": [".npy", ".tiff", ".tif"],
        "endpoints": {
            "/": "API Info",
            "/health": "Health Check",
            "/predict": "Predict (.npy or .tiff)",
            "/docs": "Interactive Docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": str(model_path) if model_path else None,
        "accuracy": model_accuracy if model is not None else 0,
        "pca_loaded": pca_model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Make a prediction on an HSI file.
    
    Supported formats:
    - .npy: Preprocessed data (15, 15, 20) - direct prediction
    - .tiff / .tif: Raw data (15, 15, 155) - automatic preprocessing
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        contents = await file.read()
        filename = file.filename.lower()
        print(f"[INFO] Processing: {filename}")
        
        # ============================================================
        # Handle .npy files (already preprocessed)
        # ============================================================
        if filename.endswith('.npy'):
            print("[INFO] Detected .npy format - direct prediction")
            with io.BytesIO(contents) as f:
                data = np.load(f)
            
            print(f"[INFO] Data shape: {data.shape}")
            
            if len(data.shape) == 3:
                data = np.expand_dims(data, axis=0)
            elif len(data.shape) != 4:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected 3D array (15, 15, 20), got {data.shape}"
                )
        
        # ============================================================
        # Handle .tiff / .tif files (need preprocessing)
        # ============================================================
        elif filename.endswith('.tiff') or filename.endswith('.tif'):
            print("[INFO] Detected .tiff format - applying preprocessing")
            
            if pca_model is None:
                raise HTTPException(
                    status_code=400,
                    detail="PCA model not available. Cannot process TIFF files."
                )
            
            try:
                import tifffile
                with io.BytesIO(contents) as f:
                    data = tifffile.imread(f)
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="tifffile not installed. Install with: pip install tifffile"
                )
            
            print(f"[INFO] Raw TIFF shape: {data.shape}")
            
            # Handle different shapes
            if len(data.shape) == 3:
                # Check if shape is (155, 15, 15) -> transpose
                if data.shape[0] == 155 and data.shape[1] == 15 and data.shape[2] == 15:
                    print("[INFO] Transposing (155,15,15) -> (15,15,155)")
                    data = np.moveaxis(data, 0, -1)
                
                # Check if shape is (15, 15, 68) -> interpolate to 155
                if data.shape[2] == 68:
                    print("[INFO] Interpolating 68 -> 155 bands")
                    try:
                        from scipy.interpolate import interp1d
                        h, w, bands = data.shape
                        original_indices = np.linspace(0, 1, bands)
                        target_indices = np.linspace(0, 1, 155)
                        interpolated = np.zeros((h, w, 155))
                        for i in range(h):
                            for j in range(w):
                                f = interp1d(original_indices, data[i, j, :],
                                            kind='linear', fill_value='extrapolate')
                                interpolated[i, j, :] = f(target_indices)
                        data = interpolated
                        print("[INFO] Interpolation complete")
                    except ImportError:
                        raise HTTPException(
                            status_code=400,
                            detail="scipy not installed. Cannot interpolate 68 bands."
                        )
                
                # Check if already 155 bands
                if data.shape[2] != 155:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Expected 155 bands, got {data.shape[2]}. Shape: {data.shape}"
                    )
                
                print("[INFO] Applying SNV + PCA preprocessing...")
                data = preprocess_tiff(data, pca_model)
                print(f"[INFO] Preprocessed shape: {data.shape}")
                
                # Add batch dimension
                data = np.expand_dims(data, axis=0)
            
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected 3D array, got shape {data.shape}"
                )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {filename}. Use .npy, .tiff, or .tif"
            )
        
        # ============================================================
        # Make Prediction
        # ============================================================
        predictions = model.predict(data, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        probabilities = {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }
        
        return {
            "prediction": class_names[predicted_class],
            "class_id": int(predicted_class),
            "confidence": confidence,
            "probabilities": probabilities,
            "file_format": filename.split('.')[-1]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "serve_model:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )