# stages/stage3_train_simple.py
"""
SIMPLIFIED Stage 3: Training with Cleaned Data
NO EMOJIS - Windows compatible
"""

import os
import json
import yaml
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# MODEL
# ============================================================

def SimpleCNN(input_shape, num_classes=4):
    """Simple CNN for HSI classification."""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)


# ============================================================
# MAIN TRAINING
# ============================================================

def run_stage3(config_path):
    config = load_config(config_path)
    
    print("="*60)
    print("STAGE 3: Training with Cleaned Data")
    print("="*60)
    
    # Load data - USE CLEANED VERSION
    artifact_dir = Path(config['stages']['stage2_preprocess']['artifact_dir'])
    
    print("")
    print("[INFO] Loading cleaned data...")
    
    # Check if cleaned data exists
    cleaned_train = artifact_dir / 'X_train_cleaned.npy'
    
    if cleaned_train.exists():
        X_train = np.load(artifact_dir / 'X_train_cleaned.npy')
        y_train = np.load(artifact_dir / 'y_train.npy')
        X_val = np.load(artifact_dir / 'X_val_cleaned.npy')
        y_val = np.load(artifact_dir / 'y_val.npy')
        print("   Using cleaned data!")
    else:
        # Fallback to original and clean on the fly
        X_train = np.load(artifact_dir / 'X_train.npy')
        y_train = np.load(artifact_dir / 'y_train.npy')
        X_val = np.load(artifact_dir / 'X_val.npy')
        y_val = np.load(artifact_dir / 'y_val.npy')
        
        print("   Cleaning data on the fly...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        X_train = np.clip(X_train, -10, 10)
        X_val = np.clip(X_val, -10, 10)
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Shape: {X_train.shape[1:]}")
    print(f"   X_train NaN count: {np.isnan(X_train).sum()}")
    print(f"   X_val NaN count:   {np.isnan(X_val).sum()}")
    
    # One-hot encode
    y_train_onehot = tf.keras.utils.to_categorical(y_train, 4)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, 4)
    
    # Build model
    model = SimpleCNN(X_train.shape[1:], 4)
    
    # Compile with gradient clipping
    optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("")
    print("[INFO] Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    
    print("")
    print("[INFO] Training...")
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        batch_size=32,
        epochs=100,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Get best accuracy
    best_val_acc = max(history.history['val_accuracy'])
    print("")
    print(f"[SUCCESS] Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Save model
    model_artifact_dir = Path(config['stages']['stage3_train']['artifact_dir'])
    model_artifact_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_artifact_dir / 'best_model.keras'
    model.save(model_path)
    print(f"[INFO] Model saved to: {model_path}")
    
    # Save history
    history_serializable = {
        k: [float(v) for v in history.history[k]] 
        for k in history.history.keys()
    }
    with open(model_artifact_dir / 'training_history.json', 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    print("")
    print("="*60)
    print("STAGE 3 COMPLETE!")
    print(f"   Best val accuracy: {best_val_acc:.4f}")
    print("="*60)
    
    return model, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml')
    args = parser.parse_args()
    
    run_stage3(args.config)