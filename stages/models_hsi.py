# stages/models_hsi.py
"""
HSI-Optimized CNN Architectures for 15x15x155 patches.
These models are specifically designed for HSI data.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def HSI_CNN(input_shape=(15, 15, 155), num_classes=4):
    """
    HSI-Optimized CNN with 1D spectral processing + 2D spatial processing.
    
    Strategy:
    1. Process each spectral band independently with 1x1 convs
    2. Then process spatial features
    3. No aggressive pooling to preserve 15x15
    """
    inputs = layers.Input(shape=input_shape)
    
    # Spectral feature extraction (per-band processing)
    x = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Spatial feature extraction with depthwise separable
    # Using depthwise convolutions to reduce parameters
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Second block
    x = layers.Conv2D(128, (1, 1), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Third block - more filters
    x = layers.Conv2D(256, (1, 1), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Global pooling to get feature vectors
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)


def HSI_ResNet(input_shape=(15, 15, 155), num_classes=4):
    """
    HSI ResNet with skip connections.
    Better gradient flow for deeper networks.
    """
    def residual_block(x, filters, kernel_size=3):
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, (1, 1), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.DepthwiseConv2D((kernel_size, kernel_size), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (1, 1), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Skip connection (adjust dimensions if needed)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1))(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial spectral reduction
    x = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)


def SpectralSpatialCNN(input_shape=(15, 15, 155), num_classes=4):
    """
    Original Spectral-Spatial CNN (optimized version)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Spectral reduction
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Spatial features
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)