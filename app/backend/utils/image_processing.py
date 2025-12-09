"""
Image preprocessing utilities for SliceWise Backend.

This module provides image preprocessing functions for both classification
and segmentation models. Critical: preprocessing must match training!

Functions copied and adapted from main_v2.py (lines 298-330).
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================================
# Segmentation Preprocessing (Z-Score Normalization)
# ============================================================================

def preprocess_image_for_segmentation(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocess image for segmentation model.
    
    CRITICAL: Must match training preprocessing (z-score normalization)!
    The segmentation model was trained on z-score normalized images.
    
    Copied from main_v2.py (lines 298-330).
    
    Args:
        image_array: Input image array (grayscale or RGB)
    
    Returns:
        Preprocessed image array with z-score normalization
        
    Example:
        >>> image = np.random.rand(256, 256)
        >>> preprocessed = preprocess_image_for_segmentation(image)
        >>> # preprocessed will have mean≈0, std≈1
    """
    # Ensure grayscale
    if len(image_array.shape) == 3:
        image_array = image_array[:, :, 0]
    
    # Convert to float32
    image_array = image_array.astype(np.float32)
    
    # Normalize to [0, 1] first if needed
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    
    # CRITICAL: Apply z-score normalization (same as training!)
    # The model was trained on z-score normalized images
    mean = np.mean(image_array)
    std = np.std(image_array)
    if std > 0:
        image_array = (image_array - mean) / std
    else:
        image_array = image_array - mean
    
    return image_array


# ============================================================================
# Classification Preprocessing (Min-Max Normalization)
# ============================================================================

def preprocess_image_for_classification(
    image_array: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Preprocess image for classification model.
    
    Applies min-max normalization to [0, 1] range.
    
    Args:
        image_array: Input image array (grayscale or RGB)
        target_size: Optional target size for resizing (height, width)
    
    Returns:
        Preprocessed image array normalized to [0, 1]
        
    Example:
        >>> image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        >>> preprocessed = preprocess_image_for_classification(image)
        >>> assert preprocessed.min() >= 0.0 and preprocessed.max() <= 1.0
    """
    # Ensure grayscale
    if len(image_array.shape) == 3:
        image_array = image_array[:, :, 0]
    
    # Convert to float32
    image_array = image_array.astype(np.float32)
    
    # Normalize to [0, 1]
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    
    # Resize if target size specified
    if target_size is not None:
        import cv2
        image_array = cv2.resize(image_array, (target_size[1], target_size[0]))
    
    return image_array


# ============================================================================
# General Preprocessing Utilities
# ============================================================================

def ensure_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Ensure image is grayscale.
    
    If image is RGB, converts to grayscale by taking the first channel.
    
    Args:
        image_array: Input image array (grayscale or RGB)
    
    Returns:
        Grayscale image array
        
    Example:
        >>> rgb_image = np.random.rand(256, 256, 3)
        >>> gray_image = ensure_grayscale(rgb_image)
        >>> assert len(gray_image.shape) == 2
    """
    if len(image_array.shape) == 3:
        # Take first channel if RGB
        return image_array[:, :, 0]
    return image_array


def normalize_to_range(
    image_array: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Normalize image to specified range.
    
    Args:
        image_array: Input image array
        min_val: Minimum value of output range
        max_val: Maximum value of output range
    
    Returns:
        Normalized image array
        
    Example:
        >>> image = np.random.rand(256, 256) * 255
        >>> normalized = normalize_to_range(image, 0.0, 1.0)
        >>> assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    """
    # Get current min/max
    current_min = image_array.min()
    current_max = image_array.max()
    
    # Avoid division by zero
    if current_max - current_min == 0:
        return np.full_like(image_array, min_val, dtype=np.float32)
    
    # Normalize to [0, 1]
    normalized = (image_array - current_min) / (current_max - current_min)
    
    # Scale to [min_val, max_val]
    scaled = normalized * (max_val - min_val) + min_val
    
    return scaled.astype(np.float32)


def apply_zscore_normalization(image_array: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization (mean=0, std=1).
    
    Args:
        image_array: Input image array
    
    Returns:
        Z-score normalized image array
        
    Example:
        >>> image = np.random.rand(256, 256)
        >>> normalized = apply_zscore_normalization(image)
        >>> assert abs(normalized.mean()) < 1e-6  # mean ≈ 0
        >>> assert abs(normalized.std() - 1.0) < 1e-6  # std ≈ 1
    """
    mean = np.mean(image_array)
    std = np.std(image_array)
    
    if std > 0:
        return ((image_array - mean) / std).astype(np.float32)
    else:
        return (image_array - mean).astype(np.float32)


def convert_to_float32(image_array: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 dtype.
    
    Args:
        image_array: Input image array
    
    Returns:
        Image array as float32
    """
    return image_array.astype(np.float32)


def clip_to_range(
    image_array: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Clip image values to specified range.
    
    Args:
        image_array: Input image array
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clipped image array
        
    Example:
        >>> image = np.array([-0.5, 0.5, 1.5])
        >>> clipped = clip_to_range(image, 0.0, 1.0)
        >>> assert clipped.min() >= 0.0 and clipped.max() <= 1.0
    """
    return np.clip(image_array, min_val, max_val)


# ============================================================================
# Batch Processing
# ============================================================================

def preprocess_batch_for_segmentation(
    images: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Preprocess a batch of images for segmentation.
    
    Args:
        images: List of input image arrays
    
    Returns:
        List of preprocessed image arrays
        
    Example:
        >>> images = [np.random.rand(256, 256) for _ in range(5)]
        >>> preprocessed = preprocess_batch_for_segmentation(images)
        >>> assert len(preprocessed) == 5
    """
    return [preprocess_image_for_segmentation(img) for img in images]


def preprocess_batch_for_classification(
    images: list[np.ndarray],
    target_size: Optional[Tuple[int, int]] = None
) -> list[np.ndarray]:
    """
    Preprocess a batch of images for classification.
    
    Args:
        images: List of input image arrays
        target_size: Optional target size for resizing
    
    Returns:
        List of preprocessed image arrays
        
    Example:
        >>> images = [np.random.randint(0, 255, (256, 256)) for _ in range(5)]
        >>> preprocessed = preprocess_batch_for_classification(images)
        >>> assert len(preprocessed) == 5
    """
    return [preprocess_image_for_classification(img, target_size) for img in images]


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing image preprocessing utilities...")
    
    # Test 1: Segmentation preprocessing
    print("\n1. Testing segmentation preprocessing:")
    test_image = np.random.rand(256, 256) * 255
    seg_preprocessed = preprocess_image_for_segmentation(test_image)
    print(f"   Input range: [{test_image.min():.2f}, {test_image.max():.2f}]")
    print(f"   Output mean: {seg_preprocessed.mean():.6f} (should be ≈0)")
    print(f"   Output std: {seg_preprocessed.std():.6f} (should be ≈1)")
    
    # Test 2: Classification preprocessing
    print("\n2. Testing classification preprocessing:")
    cls_preprocessed = preprocess_image_for_classification(test_image)
    print(f"   Input range: [{test_image.min():.2f}, {test_image.max():.2f}]")
    print(f"   Output range: [{cls_preprocessed.min():.2f}, {cls_preprocessed.max():.2f}]")
    
    # Test 3: Grayscale conversion
    print("\n3. Testing grayscale conversion:")
    rgb_image = np.random.rand(256, 256, 3)
    gray_image = ensure_grayscale(rgb_image)
    print(f"   Input shape: {rgb_image.shape}")
    print(f"   Output shape: {gray_image.shape}")
    
    # Test 4: Z-score normalization
    print("\n4. Testing z-score normalization:")
    zscore_normalized = apply_zscore_normalization(test_image)
    print(f"   Mean: {zscore_normalized.mean():.6f}")
    print(f"   Std: {zscore_normalized.std():.6f}")
    
    print("\n✅ All tests passed!")
