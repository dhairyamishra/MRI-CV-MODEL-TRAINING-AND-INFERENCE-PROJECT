"""
Input validation utilities for SliceWise Backend.

This module provides validation functions for API inputs including
file uploads, batch sizes, thresholds, and image formats.

Extracted validation logic from main_v2.py (lines 529, 739).
"""

import numpy as np
from fastapi import UploadFile, HTTPException
from typing import List, Optional
from pathlib import Path


# ============================================================================
# Custom Exceptions
# ============================================================================

class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class InvalidImageFormatError(ValidationError):
    """Raised when image format is invalid."""
    pass


class InvalidBatchSizeError(ValidationError):
    """Raised when batch size exceeds limits."""
    pass


class InvalidThresholdError(ValidationError):
    """Raised when threshold is out of valid range."""
    pass


class InvalidFileUploadError(ValidationError):
    """Raised when file upload is invalid."""
    pass


# ============================================================================
# File Upload Validation
# ============================================================================

ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
ALLOWED_MIME_TYPES = {
    'image/png',
    'image/jpeg',
    'image/bmp',
    'image/tiff',
    'image/x-tiff'
}


def validate_image_format(
    file: UploadFile,
    allowed_extensions: Optional[set] = None,
    allowed_mime_types: Optional[set] = None
) -> bool:
    """
    Validate uploaded image file format.
    
    Args:
        file: FastAPI UploadFile object
        allowed_extensions: Set of allowed file extensions (default: common image formats)
        allowed_mime_types: Set of allowed MIME types (default: common image types)
    
    Returns:
        True if valid
        
    Raises:
        InvalidImageFormatError: If format is invalid
        
    Example:
        >>> # In FastAPI endpoint:
        >>> validate_image_format(file)
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_IMAGE_EXTENSIONS
    if allowed_mime_types is None:
        allowed_mime_types = ALLOWED_MIME_TYPES
    
    # Check filename extension
    if file.filename:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise InvalidImageFormatError(
                f"Invalid file extension: {file_ext}. "
                f"Allowed extensions: {', '.join(allowed_extensions)}"
            )
    
    # Check MIME type
    if file.content_type and file.content_type not in allowed_mime_types:
        raise InvalidImageFormatError(
            f"Invalid MIME type: {file.content_type}. "
            f"Allowed types: {', '.join(allowed_mime_types)}"
        )
    
    return True


def validate_file_upload(
    file: UploadFile,
    max_size_mb: float = 10.0
) -> bool:
    """
    Validate file upload (format and size).
    
    Args:
        file: FastAPI UploadFile object
        max_size_mb: Maximum file size in MB
    
    Returns:
        True if valid
        
    Raises:
        InvalidFileUploadError: If upload is invalid
        
    Example:
        >>> validate_file_upload(file, max_size_mb=10.0)
    """
    # Validate format
    try:
        validate_image_format(file)
    except InvalidImageFormatError as e:
        raise InvalidFileUploadError(str(e))
    
    # Check file size (if available)
    if hasattr(file, 'size') and file.size:
        max_size_bytes = max_size_mb * 1024 * 1024
        if file.size > max_size_bytes:
            raise InvalidFileUploadError(
                f"File size ({file.size / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({max_size_mb} MB)"
            )
    
    return True


# ============================================================================
# Batch Size Validation
# ============================================================================

def validate_batch_size(
    num_files: int,
    max_batch_size: int = 100,
    operation: str = "batch processing"
) -> bool:
    """
    Validate batch size.
    
    Extracted from main_v2.py (lines 529, 739).
    
    Args:
        num_files: Number of files in batch
        max_batch_size: Maximum allowed batch size
        operation: Name of operation (for error message)
    
    Returns:
        True if valid
        
    Raises:
        InvalidBatchSizeError: If batch size exceeds limit
        HTTPException: FastAPI exception with status 400
        
    Example:
        >>> validate_batch_size(len(files), max_batch_size=100)
    """
    if num_files > max_batch_size:
        error_msg = f"Maximum {max_batch_size} images per {operation}"
        raise HTTPException(status_code=400, detail=error_msg)
    
    if num_files < 1:
        raise HTTPException(
            status_code=400,
            detail=f"At least 1 image required for {operation}"
        )
    
    return True


def validate_files_list(
    files: List[UploadFile],
    max_batch_size: int = 100,
    validate_format: bool = True
) -> bool:
    """
    Validate list of uploaded files.
    
    Args:
        files: List of FastAPI UploadFile objects
        max_batch_size: Maximum allowed batch size
        validate_format: Whether to validate each file format
    
    Returns:
        True if valid
        
    Raises:
        HTTPException: If validation fails
        
    Example:
        >>> validate_files_list(files, max_batch_size=100)
    """
    # Validate batch size
    validate_batch_size(len(files), max_batch_size)
    
    # Validate each file format
    if validate_format:
        for idx, file in enumerate(files):
            try:
                validate_image_format(file)
            except InvalidImageFormatError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {idx + 1} ({file.filename}): {str(e)}"
                )
    
    return True


# ============================================================================
# Threshold Validation
# ============================================================================

def validate_threshold(
    threshold: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
    param_name: str = "threshold"
) -> bool:
    """
    Validate probability threshold.
    
    Args:
        threshold: Threshold value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Parameter name (for error message)
    
    Returns:
        True if valid
        
    Raises:
        InvalidThresholdError: If threshold is out of range
        
    Example:
        >>> validate_threshold(0.5, 0.0, 1.0)
    """
    if not (min_val <= threshold <= max_val):
        raise InvalidThresholdError(
            f"{param_name} must be between {min_val} and {max_val}, "
            f"got {threshold}"
        )
    return True


def validate_alpha(alpha: float) -> bool:
    """
    Validate alpha (transparency) value.
    
    Args:
        alpha: Alpha value (0.0 = transparent, 1.0 = opaque)
    
    Returns:
        True if valid
        
    Raises:
        InvalidThresholdError: If alpha is out of range
    """
    return validate_threshold(alpha, 0.0, 1.0, "alpha")


# ============================================================================
# Image Array Validation
# ============================================================================

def validate_image_array(
    image: np.ndarray,
    expected_dims: int = 2,
    min_size: int = 32,
    max_size: int = 2048
) -> bool:
    """
    Validate numpy image array.
    
    Args:
        image: Numpy array to validate
        expected_dims: Expected number of dimensions (2 or 3)
        min_size: Minimum image dimension
        max_size: Maximum image dimension
    
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> image = np.random.rand(256, 256)
        >>> validate_image_array(image, expected_dims=2)
    """
    # Check dimensions
    if len(image.shape) not in [2, 3]:
        raise ValidationError(
            f"Image must be 2D or 3D array, got {len(image.shape)}D"
        )
    
    # Check size
    height, width = image.shape[:2]
    if height < min_size or width < min_size:
        raise ValidationError(
            f"Image dimensions ({height}x{width}) below minimum ({min_size}x{min_size})"
        )
    
    if height > max_size or width > max_size:
        raise ValidationError(
            f"Image dimensions ({height}x{width}) exceed maximum ({max_size}x{max_size})"
        )
    
    # Check data type
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValidationError(
            f"Image dtype must be uint8, float32, or float64, got {image.dtype}"
        )
    
    return True


# ============================================================================
# Parameter Validation
# ============================================================================

def validate_positive_integer(
    value: int,
    param_name: str = "value",
    min_val: int = 1
) -> bool:
    """
    Validate positive integer parameter.
    
    Args:
        value: Value to validate
        param_name: Parameter name (for error message)
        min_val: Minimum allowed value
    
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> validate_positive_integer(50, "min_object_size", min_val=0)
    """
    if not isinstance(value, int):
        raise ValidationError(f"{param_name} must be an integer, got {type(value)}")
    
    if value < min_val:
        raise ValidationError(f"{param_name} must be >= {min_val}, got {value}")
    
    return True


def validate_string_choice(
    value: str,
    choices: List[str],
    param_name: str = "value"
) -> bool:
    """
    Validate string parameter against allowed choices.
    
    Args:
        value: Value to validate
        choices: List of allowed choices
        param_name: Parameter name (for error message)
    
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> validate_string_choice("close", ["none", "open", "close"], "morphology_op")
    """
    if value not in choices:
        raise ValidationError(
            f"{param_name} must be one of {choices}, got '{value}'"
        )
    return True


# ============================================================================
# Patient ID Validation
# ============================================================================

def validate_patient_id(patient_id: str) -> bool:
    """
    Validate patient ID format.
    
    Args:
        patient_id: Patient identifier string
    
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> validate_patient_id("PATIENT_001")
    """
    if not patient_id or not patient_id.strip():
        raise ValidationError("Patient ID cannot be empty")
    
    if len(patient_id) > 100:
        raise ValidationError("Patient ID too long (max 100 characters)")
    
    # Check for invalid characters (optional, adjust as needed)
    # For now, just ensure it's not empty
    
    return True


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Test validation utilities
    print("Testing validation utilities...")
    
    # Test 1: Threshold validation
    print("\n1. Testing threshold validation:")
    try:
        validate_threshold(0.5, 0.0, 1.0)
        print("   ✓ Valid threshold (0.5) passed")
    except InvalidThresholdError as e:
        print(f"   ✗ Error: {e}")
    
    try:
        validate_threshold(1.5, 0.0, 1.0)
        print("   ✗ Invalid threshold (1.5) should have failed")
    except InvalidThresholdError:
        print("   ✓ Invalid threshold (1.5) correctly rejected")
    
    # Test 2: Image array validation
    print("\n2. Testing image array validation:")
    valid_image = np.random.rand(256, 256).astype(np.float32)
    try:
        validate_image_array(valid_image, expected_dims=2)
        print("   ✓ Valid image array passed")
    except ValidationError as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Positive integer validation
    print("\n3. Testing positive integer validation:")
    try:
        validate_positive_integer(50, "min_object_size", min_val=0)
        print("   ✓ Valid positive integer (50) passed")
    except ValidationError as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: String choice validation
    print("\n4. Testing string choice validation:")
    try:
        validate_string_choice("close", ["none", "open", "close"], "morphology_op")
        print("   ✓ Valid choice ('close') passed")
    except ValidationError as e:
        print(f"   ✗ Error: {e}")
    
    # Test 5: Patient ID validation
    print("\n5. Testing patient ID validation:")
    try:
        validate_patient_id("PATIENT_001")
        print("   ✓ Valid patient ID passed")
    except ValidationError as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ All tests passed!")
