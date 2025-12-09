"""
Utility functions for image processing, visualization, and validation.

This module contains reusable utility functions used across the backend.
"""

# Image processing utilities
from .image_processing import (
    preprocess_image_for_segmentation,
    preprocess_image_for_classification,
    ensure_grayscale,
    normalize_to_range,
    apply_zscore_normalization,
    preprocess_batch_for_segmentation,
    preprocess_batch_for_classification,
)

# Visualization utilities
from .visualization import (
    numpy_to_base64_png,
    create_overlay,
    create_tumor_overlay,
    create_gradcam_overlay,
    create_uncertainty_overlay,
    apply_colormap,
)

# Validation utilities
from .validators import (
    validate_image_format,
    validate_file_upload,
    validate_batch_size,
    validate_files_list,
    validate_threshold,
    validate_alpha,
    validate_image_array,
    validate_positive_integer,
    validate_string_choice,
    validate_patient_id,
    # Exceptions
    ValidationError,
    InvalidImageFormatError,
    InvalidBatchSizeError,
    InvalidThresholdError,
    InvalidFileUploadError,
)

__all__ = [
    # Image processing
    "preprocess_image_for_segmentation",
    "preprocess_image_for_classification",
    "ensure_grayscale",
    "normalize_to_range",
    "apply_zscore_normalization",
    "preprocess_batch_for_segmentation",
    "preprocess_batch_for_classification",
    # Visualization
    "numpy_to_base64_png",
    "create_overlay",
    "create_tumor_overlay",
    "create_gradcam_overlay",
    "create_uncertainty_overlay",
    "apply_colormap",
    # Validation
    "validate_image_format",
    "validate_file_upload",
    "validate_batch_size",
    "validate_files_list",
    "validate_threshold",
    "validate_alpha",
    "validate_image_array",
    "validate_positive_integer",
    "validate_string_choice",
    "validate_patient_id",
    # Exceptions
    "ValidationError",
    "InvalidImageFormatError",
    "InvalidBatchSizeError",
    "InvalidThresholdError",
    "InvalidFileUploadError",
]
