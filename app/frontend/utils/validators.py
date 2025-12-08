"""
Validation Utilities for SliceWise Frontend.

This module provides validation functions for file uploads,
user inputs, and data integrity checks.
"""

from typing import Optional, Tuple, List
from pathlib import Path
from PIL import Image
import streamlit as st

# Import settings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import UIConfig


# ============================================================================
# File Upload Validation
# ============================================================================

def validate_file_upload(
    uploaded_file,
    max_size_mb: float = UIConfig.MAX_FILE_SIZE_MB,
    allowed_formats: List[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file for size and format.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        max_size_mb: Maximum file size in MB
        allowed_formats: List of allowed file extensions
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_file_upload(uploaded_file)
        >>> if not is_valid:
        >>>     st.error(error)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Use default allowed formats if not specified
    if allowed_formats is None:
        allowed_formats = UIConfig.ALLOWED_IMAGE_FORMATS
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large ({file_size_mb:.2f} MB). Maximum size: {max_size_mb} MB"
    
    # Check file extension
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext not in allowed_formats:
        return False, f"Invalid file format '.{file_ext}'. Allowed formats: {', '.join(allowed_formats)}"
    
    return True, None


def validate_batch_upload(
    uploaded_files: List,
    max_batch_size: int = UIConfig.MAX_BATCH_SIZE,
    max_size_mb: float = UIConfig.MAX_FILE_SIZE_MB
) -> Tuple[bool, Optional[str], List]:
    """
    Validate batch file upload.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        max_batch_size: Maximum number of files
        max_size_mb: Maximum size per file in MB
        
    Returns:
        Tuple of (is_valid, error_message, valid_files)
        
    Example:
        >>> is_valid, error, valid_files = validate_batch_upload(uploaded_files)
        >>> if not is_valid:
        >>>     st.error(error)
        >>> else:
        >>>     st.success(f"Processing {len(valid_files)} files")
    """
    if not uploaded_files:
        return False, "No files uploaded", []
    
    # Check batch size
    if len(uploaded_files) > max_batch_size:
        return False, f"Too many files ({len(uploaded_files)}). Maximum: {max_batch_size}", []
    
    # Validate each file
    valid_files = []
    invalid_files = []
    
    for file in uploaded_files:
        is_valid, error = validate_file_upload(file, max_size_mb)
        if is_valid:
            valid_files.append(file)
        else:
            invalid_files.append((file.name, error))
    
    # If some files are invalid, return error with details
    if invalid_files:
        error_msg = "Some files are invalid:\n"
        for filename, error in invalid_files[:5]:  # Show first 5 errors
            error_msg += f"- {filename}: {error}\n"
        if len(invalid_files) > 5:
            error_msg += f"... and {len(invalid_files) - 5} more"
        return False, error_msg, valid_files
    
    return True, None, valid_files


def validate_image_file(
    uploaded_file,
    min_dimensions: Tuple[int, int] = (64, 64),
    max_dimensions: Tuple[int, int] = (2048, 2048)
) -> Tuple[bool, Optional[str]]:
    """
    Validate image file dimensions and format.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        min_dimensions: Minimum (width, height)
        max_dimensions: Maximum (width, height)
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_image_file(uploaded_file)
        >>> if not is_valid:
        >>>     st.error(error)
    """
    # First validate file upload
    is_valid, error = validate_file_upload(uploaded_file)
    if not is_valid:
        return False, error
    
    try:
        # Try to open as image
        image = Image.open(uploaded_file)
        
        # Check dimensions
        width, height = image.size
        
        if width < min_dimensions[0] or height < min_dimensions[1]:
            return False, f"Image too small ({width}x{height}). Minimum: {min_dimensions[0]}x{min_dimensions[1]}"
        
        if width > max_dimensions[0] or height > max_dimensions[1]:
            return False, f"Image too large ({width}x{height}). Maximum: {max_dimensions[0]}x{max_dimensions[1]}"
        
        # Check image mode
        if image.mode not in ['L', 'RGB', 'RGBA', 'LA']:
            return False, f"Unsupported image mode: {image.mode}. Expected: L, RGB, RGBA, or LA"
        
        # Reset file pointer for later use
        uploaded_file.seek(0)
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


# ============================================================================
# Input Parameter Validation
# ============================================================================

def validate_threshold(
    threshold: float,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate threshold parameter.
    
    Args:
        threshold: Threshold value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_threshold(threshold)
        >>> if not is_valid:
        >>>     st.error(error)
    """
    if not isinstance(threshold, (int, float)):
        return False, f"Threshold must be a number, got {type(threshold).__name__}"
    
    if threshold < min_val or threshold > max_val:
        return False, f"Threshold must be between {min_val} and {max_val}, got {threshold}"
    
    return True, None


def validate_patient_id(patient_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate patient ID format.
    
    Args:
        patient_id: Patient identifier string
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_patient_id(patient_id)
        >>> if not is_valid:
        >>>     st.error(error)
    """
    if not patient_id:
        return False, "Patient ID cannot be empty"
    
    if len(patient_id) < 3:
        return False, "Patient ID must be at least 3 characters"
    
    if len(patient_id) > 50:
        return False, "Patient ID must be less than 50 characters"
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not all(c.isalnum() or c in ['_', '-'] for c in patient_id):
        return False, "Patient ID can only contain letters, numbers, underscores, and hyphens"
    
    return True, None


def validate_numeric_range(
    value: float,
    min_val: float,
    max_val: float,
    param_name: str = "Value"
) -> Tuple[bool, Optional[str]]:
    """
    Validate numeric value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Parameter name for error message
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_numeric_range(mc_iterations, 1, 50, "MC Iterations")
        >>> if not is_valid:
        >>>     st.error(error)
    """
    if not isinstance(value, (int, float)):
        return False, f"{param_name} must be a number, got {type(value).__name__}"
    
    if value < min_val or value > max_val:
        return False, f"{param_name} must be between {min_val} and {max_val}, got {value}"
    
    return True, None


# ============================================================================
# Data Integrity Validation
# ============================================================================

def validate_api_response(
    response: dict,
    required_keys: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate API response has required keys.
    
    Args:
        response: API response dictionary
        required_keys: List of required keys
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_api_response(result, ['predicted_label', 'confidence'])
        >>> if not is_valid:
        >>>     st.error(error)
    """
    if not isinstance(response, dict):
        return False, f"Response must be a dictionary, got {type(response).__name__}"
    
    missing_keys = [key for key in required_keys if key not in response]
    
    if missing_keys:
        return False, f"Response missing required keys: {', '.join(missing_keys)}"
    
    return True, None


def validate_prediction_result(result: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate prediction result structure.
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_prediction_result(result)
        >>> if not is_valid:
        >>>     st.error(f"Invalid result: {error}")
    """
    # Check for classification results
    if 'predicted_label' in result:
        required_keys = ['predicted_label', 'confidence', 'probabilities']
        return validate_api_response(result, required_keys)
    
    # Check for segmentation results
    elif 'has_tumor' in result:
        required_keys = ['has_tumor', 'tumor_probability', 'mask_base64']
        return validate_api_response(result, required_keys)
    
    # Check for multi-task results
    elif 'classification' in result:
        if not isinstance(result['classification'], dict):
            return False, "Classification result must be a dictionary"
        return True, None
    
    return False, "Unknown result format"


# ============================================================================
# Streamlit-Specific Validators
# ============================================================================

def validate_and_display_file(
    uploaded_file,
    file_type: str = "image"
) -> Tuple[bool, Optional[Image.Image]]:
    """
    Validate file and display appropriate messages in Streamlit.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        file_type: Type of file ("image", "batch", etc.)
        
    Returns:
        Tuple of (is_valid, image_object)
        
    Example:
        >>> is_valid, image = validate_and_display_file(uploaded_file)
        >>> if is_valid:
        >>>     st.image(image, caption="Uploaded Image")
    """
    if uploaded_file is None:
        return False, None
    
    # Validate file
    is_valid, error = validate_image_file(uploaded_file)
    
    if not is_valid:
        st.error(f"❌ {error}")
        return False, None
    
    try:
        # Load image
        image = Image.open(uploaded_file)
        
        # Display success message
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ Valid {file_type} uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Display image info
        with st.expander("📊 Image Information"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Width", f"{image.width} px")
            with col2:
                st.metric("Height", f"{image.height} px")
            with col3:
                st.metric("Mode", image.mode)
        
        return True, image
        
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        return False, None


def validate_batch_and_display(
    uploaded_files: List,
    max_batch_size: int = UIConfig.MAX_BATCH_SIZE
) -> Tuple[bool, List]:
    """
    Validate batch upload and display appropriate messages.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        max_batch_size: Maximum number of files
        
    Returns:
        Tuple of (is_valid, valid_files)
        
    Example:
        >>> is_valid, files = validate_batch_and_display(uploaded_files)
        >>> if is_valid:
        >>>     st.success(f"Processing {len(files)} files")
    """
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one file")
        return False, []
    
    # Validate batch
    is_valid, error, valid_files = validate_batch_upload(uploaded_files, max_batch_size)
    
    if not is_valid:
        st.error(f"❌ {error}")
        if valid_files:
            st.info(f"ℹ️ {len(valid_files)} out of {len(uploaded_files)} files are valid")
        return False, valid_files
    
    # Display success
    total_size = sum(f.size for f in valid_files) / (1024 * 1024)
    st.success(f"✅ {len(valid_files)} valid files uploaded ({total_size:.2f} MB total)")
    
    return True, valid_files


# ============================================================================
# Export all functions
# ============================================================================

__all__ = [
    # File upload validation
    'validate_file_upload',
    'validate_batch_upload',
    'validate_image_file',
    # Input parameter validation
    'validate_threshold',
    'validate_patient_id',
    'validate_numeric_range',
    # Data integrity validation
    'validate_api_response',
    'validate_prediction_result',
    # Streamlit-specific validators
    'validate_and_display_file',
    'validate_batch_and_display',
]
