"""
Image Utilities for SliceWise Frontend.

This module provides helper functions for image processing,
conversion, and manipulation in the Streamlit UI.
"""

from typing import Optional, Tuple, Union
import base64
import io
import numpy as np
from PIL import Image


# ============================================================================
# Base64 Conversion Functions
# ============================================================================

def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        PIL Image object
        
    Example:
        >>> img = base64_to_image(result['gradcam_overlay'])
        >>> st.image(img, caption="Grad-CAM")
    """
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes))


def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string
        
    Example:
        >>> base64_str = image_to_base64(image)
        >>> # Can be embedded in HTML or sent to API
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Convert PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Image as bytes
        
    Example:
        >>> img_bytes = image_to_bytes(image)
        >>> result, error = classify_image(img_bytes)
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """
    Convert bytes to PIL Image.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        PIL Image object
        
    Example:
        >>> with open('image.png', 'rb') as f:
        >>>     img = bytes_to_image(f.read())
    """
    return Image.open(io.BytesIO(image_bytes))


# ============================================================================
# Image Processing Functions
# ============================================================================

def resize_image(
    image: Image.Image,
    size: Tuple[int, int],
    maintain_aspect_ratio: bool = True,
    resample: int = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize image to specified size.
    
    Args:
        image: PIL Image object
        size: Target size as (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        resample: Resampling filter
        
    Returns:
        Resized PIL Image
        
    Example:
        >>> resized = resize_image(image, (256, 256))
    """
    if maintain_aspect_ratio:
        image.thumbnail(size, resample)
        return image
    else:
        return image.resize(size, resample)


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert image to grayscale.
    
    Args:
        image: PIL Image object
        
    Returns:
        Grayscale PIL Image
        
    Example:
        >>> gray_img = convert_to_grayscale(image)
    """
    return image.convert('L')


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB.
    
    Args:
        image: PIL Image object
        
    Returns:
        RGB PIL Image
        
    Example:
        >>> rgb_img = convert_to_rgb(image)
    """
    return image.convert('RGB')


def normalize_image(
    image: Union[Image.Image, np.ndarray],
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize image array.
    
    Args:
        image: PIL Image or numpy array
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized numpy array
        
    Example:
        >>> normalized = normalize_image(image, method='minmax')
    """
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Convert to float
    img_array = img_array.astype(np.float32)
    
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min)
    elif method == 'zscore':
        # Z-score normalization
        mean = img_array.mean()
        std = img_array.std()
        if std > 0:
            img_array = (img_array - mean) / std
        else:
            img_array = img_array - mean
    
    return img_array


# ============================================================================
# Image Validation Functions
# ============================================================================

def get_image_info(image: Image.Image) -> dict:
    """
    Get comprehensive image information.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image metadata
        
    Example:
        >>> info = get_image_info(image)
        >>> st.write(f"Size: {info['width']}x{info['height']}")
    """
    return {
        'width': image.width,
        'height': image.height,
        'size': image.size,
        'mode': image.mode,
        'format': image.format,
        'is_grayscale': image.mode in ['L', 'LA'],
        'is_rgb': image.mode in ['RGB', 'RGBA'],
        'has_alpha': image.mode in ['LA', 'RGBA', 'PA'],
        'num_channels': len(image.getbands()),
    }


def is_valid_mri_image(
    image: Image.Image,
    min_size: Tuple[int, int] = (64, 64),
    max_size: Tuple[int, int] = (2048, 2048)
) -> Tuple[bool, Optional[str]]:
    """
    Validate if image is suitable for MRI analysis.
    
    Args:
        image: PIL Image object
        min_size: Minimum acceptable size (width, height)
        max_size: Maximum acceptable size (width, height)
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> is_valid, error = is_valid_mri_image(image)
        >>> if not is_valid:
        >>>     st.error(error)
    """
    # Check dimensions
    if image.width < min_size[0] or image.height < min_size[1]:
        return False, f"Image too small. Minimum size: {min_size[0]}x{min_size[1]}"
    
    if image.width > max_size[0] or image.height > max_size[1]:
        return False, f"Image too large. Maximum size: {max_size[0]}x{max_size[1]}"
    
    # Check mode
    if image.mode not in ['L', 'RGB', 'RGBA', 'LA']:
        return False, f"Unsupported image mode: {image.mode}"
    
    return True, None


# ============================================================================
# Display Helper Functions
# ============================================================================

def create_thumbnail(
    image: Image.Image,
    max_size: Tuple[int, int] = (300, 300)
) -> Image.Image:
    """
    Create a thumbnail of the image.
    
    Args:
        image: PIL Image object
        max_size: Maximum thumbnail size
        
    Returns:
        Thumbnail PIL Image
        
    Example:
        >>> thumb = create_thumbnail(image, max_size=(200, 200))
        >>> st.image(thumb, caption="Preview")
    """
    thumb = image.copy()
    thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
    return thumb


def stack_images_horizontally(images: list) -> Image.Image:
    """
    Stack multiple images horizontally.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        Combined PIL Image
        
    Example:
        >>> combined = stack_images_horizontally([img1, img2, img3])
        >>> st.image(combined, caption="Comparison")
    """
    if not images:
        raise ValueError("No images provided")
    
    # Get max height and total width
    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images)
    
    # Create new image
    combined = Image.new('RGB', (total_width, max_height), color='white')
    
    # Paste images
    x_offset = 0
    for img in images:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined


def stack_images_vertically(images: list) -> Image.Image:
    """
    Stack multiple images vertically.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        Combined PIL Image
        
    Example:
        >>> combined = stack_images_vertically([img1, img2, img3])
        >>> st.image(combined, caption="Stack")
    """
    if not images:
        raise ValueError("No images provided")
    
    # Get max width and total height
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    
    # Create new image
    combined = Image.new('RGB', (max_width, total_height), color='white')
    
    # Paste images
    y_offset = 0
    for img in images:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        combined.paste(img, (0, y_offset))
        y_offset += img.height
    
    return combined


# ============================================================================
# Export all functions
# ============================================================================

__all__ = [
    # Base64 conversion
    'base64_to_image',
    'image_to_base64',
    'image_to_bytes',
    'bytes_to_image',
    # Image processing
    'resize_image',
    'convert_to_grayscale',
    'convert_to_rgb',
    'normalize_image',
    # Validation
    'get_image_info',
    'is_valid_mri_image',
    # Display helpers
    'create_thumbnail',
    'stack_images_horizontally',
    'stack_images_vertically',
]
