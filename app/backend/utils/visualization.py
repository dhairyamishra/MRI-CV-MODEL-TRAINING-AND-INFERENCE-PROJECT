"""
Visualization utilities for SliceWise Backend.

This module provides functions for creating overlays, encoding images to base64,
and generating visualizations for Grad-CAM and uncertainty maps.

Functions copied and adapted from main_v2.py (lines 259-295, 949-957).
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional
import matplotlib.pyplot as plt


# ============================================================================
# Base64 Encoding
# ============================================================================

def numpy_to_base64_png(array: np.ndarray) -> str:
    """
    Convert numpy array to base64 encoded PNG.
    
    Copied from main_v2.py (lines 259-274).
    
    Args:
        array: Numpy array to encode (2D grayscale or 3D RGB)
    
    Returns:
        Base64 encoded PNG string
        
    Example:
        >>> image = np.random.rand(256, 256)
        >>> base64_str = numpy_to_base64_png(image)
        >>> assert isinstance(base64_str, str)
        >>> assert len(base64_str) > 0
    """
    # Normalize to 0-255 if needed
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(array.shape) == 2:
        image = Image.fromarray(array, mode='L')
    else:
        image = Image.fromarray(array)
    
    # Encode to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


# ============================================================================
# Overlay Creation
# ============================================================================

def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Create overlay of mask on image.
    
    Copied and enhanced from main_v2.py (lines 277-295).
    
    Args:
        image: Original image (2D grayscale or 3D RGB)
        mask: Binary mask (2D)
        alpha: Transparency of overlay (0.0 = transparent, 1.0 = opaque)
        color: RGB color for mask overlay (default: red)
    
    Returns:
        RGB overlay image as uint8 array
        
    Example:
        >>> image = np.random.rand(256, 256)
        >>> mask = np.random.randint(0, 2, (256, 256))
        >>> overlay = create_overlay(image, mask, alpha=0.4)
        >>> assert overlay.shape == (256, 256, 3)
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
    
    # Normalize image to 0-255
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 0] = color
    
    # Blend
    overlay = ((1 - alpha) * image_rgb + alpha * colored_mask).astype(np.uint8)
    return overlay


def create_tumor_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create tumor overlay with red color (convenience function).
    
    Args:
        image: Original image
        mask: Binary tumor mask
        alpha: Transparency of overlay
    
    Returns:
        RGB overlay image
    """
    return create_overlay(image, mask, alpha=alpha, color=(255, 0, 0))


# ============================================================================
# Grad-CAM Visualization
# ============================================================================

def create_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Create Grad-CAM overlay visualization.
    
    Extracted and enhanced from main_v2.py (lines 949-957).
    
    Args:
        image: Original image (2D grayscale)
        heatmap: Grad-CAM heatmap (2D, values in [0, 1])
        alpha: Transparency of heatmap overlay
        colormap: Matplotlib colormap name
    
    Returns:
        RGB Grad-CAM overlay as uint8 array
        
    Example:
        >>> image = np.random.rand(256, 256)
        >>> heatmap = np.random.rand(256, 256)
        >>> overlay = create_gradcam_overlay(image, heatmap)
        >>> assert overlay.shape == (256, 256, 3)
    """
    # Resize heatmap to match image size if needed
    if heatmap.shape != image.shape:
        import cv2
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap to heatmap
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # RGB only (no alpha)
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Ensure image is RGB
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
    
    # Normalize image to 0-255
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # Blend image and heatmap
    overlay = ((1 - alpha) * image_rgb + alpha * heatmap_colored).astype(np.uint8)
    return overlay


# ============================================================================
# Uncertainty Visualization
# ============================================================================

def create_uncertainty_overlay(
    image: np.ndarray,
    uncertainty_map: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'viridis'
) -> np.ndarray:
    """
    Create uncertainty map overlay visualization.
    
    Args:
        image: Original image (2D grayscale)
        uncertainty_map: Uncertainty values (2D, higher = more uncertain)
        alpha: Transparency of uncertainty overlay
        colormap: Matplotlib colormap name (default: viridis)
    
    Returns:
        RGB uncertainty overlay as uint8 array
        
    Example:
        >>> image = np.random.rand(256, 256)
        >>> uncertainty = np.random.rand(256, 256)
        >>> overlay = create_uncertainty_overlay(image, uncertainty)
        >>> assert overlay.shape == (256, 256, 3)
    """
    # Normalize uncertainty to [0, 1]
    if uncertainty_map.max() > 0:
        uncertainty_normalized = uncertainty_map / uncertainty_map.max()
    else:
        uncertainty_normalized = uncertainty_map
    
    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    uncertainty_colored = cmap(uncertainty_normalized)[:, :, :3]  # RGB only
    uncertainty_colored = (uncertainty_colored * 255).astype(np.uint8)
    
    # Ensure image is RGB
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
    
    # Normalize image to 0-255
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # Blend
    overlay = ((1 - alpha) * image_rgb + alpha * uncertainty_colored).astype(np.uint8)
    return overlay


# ============================================================================
# Multi-Channel Visualization
# ============================================================================

def create_side_by_side_visualization(
    images: list[np.ndarray],
    titles: Optional[list[str]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> np.ndarray:
    """
    Create side-by-side visualization of multiple images.
    
    Args:
        images: List of images to display
        titles: Optional list of titles for each image
        figsize: Figure size (width, height)
    
    Returns:
        Combined visualization as RGB array
        
    Example:
        >>> images = [np.random.rand(256, 256) for _ in range(3)]
        >>> titles = ["Original", "Mask", "Overlay"]
        >>> viz = create_side_by_side_visualization(images, titles)
    """
    n_images = len(images)
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert to numpy array
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buffer.seek(0)
    
    image = Image.open(buffer)
    return np.array(image)


# ============================================================================
# Color Utilities
# ============================================================================

def apply_colormap(
    array: np.ndarray,
    colormap: str = 'jet',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> np.ndarray:
    """
    Apply matplotlib colormap to array.
    
    Args:
        array: Input array (2D)
        colormap: Matplotlib colormap name
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
    
    Returns:
        RGB image as uint8 array
        
    Example:
        >>> data = np.random.rand(256, 256)
        >>> colored = apply_colormap(data, 'viridis')
        >>> assert colored.shape == (256, 256, 3)
    """
    # Normalize to [0, 1]
    if vmin is None:
        vmin = array.min()
    if vmax is None:
        vmax = array.max()
    
    if vmax - vmin > 0:
        normalized = (array - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(array)
    
    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    colored = cmap(normalized)[:, :, :3]  # RGB only
    return (colored * 255).astype(np.uint8)


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Test visualization utilities
    print("Testing visualization utilities...")
    
    # Test 1: Base64 encoding
    print("\n1. Testing base64 encoding:")
    test_image = np.random.rand(256, 256)
    base64_str = numpy_to_base64_png(test_image)
    print(f"   Base64 string length: {len(base64_str)}")
    print(f"   ✓ Successfully encoded to base64")
    
    # Test 2: Overlay creation
    print("\n2. Testing overlay creation:")
    test_mask = np.random.randint(0, 2, (256, 256))
    overlay = create_overlay(test_image, test_mask, alpha=0.4)
    print(f"   Input image shape: {test_image.shape}")
    print(f"   Output overlay shape: {overlay.shape}")
    print(f"   ✓ Successfully created overlay")
    
    # Test 3: Grad-CAM overlay
    print("\n3. Testing Grad-CAM overlay:")
    heatmap = np.random.rand(256, 256)
    gradcam = create_gradcam_overlay(test_image, heatmap, alpha=0.4)
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Grad-CAM overlay shape: {gradcam.shape}")
    print(f"   ✓ Successfully created Grad-CAM overlay")
    
    # Test 4: Uncertainty overlay
    print("\n4. Testing uncertainty overlay:")
    uncertainty = np.random.rand(256, 256)
    uncertainty_viz = create_uncertainty_overlay(test_image, uncertainty)
    print(f"   Uncertainty map shape: {uncertainty.shape}")
    print(f"   Uncertainty overlay shape: {uncertainty_viz.shape}")
    print(f"   ✓ Successfully created uncertainty overlay")
    
    # Test 5: Colormap application
    print("\n5. Testing colormap application:")
    colored = apply_colormap(test_image, 'jet')
    print(f"   Input shape: {test_image.shape}")
    print(f"   Colored output shape: {colored.shape}")
    print(f"   ✓ Successfully applied colormap")
    
    print("\n✅ All tests passed!")
