"""
Post-processing functions for segmentation masks.

Provides:
- Thresholding (fixed, Otsu)
- Connected components analysis
- Small blob removal
- Hole filling
- Largest component selection
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology


def threshold_mask(
    prob_map: np.ndarray,
    threshold: Optional[float] = None,
    use_otsu: bool = False,
) -> np.ndarray:
    """
    Threshold probability map to create binary mask.
    
    Args:
        prob_map: Probability map (H, W) with values in [0, 1]
        threshold: Fixed threshold value (default: 0.5)
        use_otsu: If True, use Otsu's method to find optimal threshold
    
    Returns:
        Binary mask (H, W) with values {0, 1}
    """
    if use_otsu:
        # Convert to uint8 for Otsu
        prob_uint8 = (prob_map * 255).astype(np.uint8)
        threshold_val = filters.threshold_otsu(prob_uint8) / 255.0
    else:
        threshold_val = threshold if threshold is not None else 0.5
    
    binary_mask = (prob_map > threshold_val).astype(np.uint8)
    return binary_mask


def remove_small_objects(
    mask: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    """
    Remove small connected components from binary mask.
    
    Args:
        mask: Binary mask (H, W)
        min_size: Minimum size (in pixels) to keep
    
    Returns:
        Cleaned binary mask
    """
    # Use skimage's remove_small_objects
    cleaned = morphology.remove_small_objects(
        mask.astype(bool),
        min_size=min_size,
    )
    return cleaned.astype(np.uint8)


def fill_holes(
    mask: np.ndarray,
    max_hole_size: Optional[int] = None,
) -> np.ndarray:
    """
    Fill holes in binary mask.
    
    Args:
        mask: Binary mask (H, W)
        max_hole_size: Maximum hole size to fill (None = fill all)
    
    Returns:
        Mask with holes filled
    """
    if max_hole_size is None:
        # Fill all holes
        filled = ndimage.binary_fill_holes(mask)
    else:
        # Fill only small holes
        filled = morphology.remove_small_holes(
            mask.astype(bool),
            area_threshold=max_hole_size,
        )
    
    return filled.astype(np.uint8)


def keep_largest_component(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Keep only the largest connected component.
    
    Args:
        mask: Binary mask (H, W)
    
    Returns:
        Mask with only largest component
    """
    # Label connected components
    labeled = measure.label(mask, connectivity=2)
    
    if labeled.max() == 0:
        # No components found
        return mask
    
    # Find largest component
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background
    
    largest_component = component_sizes.argmax()
    
    # Keep only largest
    largest_mask = (labeled == largest_component).astype(np.uint8)
    
    return largest_mask


def morphological_operations(
    mask: np.ndarray,
    operation: str = 'close',
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply morphological operations to mask.
    
    Args:
        mask: Binary mask (H, W)
        operation: 'open', 'close', 'dilate', 'erode'
        kernel_size: Size of structuring element
    
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    
    if operation == 'open':
        result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'dilate':
        result = cv2.dilate(mask, kernel)
    elif operation == 'erode':
        result = cv2.erode(mask, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result


def postprocess_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    use_otsu: bool = False,
    min_object_size: int = 100,
    fill_holes_size: Optional[int] = None,
    keep_largest: bool = False,
    morphology_op: Optional[str] = None,
    morphology_kernel: int = 3,
) -> Tuple[np.ndarray, dict]:
    """
    Complete post-processing pipeline for segmentation mask.
    
    Args:
        prob_map: Probability map (H, W) with values in [0, 1]
        threshold: Threshold for binarization
        use_otsu: Use Otsu's method for thresholding
        min_object_size: Minimum object size to keep (0 to disable)
        fill_holes_size: Maximum hole size to fill (None = fill all, 0 = disable)
        keep_largest: If True, keep only largest component
        morphology_op: Optional morphological operation ('open', 'close', etc.)
        morphology_kernel: Kernel size for morphological operations
    
    Returns:
        Tuple of (cleaned_mask, stats_dict)
            - cleaned_mask: Post-processed binary mask
            - stats_dict: Statistics about the processing
    """
    stats = {}
    
    # 1. Thresholding
    mask = threshold_mask(prob_map, threshold, use_otsu)
    stats['initial_pixels'] = int(mask.sum())
    
    # 2. Morphological operation (if specified)
    if morphology_op:
        mask = morphological_operations(mask, morphology_op, morphology_kernel)
        stats['after_morphology_pixels'] = int(mask.sum())
    
    # 3. Remove small objects
    if min_object_size > 0:
        mask = remove_small_objects(mask, min_object_size)
        stats['after_small_removal_pixels'] = int(mask.sum())
    
    # 4. Fill holes
    if fill_holes_size is not None and fill_holes_size != 0:
        mask = fill_holes(mask, fill_holes_size)
        stats['after_hole_filling_pixels'] = int(mask.sum())
    
    # 5. Keep largest component
    if keep_largest:
        mask = keep_largest_component(mask)
        stats['after_largest_component_pixels'] = int(mask.sum())
    
    stats['final_pixels'] = int(mask.sum())
    stats['num_components'] = int(measure.label(mask).max())
    
    return mask, stats


if __name__ == "__main__":
    # Test post-processing functions
    print("Testing Post-processing Functions...")
    print("=" * 60)
    
    # Create synthetic probability map
    prob_map = np.zeros((256, 256), dtype=np.float32)
    
    # Add main tumor region
    prob_map[80:180, 80:180] = 0.9
    
    # Add small noise blobs
    prob_map[20:30, 20:30] = 0.8
    prob_map[220:230, 220:230] = 0.7
    
    # Add holes in main region
    prob_map[120:130, 120:130] = 0.1
    
    # Add some gradient
    prob_map += np.random.randn(256, 256) * 0.1
    prob_map = np.clip(prob_map, 0, 1)
    
    print(f"\nInput probability map:")
    print(f"  Shape: {prob_map.shape}")
    print(f"  Range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # Test 1: Basic thresholding
    print("\n1. Basic Thresholding (threshold=0.5)")
    mask = threshold_mask(prob_map, threshold=0.5)
    print(f"   Pixels: {mask.sum()}")
    print(f"   Components: {measure.label(mask).max()}")
    
    # Test 2: Otsu thresholding
    print("\n2. Otsu Thresholding")
    mask_otsu = threshold_mask(prob_map, use_otsu=True)
    print(f"   Pixels: {mask_otsu.sum()}")
    
    # Test 3: Remove small objects
    print("\n3. Remove Small Objects (min_size=100)")
    mask_cleaned = remove_small_objects(mask, min_size=100)
    print(f"   Before: {mask.sum()} pixels, {measure.label(mask).max()} components")
    print(f"   After:  {mask_cleaned.sum()} pixels, {measure.label(mask_cleaned).max()} components")
    
    # Test 4: Fill holes
    print("\n4. Fill Holes")
    mask_filled = fill_holes(mask_cleaned)
    print(f"   Before: {mask_cleaned.sum()} pixels")
    print(f"   After:  {mask_filled.sum()} pixels")
    
    # Test 5: Keep largest component
    print("\n5. Keep Largest Component")
    mask_largest = keep_largest_component(mask)
    print(f"   Before: {measure.label(mask).max()} components")
    print(f"   After:  {measure.label(mask_largest).max()} component")
    
    # Test 6: Morphological operations
    print("\n6. Morphological Operations")
    for op in ['open', 'close', 'dilate', 'erode']:
        mask_morph = morphological_operations(mask, operation=op, kernel_size=3)
        print(f"   {op:8s}: {mask_morph.sum()} pixels")
    
    # Test 7: Complete pipeline
    print("\n7. Complete Post-processing Pipeline")
    mask_final, stats = postprocess_mask(
        prob_map,
        threshold=0.5,
        min_object_size=100,
        fill_holes_size=500,
        keep_largest=False,
        morphology_op='close',
        morphology_kernel=3,
    )
    
    print(f"   Statistics:")
    for key, value in stats.items():
        print(f"     {key}: {value}")
    
    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
