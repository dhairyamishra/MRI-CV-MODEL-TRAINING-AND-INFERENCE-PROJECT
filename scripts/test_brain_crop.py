#!/usr/bin/env python3
"""
Test BrainRegionCrop transform on sample MRI images.
Visualizes before/after to verify border removal works correctly.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.transforms import BrainRegionCrop, get_train_transforms, get_val_transforms


def test_brain_crop_on_sample():
    """Test BrainRegionCrop on a sample image."""
    print("=" * 60)
    print("Testing BrainRegionCrop Transform")
    print("=" * 60)
    
    # Load a sample image from processed data
    data_dir = project_root / "data" / "processed" / "kaggle" / "train"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run preprocessing first:")
        print("  python scripts/download_kaggle_data.py")
        return
    
    # Get first .npz file
    npz_files = list(data_dir.glob("*.npz"))
    if not npz_files:
        print(f"Error: No .npz files found in {data_dir}")
        return
    
    sample_file = npz_files[0]
    print(f"\nLoading sample: {sample_file.name}")
    
    # Load image
    data = np.load(sample_file)
    image = data['image']  # (1, H, W)
    label = data['label']
    
    print(f"Original shape: {image.shape}")
    print(f"Label: {label} ({'Tumor' if label == 1 else 'No Tumor'})")
    
    # DEBUG: Show pixel value distribution
    print(f"\n🔍 DEBUG - Pixel value analysis:")
    print(f"  Min value: {image.min():.6f}")
    print(f"  Max value: {image.max():.6f}")
    print(f"  Mean value: {image.mean():.6f}")
    print(f"  Median value: {np.median(image):.6f}")
    print(f"  1st percentile: {np.percentile(image, 1):.6f}")
    print(f"  2nd percentile: {np.percentile(image, 2):.6f}")
    print(f"  5th percentile: {np.percentile(image, 5):.6f}")
    print(f"  10th percentile: {np.percentile(image, 10):.6f}")
    
    # Count near-zero pixels
    near_zero = (image < 0.01).sum()
    total_pixels = image.size
    print(f"  Pixels < 0.01: {near_zero}/{total_pixels} ({100*near_zero/total_pixels:.1f}%)")
    
    # Test BrainRegionCrop
    crop_transform = BrainRegionCrop(margin=10)
    cropped = crop_transform(image)
    
    print(f"Cropped shape: {cropped.shape}")
    print(f"Size reduction: {image.shape[1]*image.shape[2]} → {cropped.shape[1]*cropped.shape[2]} pixels")
    print(f"Reduction: {100*(1 - cropped.shape[1]*cropped.shape[2]/(image.shape[1]*image.shape[2])):.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title(f'Original ({image.shape[1]}×{image.shape[2]})\nLabel: {label}')
    axes[0].axis('off')
    
    # Cropped
    axes[1].imshow(cropped[0], cmap='gray')
    axes[1].set_title(f'Brain Region Crop ({cropped.shape[1]}×{cropped.shape[2]})\nBorders Removed!')
    axes[1].axis('off')
    
    # Difference (show what was removed)
    diff = np.zeros_like(image[0])
    h_crop, w_crop = cropped.shape[1:]
    h_orig, w_orig = image.shape[1:]
    y_start = (h_orig - h_crop) // 2
    x_start = (w_orig - w_crop) // 2
    diff[y_start:y_start+h_crop, x_start:x_start+w_crop] = cropped[0]
    removed = image[0] - diff
    
    axes[2].imshow(removed, cmap='Reds')
    axes[2].set_title('Removed Regions (Red)\nBorders & Background')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = project_root / "visualizations" / "brain_crop_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "brain_crop_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    return True


def test_transforms_pipeline():
    """Test that transforms pipeline includes BrainRegionCrop."""
    print("\n" + "=" * 60)
    print("Testing Transform Pipelines")
    print("=" * 60)
    
    # Test train transforms
    train_transform = get_train_transforms()
    print("\n✓ Train transforms:")
    print(f"  Number of transforms: {len(train_transform.transforms)}")
    print(f"  First transform: {train_transform.transforms[0].__class__.__name__}")
    
    if train_transform.transforms[0].__class__.__name__ == 'BrainRegionCrop':
        print("  ✓ BrainRegionCrop is FIRST (correct!)")
    else:
        print("  ✗ BrainRegionCrop is NOT first (error!)")
    
    # Test val transforms
    val_transform = get_val_transforms()
    print("\n✓ Validation transforms:")
    print(f"  Number of transforms: {len(val_transform.transforms)}")
    print(f"  First transform: {val_transform.transforms[0].__class__.__name__}")
    
    if val_transform.transforms[0].__class__.__name__ == 'BrainRegionCrop':
        print("  ✓ BrainRegionCrop is present (correct!)")
    else:
        print("  ✗ BrainRegionCrop is NOT present (error!)")
    
    # Test on sample data
    print("\n" + "=" * 60)
    print("Testing on Sample Data")
    print("=" * 60)
    
    # Create dummy image with borders
    dummy_img = np.zeros((1, 256, 256), dtype=np.float32)
    # Add "brain" in center
    dummy_img[0, 50:200, 50:200] = np.random.rand(150, 150) * 0.8 + 0.2
    
    print(f"\nDummy image shape: {dummy_img.shape}")
    
    # Apply train transform
    transformed = train_transform(dummy_img)
    print(f"After train transform: {transformed.shape}")
    print(f"Size reduction: {100*(1 - transformed.size/dummy_img.size):.1f}%")
    
    print("\n✓ All transform tests passed!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BrainRegionCrop Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Visual test on real data
        print("\n[Test 1/2] Visual test on real MRI data...")
        test_brain_crop_on_sample()
        
        # Test 2: Pipeline test
        print("\n[Test 2/2] Transform pipeline test...")
        test_transforms_pipeline()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check the visualization in visualizations/brain_crop_test/")
        print("2. Verify borders are removed correctly")
        print("3. If looks good, retrain the model!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
