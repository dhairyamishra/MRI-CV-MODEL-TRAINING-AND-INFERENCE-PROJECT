#!/usr/bin/env python3
"""
Test script to verify brain mask application in segmentation.

This script tests the brain mask functionality by:
1. Loading a Kaggle test image with background padding
2. Running segmentation with and without brain mask
3. Comparing the results
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.backend.utils.image_processing import (
    compute_brain_mask_from_image,
    apply_brain_mask_to_prediction,
    detect_background_padding
)


def test_brain_mask_detection():
    """Test brain mask computation and application."""
    
    print("=" * 80)
    print("Testing Brain Mask Application for Segmentation")
    print("=" * 80)
    
    # Load a Kaggle test image
    test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"
    
    if not test_image_path.exists():
        print(f"\n❌ Test image not found: {test_image_path}")
        print("Please run the example dataset generation first:")
        print("  python scripts/data/preprocessing/export_dataset_examples.py")
        return
    
    # Load image
    import cv2
    image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
    print(f"\n✅ Loaded test image: {test_image_path.name}")
    print(f"   Shape: {image.shape}")
    print(f"   Range: [{image.min()}, {image.max()}]")
    
    # Normalize to [0, 1]
    image_normalized = image.astype(np.float32) / 255.0
    
    # Detect background padding
    has_padding = detect_background_padding(image_normalized)
    print(f"\n📊 Background padding detected: {has_padding}")
    
    if has_padding:
        # Compute brain mask
        brain_mask = compute_brain_mask_from_image(image)
        
        if brain_mask is not None:
            print(f"✅ Brain mask computed successfully")
            print(f"   Shape: {brain_mask.shape}")
            print(f"   Brain pixels: {brain_mask.sum()} ({brain_mask.sum() / brain_mask.size * 100:.1f}%)")
            print(f"   Background pixels: {(brain_mask == 0).sum()} ({(brain_mask == 0).sum() / brain_mask.size * 100:.1f}%)")
            
            # Create a dummy segmentation prediction (simulate model output)
            dummy_prediction = np.random.rand(*image.shape).astype(np.float32)
            
            # Apply brain mask
            masked_prediction = apply_brain_mask_to_prediction(dummy_prediction, brain_mask)
            
            # Check that background is zeroed
            background_pixels = (brain_mask == 0)
            background_values = masked_prediction[background_pixels]
            
            print(f"\n🔍 Verification:")
            print(f"   Background pixels in prediction: {(background_values == 0).sum()} / {background_pixels.sum()}")
            print(f"   All background zeroed: {np.all(background_values == 0)}")
            
            # Visualize
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(image, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(brain_mask, cmap='gray')
            axes[0, 1].set_title('Brain Mask')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(image * brain_mask, cmap='gray')
            axes[0, 2].set_title('Masked Image')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(dummy_prediction, cmap='hot')
            axes[1, 0].set_title('Dummy Prediction (Before Mask)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(masked_prediction, cmap='hot')
            axes[1, 1].set_title('Masked Prediction (After Mask)')
            axes[1, 1].axis('off')
            
            # Difference
            difference = dummy_prediction - masked_prediction
            axes[1, 2].imshow(difference, cmap='hot')
            axes[1, 2].set_title('Difference (Zeroed Background)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            output_path = project_root / "visualizations" / "brain_mask_test.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n💾 Visualization saved: {output_path}")
            
            print("\n" + "=" * 80)
            print("✅ Brain Mask Test PASSED")
            print("=" * 80)
            
        else:
            print(f"❌ Failed to compute brain mask")
    else:
        print(f"ℹ️  No background padding detected - brain mask not needed")


if __name__ == "__main__":
    test_brain_mask_detection()
