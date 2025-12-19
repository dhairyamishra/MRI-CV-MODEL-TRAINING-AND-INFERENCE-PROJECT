#!/usr/bin/env python3
"""
Preprocess Kaggle Brain MRI dataset.
Converts JPG images to normalized .npz format with robust brain masking.

Key improvements:
- Robust brain foreground masking (not skull rings)
- Z-score normalization using foreground-only statistics
- Quality checks and logging for mask failures
- Saves both image and mask for debugging
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm

# Import robust brain masking
from brain_mask import (
    compute_brain_mask,
    zscore_normalize_foreground,
)


def load_and_process_image(
    image_path: Path,
    target_size: Tuple[int, int] = (256, 256),
    use_robust_masking: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load image, apply robust brain masking, and z-score normalize.
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing (H, W)
        use_robust_masking: If True, use robust brain masking
        
    Returns:
        Tuple of (image, mask, quality_dict):
        - image: Z-score normalized array (1, H, W), background = 0
        - mask: Binary brain mask (H, W) in {0, 255}
        - quality_dict: Quality metrics from masking
    """
    # Load image as grayscale uint8
    img_u8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img_u8 is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_size = img_u8.shape
    
    # Resize if needed
    if original_size != target_size:
        img_u8 = cv2.resize(
            img_u8,
            (target_size[1], target_size[0]),  # cv2 uses (W, H)
            interpolation=cv2.INTER_LINEAR,
        )
    
    if use_robust_masking:
        # Compute robust brain mask with quality check
        mask, quality = compute_brain_mask(img_u8, return_quality_score=True)
        
        # Z-score normalize using foreground-only statistics
        img_normalized = zscore_normalize_foreground(img_u8, mask)
    else:
        # Fallback: simple normalization without masking
        mask = np.ones_like(img_u8) * 255
        img_normalized = img_u8.astype(np.float32) / 255.0
        quality = {'passed': True, 'reason': 'masking_disabled'}
    
    # Add channel dimension: (H, W) -> (1, H, W)
    img_normalized = np.expand_dims(img_normalized, axis=0)
    
    return img_normalized, mask, quality


def preprocess_kaggle_dataset(
    raw_dir: str = "data/raw/kaggle_brain_mri",
    processed_dir: str = "data/processed/kaggle",
    target_size: Tuple[int, int] = (256, 256),
    use_robust_masking: bool = True,
    save_masks: bool = True,
):
    """
    Preprocess Kaggle Brain MRI dataset with robust brain masking.
    
    Converts images from yes/no folders to unified .npz format with:
    - image: z-score normalized array (1, H, W), background = 0
    - label: 0 (no tumor) or 1 (tumor present)
    - mask: binary brain mask (H, W) in {0, 255}
    - metadata: dict with image_id, quality metrics, etc.
    
    Args:
        raw_dir: Directory containing yes/ and no/ folders
        processed_dir: Output directory for .npz files
        target_size: Target size for resizing (H, W)
        use_robust_masking: If True, use robust brain masking
        save_masks: If True, save brain masks in .npz files
    """
    print("=" * 70)
    print("Preprocessing Kaggle Brain MRI Dataset (Robust Masking)")
    print("=" * 70)
    
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Check if raw data exists
    if not raw_path.exists():
        print(f"Error: Raw data directory not found: {raw_path}")
        print("Please run scripts/download_kaggle_data.py first")
        sys.exit(1)
    
    yes_dir = raw_path / "yes"
    no_dir = raw_path / "no"
    
    if not yes_dir.exists() or not no_dir.exists():
        print(f"Error: Expected 'yes/' and 'no/' directories not found in {raw_path}")
        sys.exit(1)
    
    # Create output directory
    processed_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInput directory: {raw_path}")
    print(f"Output directory: {processed_path}")
    print(f"Target size: {target_size}")
    print(f"Robust masking: {use_robust_masking}")
    print(f"Save masks: {save_masks}")
    
    # Process both classes
    stats = {
        "yes": 0, 
        "no": 0, 
        "errors": 0,
        "mask_passed": 0,
        "mask_failed": 0,
        "mask_reasons": {},
    }
    
    for class_name, class_dir, label in [("yes", yes_dir, 1), ("no", no_dir, 0)]:
        print(f"\n[Processing '{class_name}' class (label={label})]")
        
        # Get all image files
        image_files = sorted(class_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"  Warning: No .jpg files found in {class_dir}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            try:
                # Load, mask, and normalize image
                img, mask, quality = load_and_process_image(
                    img_path,
                    target_size=target_size,
                    use_robust_masking=use_robust_masking,
                )
                
                # Track quality statistics
                if quality['passed']:
                    stats['mask_passed'] += 1
                else:
                    stats['mask_failed'] += 1
                    reason = quality['reason']
                    stats['mask_reasons'][reason] = stats['mask_reasons'].get(reason, 0) + 1
                
                # Create metadata
                metadata = {
                    "image_id": img_path.stem,
                    "class": class_name,
                    "label": label,
                    "target_size": target_size,
                    "source": "kaggle_brain_mri",
                    "robust_masking": use_robust_masking,
                    "mask_quality": quality,
                }
                
                # Save as .npz
                output_filename = f"{class_name}_{img_path.stem}.npz"
                output_path = processed_path / output_filename
                
                # Save with or without mask
                if save_masks:
                    np.savez_compressed(
                        output_path,
                        image=img,
                        label=label,
                        mask=mask,
                        metadata=metadata,
                    )
                else:
                    np.savez_compressed(
                        output_path,
                        image=img,
                        label=label,
                        metadata=metadata,
                    )
                
                stats[class_name] += 1
                
            except Exception as e:
                print(f"\n  Error processing {img_path.name}: {e}")
                stats["errors"] += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"\nProcessed images:")
    print(f"  - Tumor (yes):     {stats['yes']}")
    print(f"  - No tumor (no):   {stats['no']}")
    print(f"  - Total:           {stats['yes'] + stats['no']}")
    print(f"  - Errors:          {stats['errors']}")
    
    # Mask quality statistics
    if use_robust_masking:
        total_processed = stats['mask_passed'] + stats['mask_failed']
        if total_processed > 0:
            pass_rate = (stats['mask_passed'] / total_processed) * 100
            print(f"\nBrain Mask Quality:")
            print(f"  - Passed:  {stats['mask_passed']} ({pass_rate:.1f}%)")
            print(f"  - Failed:  {stats['mask_failed']} ({100-pass_rate:.1f}%)")
            
            if stats['mask_reasons']:
                print(f"\n  Failure reasons:")
                for reason, count in sorted(stats['mask_reasons'].items(), 
                                           key=lambda x: x[1], reverse=True):
                    pct = (count / stats['mask_failed']) * 100
                    print(f"    - {reason}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput directory: {processed_path.absolute()}")
    
    # Calculate class balance
    total = stats['yes'] + stats['no']
    if total > 0:
        yes_pct = (stats['yes'] / total) * 100
        no_pct = (stats['no'] / total) * 100
        print(f"\nClass balance:")
        print(f"  - Tumor:     {yes_pct:.1f}%")
        print(f"  - No tumor:  {no_pct:.1f}%")
    
    print("\nNext steps:")
    print("1. Create train/val/test splits with split_kaggle_data.py")
    print("2. Visualize results with visualize_augmentations.py")
    print("3. Compare old vs new preprocessing quality")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Preprocess Kaggle Brain MRI dataset to .npz format"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/kaggle_brain_mri",
        help="Input directory with yes/ and no/ folders",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/kaggle",
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Target size (H W) for resizing images",
    )
    parser.add_argument(
        "--no-robust-masking",
        action="store_true",
        help="Disable robust brain masking (use simple normalization)",
    )
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="Don't save brain masks in .npz files (saves space)",
    )
    
    args = parser.parse_args()
    
    preprocess_kaggle_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        target_size=tuple(args.target_size),
        use_robust_masking=not args.no_robust_masking,
        save_masks=not args.no_save_masks,
    )


if __name__ == "__main__":
    main()
