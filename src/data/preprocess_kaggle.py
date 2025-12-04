#!/usr/bin/env python3
"""
Preprocess Kaggle Brain MRI dataset.
Converts JPG images to normalized .npz format.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


def load_and_normalize_image(image_path: Path) -> np.ndarray:
    """
    Load image and normalize to [0, 1] range.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Normalized image array of shape (1, H, W)
    """
    # Load image as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add channel dimension: (H, W) -> (1, H, W)
    img = np.expand_dims(img, axis=0)
    
    return img


def preprocess_kaggle_dataset(
    raw_dir: str = "data/raw/kaggle_brain_mri",
    processed_dir: str = "data/processed/kaggle",
    target_size: Tuple[int, int] = (256, 256),
):
    """
    Preprocess Kaggle Brain MRI dataset.
    
    Converts images from yes/no folders to unified .npz format with:
    - image: normalized array (1, H, W)
    - label: 0 (no tumor) or 1 (tumor present)
    - metadata: dict with image_id, original_size, etc.
    
    Args:
        raw_dir: Directory containing yes/ and no/ folders
        processed_dir: Output directory for .npz files
        target_size: Target size for resizing (H, W)
    """
    print("=" * 60)
    print("Preprocessing Kaggle Brain MRI Dataset")
    print("=" * 60)
    
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
    
    # Process both classes
    stats = {"yes": 0, "no": 0, "errors": 0}
    
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
                # Load and normalize image
                img = load_and_normalize_image(img_path)
                original_size = img.shape[1:]  # (H, W)
                
                # Resize if needed
                if original_size != target_size:
                    # Transpose to (H, W, C) for cv2.resize
                    img_resized = cv2.resize(
                        img[0],  # Remove channel dim
                        (target_size[1], target_size[0]),  # cv2 uses (W, H)
                        interpolation=cv2.INTER_LINEAR,
                    )
                    # Add channel dimension back
                    img = np.expand_dims(img_resized, axis=0)
                
                # Create metadata
                metadata = {
                    "image_id": img_path.stem,
                    "class": class_name,
                    "label": label,
                    "original_size": original_size,
                    "target_size": target_size,
                    "source": "kaggle_brain_mri",
                }
                
                # Save as .npz
                output_filename = f"{class_name}_{img_path.stem}.npz"
                output_path = processed_path / output_filename
                
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
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nProcessed images:")
    print(f"  - Tumor (yes):     {stats['yes']}")
    print(f"  - No tumor (no):   {stats['no']}")
    print(f"  - Total:           {stats['yes'] + stats['no']}")
    print(f"  - Errors:          {stats['errors']}")
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
    print("1. Create train/val/test splits with split_patients.py")
    print("2. Verify data with visualization notebook")


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
    
    args = parser.parse_args()
    
    preprocess_kaggle_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        target_size=tuple(args.target_size),
    )


if __name__ == "__main__":
    main()
