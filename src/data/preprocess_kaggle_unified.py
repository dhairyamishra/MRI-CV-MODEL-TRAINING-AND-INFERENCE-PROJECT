"""
Unified preprocessing for Kaggle Brain MRI dataset.

Converts raw Kaggle images to the same format as BraTS:
- Resize to 256×256
- Z-score normalization (matching BraTS preprocessing)
- Save as .npz files with metadata
- Compatible with MultiSourceDataset

This ensures both datasets produce identical tensor formats for multi-task learning.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def normalize_zscore(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalization (mean=0, std=1).
    
    Matches the normalization used in BraTS preprocessing.
    
    Args:
        image: Input image array
    
    Returns:
        Normalized image
    """
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        normalized = (image - mean) / std
    else:
        normalized = image - mean
    return normalized.astype(np.float32)


def load_and_preprocess_image(
    image_path: Path,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to image file
        target_size: Target size (H, W)
    
    Returns:
        Preprocessed image of shape (1, H, W)
    """
    # Load image as grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize to target size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to float
    image = image.astype(np.float32)
    
    # Z-score normalization (matching BraTS)
    image = normalize_zscore(image)
    
    # Add channel dimension: (H, W) → (1, H, W)
    image = image[np.newaxis, :, :]
    
    return image


def process_kaggle_dataset(
    input_dir: str = "data/raw/kaggle_brain_mri",
    output_dir: str = "data/processed/kaggle_unified",
    target_size: Tuple[int, int] = (256, 256),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Preprocess entire Kaggle dataset with unified format.
    
    Args:
        input_dir: Directory containing 'yes/' and 'no/' folders
        output_dir: Output directory for processed .npz files
        target_size: Target size (H, W)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("Kaggle Brain MRI → Unified Format Preprocessing")
    print("=" * 70)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"✗ Error: Input directory not found: {input_path}")
        sys.exit(1)
    
    # Check for yes/no directories
    yes_dir = input_path / "yes"
    no_dir = input_path / "no"
    
    if not yes_dir.exists() or not no_dir.exists():
        print(f"✗ Error: Expected 'yes/' and 'no/' directories in {input_path}")
        sys.exit(1)
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Input:  {input_path.absolute()}")
    print(f"  Output: {output_path.absolute()}")
    print(f"  Target size: {target_size}")
    print(f"  Normalization: Z-score (matching BraTS)")
    print(f"  Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    
    # Collect all image paths with labels
    print("\n[1/3] Collecting images...")
    
    tumor_images = sorted(yes_dir.glob("*.jpg"))
    no_tumor_images = sorted(no_dir.glob("*.jpg"))
    
    print(f"  Tumor images: {len(tumor_images)}")
    print(f"  No tumor images: {len(no_tumor_images)}")
    print(f"  Total: {len(tumor_images) + len(no_tumor_images)}")
    
    # Create dataset list with labels
    dataset = []
    for img_path in tumor_images:
        dataset.append((img_path, 1, "tumor"))
    for img_path in no_tumor_images:
        dataset.append((img_path, 0, "no_tumor"))
    
    # Shuffle with seed
    np.random.seed(seed)
    np.random.shuffle(dataset)
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]
    
    print(f"\n[2/3] Split sizes:")
    print(f"  Train: {len(train_data)} ({len(train_data)/n_total:.1%})")
    print(f"  Val:   {len(val_data)} ({len(val_data)/n_total:.1%})")
    print(f"  Test:  {len(test_data)} ({len(test_data)/n_total:.1%})")
    
    # Process each split
    print(f"\n[3/3] Processing images...")
    
    splits = [
        ("train", train_data, train_dir),
        ("val", val_data, val_dir),
        ("test", test_data, test_dir),
    ]
    
    total_processed = 0
    
    for split_name, split_data, split_dir in splits:
        print(f"\n  Processing {split_name} set...")
        
        for idx, (img_path, label, class_name) in enumerate(tqdm(split_data, desc=f"  {split_name}")):
            try:
                # Load and preprocess image
                image = load_and_preprocess_image(img_path, target_size)
                
                # Create metadata
                metadata = {
                    'image_id': img_path.stem,
                    'original_path': str(img_path),
                    'class': class_name,
                    'label': label,
                    'split': split_name,
                    'target_size': target_size,
                    'normalization': 'zscore',
                    'source': 'kaggle',
                }
                
                # Save as .npz
                output_file = split_dir / f"kaggle_{idx:04d}.npz"
                np.savez_compressed(
                    output_file,
                    image=image,
                    label=label,
                    metadata=metadata
                )
                
                total_processed += 1
                
            except Exception as e:
                print(f"\n    ✗ Error processing {img_path.name}: {e}")
                continue
    
    # Summary
    print("\n" + "=" * 70)
    print("[OK] Preprocessing Complete!")
    print("=" * 70)
    
    # Calculate class distribution for each split
    for split_name, split_data, split_dir in splits:
        files = list(split_dir.glob("*.npz"))
        labels = []
        for f in files:
            data = np.load(f, allow_pickle=True)
            labels.append(int(data['label']))
        
        n_tumor = sum(labels)
        n_no_tumor = len(labels) - n_tumor
        
        print(f"\n{split_name.capitalize()} set:")
        print(f"  Total: {len(files)}")
        print(f"  Tumor: {n_tumor} ({n_tumor/len(files)*100:.1f}%)")
        print(f"  No tumor: {n_no_tumor} ({n_no_tumor/len(files)*100:.1f}%)")
    
    # Storage info
    output_files = list(output_path.rglob("*.npz"))
    if output_files:
        total_size = sum(f.stat().st_size for f in output_files)
        print(f"\nTotal files: {len(output_files)}")
        print(f"Total size: {total_size / (1024**2):.2f} MB")
    
    print(f"\nOutput directory: {output_path.absolute()}")
    
    print("\nNext steps:")
    print("1. Verify the preprocessed data:")
    print(f"   python -c \"from src.data.multi_source_dataset import *; test_kaggle_data('{output_dir}')\"")
    print("2. Create MultiSourceDataset combining BraTS + Kaggle")
    print("3. Start multi-task training")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Kaggle Brain MRI dataset to unified format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings
  python src/data/preprocess_kaggle_unified.py

  # Custom split ratios
  python src/data/preprocess_kaggle_unified.py \\
      --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

  # Custom directories
  python src/data/preprocess_kaggle_unified.py \\
      --input data/raw/kaggle_brain_mri \\
      --output data/processed/kaggle_unified
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/kaggle_brain_mri",
        help="Input directory with 'yes/' and 'no/' folders",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/kaggle_unified",
        help="Output directory for processed .npz files",
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Target size for images (default: 256 256)",
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70)",
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"✗ Error: Split ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)
    
    process_kaggle_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=tuple(args.target_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
