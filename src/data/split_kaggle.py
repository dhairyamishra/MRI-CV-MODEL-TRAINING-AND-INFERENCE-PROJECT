#!/usr/bin/env python3
"""
Create train/val/test splits for Kaggle Brain MRI dataset.
Uses stratified sampling to maintain class balance.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def create_kaggle_splits(
    processed_dir: str = "data/processed/kaggle",
    output_dir: str = "data/processed/kaggle",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split Kaggle dataset into train/val/test sets.
    
    Args:
        processed_dir: Directory containing .npz files
        output_dir: Output directory for splits
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("Creating Train/Val/Test Splits for Kaggle Dataset")
    print("=" * 60)
    
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        print(f"Error: Ratios must sum to 1.0")
        print(f"  train={train_ratio}, val={val_ratio}, test={test_ratio}")
        print(f"  sum={train_ratio + val_ratio + test_ratio}")
        sys.exit(1)
    
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    
    if not processed_path.exists():
        print(f"Error: Processed directory not found: {processed_path}")
        print("Please run src/data/preprocess_kaggle.py first")
        sys.exit(1)
    
    # Get all .npz files
    npz_files = sorted(processed_path.glob("*.npz"))
    
    if not npz_files:
        print(f"Error: No .npz files found in {processed_path}")
        sys.exit(1)
    
    print(f"\nFound {len(npz_files)} .npz files")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Random seed: {seed}")
    
    # Load labels for stratification
    labels = []
    file_paths = []
    
    print("\nLoading labels...")
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        labels.append(int(data['label']))
        file_paths.append(npz_path)
    
    labels = np.array(labels)
    
    # Count classes
    n_tumor = np.sum(labels == 1)
    n_no_tumor = np.sum(labels == 0)
    
    print(f"\nClass distribution:")
    print(f"  - Tumor (1):     {n_tumor} ({n_tumor/len(labels)*100:.1f}%)")
    print(f"  - No tumor (0):  {n_no_tumor} ({n_no_tumor/len(labels)*100:.1f}%)")
    
    # First split: train vs (val + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        train_size=val_size,
        stratify=temp_labels,
        random_state=seed,
    )
    
    # Create split directories
    splits = {
        "train": (train_files, train_labels),
        "val": (val_files, val_labels),
        "test": (test_files, test_labels),
    }
    
    print("\n" + "=" * 60)
    print("Creating split directories and copying files...")
    print("=" * 60)
    
    for split_name, (files, split_labels) in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        n_tumor_split = np.sum(split_labels == 1)
        n_no_tumor_split = np.sum(split_labels == 0)
        
        print(f"\n[{split_name.upper()}]")
        print(f"  Total: {len(files)}")
        print(f"  - Tumor (1):     {n_tumor_split} ({n_tumor_split/len(files)*100:.1f}%)")
        print(f"  - No tumor (0):  {n_no_tumor_split} ({n_no_tumor_split/len(files)*100:.1f}%)")
        print(f"  Copying files to {split_dir}...")
        
        for file_path in files:
            dest = split_dir / file_path.name
            shutil.copy2(file_path, dest)
    
    # Summary
    print("\n" + "=" * 60)
    print("Split Creation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"\nDirectory structure:")
    print(f"  {output_path}/")
    print(f"  ├── train/  ({len(train_files)} files)")
    print(f"  ├── val/    ({len(val_files)} files)")
    print(f"  └── test/   ({len(test_files)} files)")
    
    print("\nNext steps:")
    print("1. Create PyTorch dataset class (kaggle_mri_dataset.py)")
    print("2. Verify splits with visualization notebook")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for Kaggle dataset"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/kaggle",
        help="Directory containing .npz files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/kaggle",
        help="Output directory for splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
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
    
    create_kaggle_splits(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
