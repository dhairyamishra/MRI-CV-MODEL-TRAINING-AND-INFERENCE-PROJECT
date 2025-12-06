"""
Helper script to split Kaggle data into train/val/test sets.

Performs stratified splitting to maintain class balance.

Usage:
    python scripts/split_kaggle_data.py
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def split_kaggle_dataset(
    input_dir: str = "data/processed/kaggle",
    output_dir: str = "data/processed/kaggle",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split Kaggle dataset into train/val/test sets with stratification.
    
    Args:
        input_dir: Directory containing .npz files
        output_dir: Directory to create train/val/test subdirectories
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("Kaggle Dataset Splitting (Stratified)")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Input:  {input_path.absolute()}")
    print(f"  Output: {output_path.absolute()}")
    print(f"  Split ratios: {train_ratio*100:.0f}% train / {val_ratio*100:.0f}% val / {test_ratio*100:.0f}% test")
    print(f"  Random seed: {seed}")
    print()
    
    # Get all .npz files and group by class
    print("Grouping files by class...")
    no_tumor_files = sorted([f for f in input_path.glob("*.npz") if f.stem.startswith("no_")])
    yes_tumor_files = sorted([f for f in input_path.glob("*.npz") if f.stem.startswith("yes_")])
    
    print(f"  No tumor: {len(no_tumor_files)} files")
    print(f"  Yes tumor: {len(yes_tumor_files)} files")
    print(f"  Total: {len(no_tumor_files) + len(yes_tumor_files)} files")
    print()
    
    # Set random seed
    np.random.seed(seed)
    
    # Shuffle each class
    np.random.shuffle(no_tumor_files)
    np.random.shuffle(yes_tumor_files)
    
    # Calculate split sizes for each class
    n_no = len(no_tumor_files)
    n_yes = len(yes_tumor_files)
    
    # No tumor splits
    n_no_train = int(n_no * train_ratio)
    n_no_val = int(n_no * val_ratio)
    n_no_test = n_no - n_no_train - n_no_val
    
    # Yes tumor splits
    n_yes_train = int(n_yes * train_ratio)
    n_yes_val = int(n_yes * val_ratio)
    n_yes_test = n_yes - n_yes_train - n_yes_val
    
    # Split files
    no_train = no_tumor_files[:n_no_train]
    no_val = no_tumor_files[n_no_train:n_no_train + n_no_val]
    no_test = no_tumor_files[n_no_train + n_no_val:]
    
    yes_train = yes_tumor_files[:n_yes_train]
    yes_val = yes_tumor_files[n_yes_train:n_yes_train + n_yes_val]
    yes_test = yes_tumor_files[n_yes_train + n_yes_val:]
    
    # Combine classes
    train_files = no_train + yes_train
    val_files = no_val + yes_val
    test_files = no_test + yes_test
    
    print("Split sizes:")
    print(f"  Train: {len(train_files)} files ({len(no_train)} no, {len(yes_train)} yes)")
    print(f"  Val:   {len(val_files)} files ({len(no_val)} no, {len(yes_val)} yes)")
    print(f"  Test:  {len(test_files)} files ({len(no_test)} no, {len(yes_test)} yes)")
    print()
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to splits
    print("Copying files to splits...")
    
    print("  Copying train files...")
    for file_path in tqdm(train_files, desc="train"):
        shutil.copy2(file_path, train_dir / file_path.name)
    
    print("  Copying val files...")
    for file_path in tqdm(val_files, desc="val  "):
        shutil.copy2(file_path, val_dir / file_path.name)
    
    print("  Copying test files...")
    for file_path in tqdm(test_files, desc="test "):
        shutil.copy2(file_path, test_dir / file_path.name)
    
    print()
    print("=" * 70)
    print("✓ Splitting Complete!")
    print("=" * 70)
    print()
    print("Output directories:")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")
    print(f"  Test:  {test_dir}")
    print()
    print("Verification:")
    print(f"  Train files: {len(list(train_dir.glob('*.npz')))}")
    print(f"  Val files:   {len(list(val_dir.glob('*.npz')))}")
    print(f"  Test files:  {len(list(test_dir.glob('*.npz')))}")
    print(f"  Total:       {len(list(train_dir.glob('*.npz'))) + len(list(val_dir.glob('*.npz'))) + len(list(test_dir.glob('*.npz')))}")


def main():
    parser = argparse.ArgumentParser(
        description="Split Kaggle data into train/val/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/kaggle",
        help="Input directory with .npz files (default: data/processed/kaggle)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same as input)"
    )
    
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)"
    )
    
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Error: Split ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output if args.output else args.input
    
    # Run splitting
    try:
        split_kaggle_dataset(
            input_dir=args.input,
            output_dir=output_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )
        
        print("\nYou can now proceed with Phase 2.2 training!")
        
    except Exception as e:
        print(f"\n❌ Error during splitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
