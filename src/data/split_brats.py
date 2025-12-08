"""
Split BraTS 2D slices into train/val/test sets.

Performs patient-level splitting to avoid data leakage.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def get_patient_slices(data_dir: Path) -> Dict[str, List[Path]]:
    """
    Group slices by patient ID.
    
    Args:
        data_dir: Directory containing .npz files
    
    Returns:
        Dictionary mapping patient_id to list of slice file paths
    """
    patient_slices = {}
    
    for file_path in data_dir.glob("*.npz"):
        # Extract patient ID from filename (e.g., "BraTS20_Training_001_slice_075.npz")
        patient_id = "_".join(file_path.stem.split("_")[:-2])
        
        if patient_id not in patient_slices:
            patient_slices[patient_id] = []
        
        patient_slices[patient_id].append(file_path)
    
    return patient_slices


def split_brats_dataset(
    input_dir: str = "data/processed/brats2d",
    output_dir: str = "data/processed/brats2d",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split BraTS dataset into train/val/test sets.
    
    Args:
        input_dir: Directory containing preprocessed .npz files
        output_dir: Output directory (will create train/val/test subdirs)
        train_ratio: Fraction of patients for training
        val_ratio: Fraction of patients for validation
        test_ratio: Fraction of patients for testing
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("BraTS Dataset Splitting (Patient-Level)")
    print("=" * 70)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"✗ Error: Input directory not found: {input_path}")
        sys.exit(1)
    
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        print(f"✗ Error: Ratios must sum to 1.0")
        print(f"   Got: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Input:  {input_path.absolute()}")
    print(f"  Output: {output_path.absolute()}")
    print(f"  Split ratios: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    print(f"  Random seed: {seed}")
    
    # Group slices by patient
    print("\nGrouping slices by patient...")
    patient_slices = get_patient_slices(input_path)
    
    patient_ids = sorted(patient_slices.keys())
    total_patients = len(patient_ids)
    total_slices = sum(len(slices) for slices in patient_slices.values())
    
    print(f"  Found {total_patients} patients")
    print(f"  Total slices: {total_slices}")
    print(f"  Avg slices per patient: {total_slices / total_patients:.1f}")
    
    # Shuffle patients
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    
    # Calculate split indices
    n_train = int(total_patients * train_ratio)
    n_val = int(total_patients * val_ratio)
    
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]
    
    splits = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients,
    }
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Val:   {len(val_patients)} patients")
    print(f"  Test:  {len(test_patients)} patients")
    
    # Create output directories
    for split_name in ['train', 'val', 'test']:
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective splits
    print("\nCopying files to splits...")
    
    for split_name, patient_list in splits.items():
        split_dir = output_path / split_name
        slice_count = 0
        
        for patient_id in tqdm(patient_list, desc=f"{split_name:5s}"):
            for slice_file in patient_slices[patient_id]:
                dest_file = split_dir / slice_file.name
                shutil.copy2(slice_file, dest_file)
                slice_count += 1
        
        print(f"  {split_name:5s}: {slice_count:,} slices from {len(patient_list)} patients")
    
    # Summary
    print("\n" + "=" * 70)
    print("[OK] Splitting Complete!")
    print("=" * 70)
    
    print(f"\nOutput directories:")
    print(f"  Train: {output_path / 'train'}")
    print(f"  Val:   {output_path / 'val'}")
    print(f"  Test:  {output_path / 'test'}")
    
    # Verify splits
    train_files = len(list((output_path / 'train').glob("*.npz")))
    val_files = len(list((output_path / 'val').glob("*.npz")))
    test_files = len(list((output_path / 'test').glob("*.npz")))
    
    print(f"\nVerification:")
    print(f"  Train files: {train_files:,}")
    print(f"  Val files:   {val_files:,}")
    print(f"  Test files:  {test_files:,}")
    print(f"  Total:       {train_files + val_files + test_files:,}")
    
    print("\nNext steps:")
    print("1. Test the dataset:")
    print(f"   python src/data/brats2d_dataset.py")
    print("2. Start training:")
    print(f"   python scripts/train_segmentation.py")


def main():
    parser = argparse.ArgumentParser(
        description="Split BraTS 2D slices into train/val/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default split (70/15/15)
  python src/data/split_brats.py --input data/processed/brats2d

  # Custom split ratios
  python src/data/split_brats.py \\
      --train-ratio 0.8 \\
      --val-ratio 0.1 \\
      --test-ratio 0.1

  # Different random seed
  python src/data/split_brats.py --seed 123
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/brats2d",
        help="Input directory with preprocessed .npz files",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
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
    
    # Use input dir as output if not specified
    output_dir = args.output if args.output else args.input
    
    split_brats_dataset(
        input_dir=args.input,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
