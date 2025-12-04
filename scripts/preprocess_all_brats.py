#!/usr/bin/env python3
"""
Helper script to preprocess all BraTS 2020 patients and split into train/val/test.

This script automates the complete preprocessing pipeline:
1. Extract 2D slices from 3D NIfTI volumes
2. Normalize and filter slices
3. Split into train/val/test sets

Usage:
    # Process all 988 patients (takes 2-4 hours)
    python scripts/preprocess_all_brats.py

    # Process first 100 patients (quick test)
    python scripts/preprocess_all_brats.py --num-patients 100

    # Use different modality
    python scripts/preprocess_all_brats.py --modality t1

    # Custom paths
    python scripts/preprocess_all_brats.py \
        --input data/raw/brats2020/MICCAI_BraTS2020_TrainingData \
        --output data/processed/brats2d_full
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command as list of strings
        description: Description of the command
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {description} failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess all BraTS patients and create train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 988 patients (recommended for production)
  python scripts/preprocess_all_brats.py

  # Quick test with 50 patients
  python scripts/preprocess_all_brats.py --num-patients 50

  # Use T1 modality instead of FLAIR
  python scripts/preprocess_all_brats.py --modality t1

  # Custom input/output paths
  python scripts/preprocess_all_brats.py \
      --input /path/to/brats2020 \
      --output data/processed/my_brats

  # Skip splitting (if you want to split manually later)
  python scripts/preprocess_all_brats.py --no-split

Estimated Time:
  - 10 patients:   ~2 minutes
  - 100 patients:  ~20 minutes
  - 988 patients:  ~2-4 hours (depends on CPU/disk speed)
        """
    )
    
    # Input/Output paths
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/brats2020/MICCAI_BraTS2020_TrainingData",
        help="Input directory with BraTS NIfTI files (default: data/raw/brats2020/MICCAI_BraTS2020_TrainingData)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/brats2d_full",
        help="Output directory for processed slices (default: data/processed/brats2d_full)",
    )
    
    # Processing parameters
    parser.add_argument(
        "--num-patients",
        type=int,
        default=988,
        help="Number of patients to process (default: 988 = all)",
    )
    
    parser.add_argument(
        "--modality",
        type=str,
        default="flair",
        choices=["flair", "t1", "t1ce", "t2"],
        help="MRI modality to use (default: flair)",
    )
    
    parser.add_argument(
        "--normalization",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "percentile"],
        help="Normalization method (default: zscore)",
    )
    
    parser.add_argument(
        "--min-tumor-pixels",
        type=int,
        default=10,
        help="Minimum tumor pixels to keep slice (default: 10)",
    )
    
    # Splitting parameters
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip train/val/test splitting",
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio (default: 0.70)",
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Error: Split ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)
    
    # Print configuration
    print("=" * 70)
    print("BraTS Preprocessing Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input dir:        {args.input}")
    print(f"  Output dir:       {args.output}")
    print(f"  Num patients:     {args.num_patients}")
    print(f"  Modality:         {args.modality}")
    print(f"  Normalization:    {args.normalization}")
    print(f"  Min tumor pixels: {args.min_tumor_pixels}")
    
    if not args.no_split:
        print(f"\nSplit ratios:")
        print(f"  Train: {args.train_ratio:.0%}")
        print(f"  Val:   {args.val_ratio:.0%}")
        print(f"  Test:  {args.test_ratio:.0%}")
        print(f"  Seed:  {args.seed}")
    else:
        print(f"\nSplitting: Disabled")
    
    # Check if input directory exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n❌ Error: Input directory not found: {args.input}")
        print(f"\nPlease download BraTS 2020 dataset first:")
        print(f"  python scripts/download_brats_data.py")
        sys.exit(1)
    
    # Count available patients
    patient_dirs = list(input_path.glob("BraTS20_*"))
    available_patients = len(patient_dirs)
    print(f"\nFound {available_patients} patients in {args.input}")
    
    if args.num_patients > available_patients:
        print(f"⚠️  Warning: Requested {args.num_patients} patients but only {available_patients} available")
        print(f"   Will process {available_patients} patients")
        args.num_patients = available_patients
    
    # Estimate time
    estimated_minutes = args.num_patients * 0.12  # ~7 seconds per patient
    print(f"\nEstimated time: ~{estimated_minutes:.0f} minutes")
    
    # Confirm
    print(f"\n{'=' * 70}")
    response = input("Proceed with preprocessing? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        sys.exit(0)
    
    start_time = time.time()
    
    # Step 1: Preprocess slices
    preprocess_cmd = [
        "python",
        "src/data/preprocess_brats_2d.py",
        "--input", args.input,
        "--output", args.output,
        "--modality", args.modality,
        "--normalization", args.normalization,
        "--min-tumor-pixels", str(args.min_tumor_pixels),
        "--num-patients", str(args.num_patients),
    ]
    
    if not run_command(preprocess_cmd, "Step 1: Preprocessing 3D → 2D slices"):
        print("\n❌ Preprocessing failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Split data (if not disabled)
    if not args.no_split:
        split_cmd = [
            "python",
            "src/data/split_brats.py",
            "--input", args.output,
            "--train-ratio", str(args.train_ratio),
            "--val-ratio", str(args.val_ratio),
            "--test-ratio", str(args.test_ratio),
            "--seed", str(args.seed),
        ]
        
        if not run_command(split_cmd, "Step 2: Splitting into train/val/test"):
            print("\n❌ Splitting failed. Exiting.")
            sys.exit(1)
    
    # Summary
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    
    print(f"\n{'=' * 70}")
    print("✓ Preprocessing Complete!")
    print(f"{'=' * 70}")
    print(f"\nTotal time: {elapsed_minutes:.1f} minutes")
    print(f"Output directory: {args.output}")
    
    if not args.no_split:
        print(f"\nDataset splits created:")
        print(f"  Train: {args.output}/train")
        print(f"  Val:   {args.output}/val")
        print(f"  Test:  {args.output}/test")
        
        print(f"\n{'=' * 70}")
        print("Next Steps:")
        print(f"{'=' * 70}")
        print(f"\n1. Verify the dataset:")
        print(f"   python src/data/brats2d_dataset.py")
        print(f"\n2. Train the model:")
        print(f"   python scripts/train_segmentation.py")
        print(f"\n3. Evaluate the model:")
        print(f"   python scripts/evaluate_segmentation.py")
    else:
        print(f"\n{'=' * 70}")
        print("Next Steps:")
        print(f"{'=' * 70}")
        print(f"\n1. Split the data:")
        print(f"   python src/data/split_brats.py --input {args.output}")
        print(f"\n2. Train the model:")
        print(f"   python scripts/train_segmentation.py")


if __name__ == "__main__":
    main()
