"""
Helper script to split BraTS 2D data into train/val/test sets.

Usage:
    python scripts/split_brats_data.py
    python scripts/split_brats_data.py --input data/processed/brats2d_full
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.split_brats import split_brats_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Split BraTS 2D data into train/val/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split data/processed/brats2d (default)
  python scripts/split_brats_data.py
  
  # Split data/processed/brats2d_full
  python scripts/split_brats_data.py --input data/processed/brats2d_full --output data/processed/brats2d_full
  
  # Custom split ratios
  python scripts/split_brats_data.py --train 0.8 --val 0.1 --test 0.1
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/brats2d",
        help="Input directory with .npz files (default: data/processed/brats2d)"
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
    
    print("=" * 80)
    print("BraTS 2D Data Splitting")
    print("=" * 80)
    print(f"\nInput directory: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: Train={args.train}, Val={args.val}, Test={args.test}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Check if input directory exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    # Count files
    npz_files = list(input_path.glob("*.npz"))
    if len(npz_files) == 0:
        print(f"❌ Error: No .npz files found in {args.input}")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} .npz files")
    
    # Run splitting
    try:
        split_brats_dataset(
            input_dir=args.input,
            output_dir=output_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )
        
        print("\n" + "=" * 80)
        print("✓ Data splitting complete!")
        print("=" * 80)
        print(f"\nData split into:")
        print(f"  - {output_dir}/train/")
        print(f"  - {output_dir}/val/")
        print(f"  - {output_dir}/test/")
        print("\nYou can now proceed with training!")
        
    except Exception as e:
        print(f"\n❌ Error during splitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
