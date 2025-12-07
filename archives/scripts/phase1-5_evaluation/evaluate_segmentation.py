#!/usr/bin/env python3
"""
Helper script to evaluate U-Net segmentation model.

Usage:
    # Evaluate on validation set
    python scripts/evaluate_segmentation.py

    # Evaluate on test set
    python scripts/evaluate_segmentation.py --split test

    # Custom checkpoint and output
    python scripts/evaluate_segmentation.py \
        --checkpoint checkpoints/seg/my_model.pth \
        --output-dir outputs/seg/my_eval
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.eval_seg2d import evaluate_segmentation


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on validation set (default)
  python scripts/evaluate_segmentation.py

  # Evaluate on test set
  python scripts/evaluate_segmentation.py --split test

  # Custom checkpoint
  python scripts/evaluate_segmentation.py --checkpoint checkpoints/seg/epoch_50.pth

  # More visualizations
  python scripts/evaluate_segmentation.py --max-vis 50

  # Different threshold
  python scripts/evaluate_segmentation.py --threshold 0.6
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seg/best_model.pth",
        help="Path to model checkpoint (default: checkpoints/seg/best_model.pth)",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom data directory (default: data/processed/brats2d/{split})",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/seg/evaluation_{split})",
    )
    
    # Inference parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarization (default: 0.5)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    
    parser.add_argument(
        "--max-vis",
        type=int,
        default=20,
        help="Maximum number of visualizations to save (default: 20)",
    )
    
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualization saving",
    )
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = f"data/processed/brats2d/{args.split}"
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"outputs/seg/evaluation_{args.split}"
    
    # Print configuration
    print("=" * 70)
    print("Segmentation Evaluation - Helper Script")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Split:        {args.split}")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Threshold:    {args.threshold}")
    print(f"  Device:       {args.device}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Max vis:      {args.max_vis if not args.no_vis else 'disabled'}")
    print()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        print(f"\nAvailable checkpoints:")
        checkpoint_dir = Path("checkpoints/seg")
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob("*.pth"):
                print(f"  - {ckpt}")
        else:
            print(f"  No checkpoints found in {checkpoint_dir}")
        sys.exit(1)
    
    # Check if data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Error: Data directory not found at {args.data_dir}")
        print(f"\nAvailable splits:")
        data_root = Path("data/processed/brats2d")
        if data_root.exists():
            for split_dir in data_root.iterdir():
                if split_dir.is_dir():
                    print(f"  - {split_dir}")
        else:
            print(f"  No data found in {data_root}")
        sys.exit(1)
    
    # Run evaluation
    try:
        evaluate_segmentation(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            threshold=args.threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_visualizations=not args.no_vis,
            max_visualizations=args.max_vis,
        )
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
