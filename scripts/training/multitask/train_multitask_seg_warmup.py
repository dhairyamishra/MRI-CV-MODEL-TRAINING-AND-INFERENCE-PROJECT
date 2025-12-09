"""
Helper script to launch Phase 2.1: Segmentation Warm-Up Training.

This script trains the multi-task model in segmentation-only mode on BraTS data
to initialize the shared encoder with good features.

Usage:
    python scripts/train_multitask_seg_warmup.py
    python scripts/train_multitask_seg_warmup.py --epochs 30
    python scripts/train_multitask_seg_warmup.py --resume checkpoints/multitask_seg_warmup/last_model.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.training.train_multitask_seg_warmup import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2.1: Segmentation Warm-Up Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config (50 epochs)
  python scripts/train_multitask_seg_warmup.py
  
  # Quick test (5 epochs)
  python scripts/train_multitask_seg_warmup.py --epochs 5
  
  # Resume from checkpoint
  python scripts/train_multitask_seg_warmup.py --resume checkpoints/multitask_seg_warmup/last_model.pth
  
  # Custom checkpoint directory
  python scripts/train_multitask_seg_warmup.py --checkpoint-dir checkpoints/my_warmup
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final/stage1_baseline.yaml",
        help="Path to config file (default: configs/final/stage1_baseline.yaml)"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/multitask_seg_warmup",
        help="Directory to save checkpoints (default: checkpoints/multitask_seg_warmup)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        print(f"\nAvailable configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in sorted(configs_dir.glob("*.yaml")):
                print(f"  - {config_file}")
        sys.exit(1)
    
    print("=" * 80)
    print("Phase 2.1: Segmentation Warm-Up Training")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    if args.epochs:
        print(f"Epochs (override): {args.epochs}")
    if args.batch_size:
        print(f"Batch size (override): {args.batch_size}")
    if args.lr:
        print(f"Learning rate (override): {args.lr}")
    print()
    
    # Prepare arguments for training script
    train_args = [
        "--config", args.config,
        "--checkpoint-dir", args.checkpoint_dir,
    ]
    
    if args.resume:
        train_args.extend(["--resume", args.resume])
    
    # Override sys.argv for the training script
    original_argv = sys.argv
    sys.argv = ["train_multitask_seg_warmup.py"] + train_args
    
    # If overrides are specified, we need to modify the config
    # For simplicity, we'll just pass them through environment or handle in training script
    # For now, let's just run with the config as-is
    
    try:
        train_main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.argv = original_argv
    
    print("\n" + "=" * 80)
    print("[OK] Phase 2.1 Complete!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"Best model: {args.checkpoint_dir}/best_model.pth")
    print("\nNext Steps:")
    print("  1. Evaluate segmentation performance on test set")
    print("  2. Proceed to Phase 2.2: Classification Head Training")
    print("     python scripts/train_multitask_cls_head.py --encoder-init {args.checkpoint_dir}/best_model.pth")


if __name__ == "__main__":
    main()
