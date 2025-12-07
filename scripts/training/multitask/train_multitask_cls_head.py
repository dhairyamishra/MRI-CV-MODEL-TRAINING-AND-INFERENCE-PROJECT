"""
Helper script to launch Phase 2.2: Classification Head Training.

This script trains the classification head with a FROZEN encoder on BraTS + Kaggle data.

Usage:
    python scripts/train_multitask_cls_head.py \
        --config configs/multitask_cls_head_quick_test.yaml \
        --encoder-init checkpoints/multitask_seg_warmup/best_model.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.training.train_multitask_cls_head import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2.2: Classification Head Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config (10 epochs)
  python scripts/train_multitask_cls_head.py \\
    --config configs/multitask_cls_head_quick_test.yaml \\
    --encoder-init checkpoints/multitask_seg_warmup/best_model.pth
  
  # Resume from checkpoint
  python scripts/train_multitask_cls_head.py \\
    --config configs/multitask_cls_head_quick_test.yaml \\
    --encoder-init checkpoints/multitask_seg_warmup/best_model.pth \\
    --resume checkpoints/multitask_cls_head_test/last_model.pth
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multitask_cls_head_quick_test.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--encoder-init",
        type=str,
        required=True,
        help="Path to Phase 2.1 checkpoint for encoder initialization (REQUIRED)"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/multitask_cls_head",
        help="Directory to save checkpoints"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
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
            for config_file in sorted(configs_dir.glob("multitask_cls*.yaml")):
                print(f"  - {config_file}")
        sys.exit(1)
    
    # Validate encoder checkpoint exists
    encoder_path = Path(args.encoder_init)
    if not encoder_path.exists():
        print(f"❌ Error: Encoder checkpoint not found: {args.encoder_init}")
        print(f"\nMake sure Phase 2.1 completed successfully.")
        print(f"Expected checkpoint: checkpoints/multitask_seg_warmup/best_model.pth")
        sys.exit(1)
    
    print("=" * 80)
    print("Phase 2.2: Classification Head Training")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"Encoder init: {args.encoder_init}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print()
    
    # Prepare arguments for training script
    train_args = [
        "--config", args.config,
        "--encoder-init", args.encoder_init,
        "--checkpoint-dir", args.checkpoint_dir,
    ]
    
    if args.resume:
        train_args.extend(["--resume", args.resume])
    
    # Override sys.argv for the training script
    original_argv = sys.argv
    sys.argv = ["train_multitask_cls_head.py"] + train_args
    
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
    print("✓ Phase 2.2 Complete!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"Best model: {args.checkpoint_dir}/best_model.pth")
    print("\nNext Steps:")
    print("  1. Evaluate classification performance on test set")
    print("  2. Proceed to Phase 2.3: Joint Fine-Tuning")
    print(f"     python scripts/train_multitask_joint.py --init-from {args.checkpoint_dir}/best_model.pth")


if __name__ == "__main__":
    main()
