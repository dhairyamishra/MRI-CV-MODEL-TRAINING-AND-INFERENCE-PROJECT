"""
Helper script to launch Phase 2.3: Joint Fine-Tuning.

This script trains both segmentation and classification tasks jointly with:
- Unfrozen encoder (all parameters trainable)
- Differential learning rates (encoder vs decoder/cls_head)
- Combined loss function

Usage:
    python scripts/train_multitask_joint.py \
        --config configs/multitask_joint_quick_test.yaml \
        --init-from checkpoints/multitask_cls_head/best_model.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.train_multitask_joint import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2.3: Joint Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config (10 epochs)
  python scripts/train_multitask_joint.py \\
    --config configs/multitask_joint_quick_test.yaml \\
    --init-from checkpoints/multitask_cls_head/best_model.pth
  
  # Resume from checkpoint
  python scripts/train_multitask_joint.py \\
    --config configs/multitask_joint_quick_test.yaml \\
    --init-from checkpoints/multitask_cls_head/best_model.pth \\
    --resume checkpoints/multitask_joint/last_model.pth
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multitask_joint_quick_test.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--init-from",
        type=str,
        required=True,
        help="Path to Phase 2.2 checkpoint for initialization (REQUIRED)"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/multitask_joint",
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
            for config_file in sorted(configs_dir.glob("multitask_joint*.yaml")):
                print(f"  - {config_file}")
        sys.exit(1)
    
    # Validate initialization checkpoint exists
    init_path = Path(args.init_from)
    if not init_path.exists():
        print(f"❌ Error: Initialization checkpoint not found: {args.init_from}")
        print(f"\nMake sure Phase 2.2 completed successfully.")
        print(f"Expected checkpoint: checkpoints/multitask_cls_head/best_model.pth")
        sys.exit(1)
    
    print("=" * 80)
    print("Phase 2.3: Joint Fine-Tuning")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"Initialize from: {args.init_from}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print()
    
    # Prepare arguments for training script
    train_args = [
        "--config", args.config,
        "--init-from", args.init_from,
        "--checkpoint-dir", args.checkpoint_dir,
    ]
    
    if args.resume:
        train_args.extend(["--resume", args.resume])
    
    # Override sys.argv for the training script
    original_argv = sys.argv
    sys.argv = ["train_multitask_joint.py"] + train_args
    
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
    print("✓ Phase 2.3 Complete!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"Best model: {args.checkpoint_dir}/best_model.pth")
    print("\nNext Steps:")
    print("  1. Evaluate on test set (both segmentation and classification)")
    print("  2. Compare with baseline models from Phase 2.1 and 2.2")
    print("  3. Generate visualizations (Grad-CAM, segmentation overlays)")
    print("  4. Create evaluation report")


if __name__ == "__main__":
    main()
