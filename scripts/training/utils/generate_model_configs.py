"""
Generate model_config.json files for existing checkpoints.

This script creates model configuration files for all trained models
to prevent architecture mismatches in the future.

Usage:
    python scripts/generate_model_configs.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_config import ModelConfig


def generate_configs():
    """Generate model_config.json for all checkpoint directories."""
    
    print("=" * 80)
    print("Generating Model Configuration Files")
    print("=" * 80)
    
    # Define checkpoint directories and their configurations
    checkpoints = {
        "multitask_seg_warmup": ModelConfig(
            base_filters=64,
            depth=4,
            in_channels=1,
            seg_out_channels=1,
            cls_num_classes=2,
            model_type="multitask",
            description="Multi-task model - Segmentation warm-up phase",
            trained_on="BraTS 2020 (segmentation only)",
            training_config="configs/multitask_seg_warmup.yaml"
        ),
        "multitask_cls_head": ModelConfig(
            base_filters=64,
            depth=4,
            in_channels=1,
            seg_out_channels=1,
            cls_num_classes=2,
            model_type="multitask",
            description="Multi-task model - Classification head training phase",
            trained_on="BraTS 2020 + Kaggle Brain MRI (classification only)",
            training_config="configs/multitask_cls_head_quick_test.yaml"
        ),
        "multitask_joint": ModelConfig(
            base_filters=64,
            depth=4,
            in_channels=1,
            seg_out_channels=1,
            cls_num_classes=2,
            model_type="multitask",
            description="Multi-task model - Joint fine-tuning (FINAL)",
            trained_on="BraTS 2020 + Kaggle Brain MRI (both tasks)",
            training_config="configs/multitask_joint_quick_test.yaml"
        ),
    }
    
    # Generate configs
    checkpoints_dir = project_root / "checkpoints"
    generated_count = 0
    
    for checkpoint_name, config in checkpoints.items():
        checkpoint_dir = checkpoints_dir / checkpoint_name
        
        if not checkpoint_dir.exists():
            print(f"\n⚠ Checkpoint directory not found: {checkpoint_dir}")
            print(f"  Skipping {checkpoint_name}")
            continue
        
        # Check if best_model.pth exists
        best_model = checkpoint_dir / "best_model.pth"
        if not best_model.exists():
            print(f"\n⚠ best_model.pth not found in: {checkpoint_dir}")
            print(f"  Skipping {checkpoint_name}")
            continue
        
        # Save config
        config_path = checkpoint_dir / "model_config.json"
        config.save(config_path)
        generated_count += 1
        
        print(f"\n✓ Generated config for: {checkpoint_name}")
        print(f"  Location: {config_path}")
        print(f"  Architecture: base_filters={config.base_filters}, depth={config.depth}")
    
    print("\n" + "=" * 80)
    print(f"✅ Generated {generated_count} configuration files")
    print("=" * 80)
    
    if generated_count > 0:
        print("\nThese configuration files will be automatically loaded during inference,")
        print("preventing architecture mismatches in the future.")
        print("\nNext time you load a model, it will use:")
        print("  config = ModelConfig.from_checkpoint_dir(checkpoint_dir)")
        print("  model = create_multi_task_model(**config.get_model_params())")


if __name__ == "__main__":
    generate_configs()
