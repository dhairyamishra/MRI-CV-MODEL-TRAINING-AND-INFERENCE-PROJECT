"""
Convenience script to train the brain tumor classifier on BraTS dataset.

This script trains classification using BraTS data with labels derived
from segmentation masks.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the BraTS classification dataset
from src.data.brats_classification_dataset import create_brats_classification_dataloaders

# Monkey-patch the kaggle dataloader with BraTS dataloader
import src.data.kaggle_mri_dataset
src.data.kaggle_mri_dataset.create_dataloaders = create_brats_classification_dataloaders

# Now import and run the standard training
from src.training.train_cls import train_classifier

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train brain tumor classifier on BraTS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_cls_brats.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SliceWise - Brain Tumor Classification Training (BraTS)")
    print("="*60)
    print(f"\nConfiguration: {args.config}")
    print("Dataset: BraTS 2020 (labels derived from segmentation masks)\n")
    
    # Start training
    train_classifier(args.config)
