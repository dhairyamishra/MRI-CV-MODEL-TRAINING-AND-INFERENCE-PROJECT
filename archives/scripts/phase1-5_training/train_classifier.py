"""
Convenience script to train the brain tumor classifier.

This script provides a simple interface to start training with default
or custom configurations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.train_cls import train_classifier

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train brain tumor classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_cls.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SliceWise - Brain Tumor Classification Training")
    print("="*60)
    print(f"\nConfiguration: {args.config}\n")
    
    # Start training
    train_classifier(args.config)
