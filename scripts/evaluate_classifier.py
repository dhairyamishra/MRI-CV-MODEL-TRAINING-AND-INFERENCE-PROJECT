"""
Convenience script to evaluate the trained classifier.

This script runs evaluation on the test set and generates all metrics
and visualizations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.eval_cls import evaluate_classifier

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate brain tumor classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_cls.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/cls/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/classification/evaluation',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SliceWise - Classifier Evaluation")
    print("="*60)
    print(f"\nConfiguration: {args.config}")
    print(f"Checkpoint: {args.checkpoint}\n")
    
    # Run evaluation
    metrics = evaluate_classifier(args.config, args.checkpoint)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
