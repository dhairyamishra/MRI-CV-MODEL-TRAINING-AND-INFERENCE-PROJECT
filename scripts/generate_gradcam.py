"""
Convenience script to generate Grad-CAM visualizations.

This script generates Grad-CAM heatmaps for sample images to understand
what the model is focusing on.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.grad_cam import generate_gradcam_visualizations

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations",
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
        '--num_samples',
        type=int,
        default=16,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SliceWise - Grad-CAM Visualization")
    print("="*60)
    print(f"\nConfiguration: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of samples: {args.num_samples}\n")
    
    # Generate visualizations
    generate_gradcam_visualizations(
        args.config,
        args.checkpoint,
        args.num_samples,
        args.save_dir
    )
    
    print("\n" + "="*60)
    print("Grad-CAM Generation Complete!")
    print("="*60)
