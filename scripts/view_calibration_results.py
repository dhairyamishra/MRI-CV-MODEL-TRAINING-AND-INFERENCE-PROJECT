#!/usr/bin/env python3
"""
View calibration results - displays reliability diagrams.

Usage:
    python scripts/view_calibration_results.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def main():
    output_dir = Path("outputs/calibration")
    
    # Check if files exist
    before_img = output_dir / "reliability_before.png"
    after_img = output_dir / "reliability_after.png"
    metrics_file = output_dir / "calibration_metrics.json"
    
    if not before_img.exists() or not after_img.exists():
        print("❌ Calibration results not found!")
        print("   Run: python scripts/calibrate_classifier.py")
        return
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Load images
    img_before = Image.open(before_img)
    img_after = Image.open(after_img)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(img_before)
    axes[0].set_title(f"Before Calibration\nECE: {metrics['ece_before']:.4f}, Brier: {metrics['brier_before']:.4f}", 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_after)
    axes[1].set_title(f"After Temperature Scaling (T={metrics['temperature']:.2f})\nECE: {metrics['ece_after']:.4f}, Brier: {metrics['brier_after']:.4f}", 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(f"Classifier Calibration Results\nECE Reduction: {(1 - metrics['ece_after']/metrics['ece_before'])*100:.1f}%", 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Calibration Summary")
    print("=" * 70)
    print(f"\nTemperature: {metrics['temperature']:.4f}")
    print(f"\nBefore Calibration:")
    print(f"  ECE:   {metrics['ece_before']:.4f}")
    print(f"  Brier: {metrics['brier_before']:.4f}")
    print(f"  NLL:   {metrics['nll_before']:.4f}")
    print(f"\nAfter Calibration:")
    print(f"  ECE:   {metrics['ece_after']:.4f}")
    print(f"  Brier: {metrics['brier_after']:.4f}")
    print(f"  NLL:   {metrics['nll_after']:.4f}")
    print(f"\nImprovement:")
    print(f"  ECE:   {(metrics['ece_before'] - metrics['ece_after']):.4f} ({(1 - metrics['ece_after']/metrics['ece_before'])*100:.1f}% reduction)")
    print(f"  Brier: {(metrics['brier_before'] - metrics['brier_after']):.4f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
