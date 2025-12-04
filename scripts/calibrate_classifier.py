#!/usr/bin/env python3
"""
Helper script to calibrate trained classifier using temperature scaling.

Usage:
    python scripts/calibrate_classifier.py
    python scripts/calibrate_classifier.py --checkpoint checkpoints/cls/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.kaggle_mri_dataset import KaggleBrainMRIDataset
from src.models.classifier import create_classifier
from src.eval.calibration import calibrate_classifier, plot_reliability_diagram
import torch.nn.functional as F
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Calibrate classifier using temperature scaling")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/cls/best_model.pth",
        help="Path to classifier checkpoint",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/kaggle/val",
        help="Validation data directory",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/calibration",
        help="Output directory for calibration results",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Classifier Calibration")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get model config
    config = checkpoint.get('config', {})
    model_name = config.get('model', {}).get('name', 'efficientnet_b0')
    
    # Map model name to base name (efficientnet_b0 -> efficientnet)
    if 'efficientnet' in model_name.lower():
        model_base_name = 'efficientnet'
    elif 'convnext' in model_name.lower():
        model_base_name = 'convnext'
    else:
        model_base_name = model_name
    
    # Create model
    print(f"Creating model: {model_name} (base: {model_base_name})...")
    model = create_classifier(model_base_name, num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load validation data
    print(f"\nLoading validation data from {args.data_dir}...")
    val_dataset = KaggleBrainMRIDataset(args.data_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Calibrate
    print("\nCalibrating model...")
    temp_scaler, metrics = calibrate_classifier(model, val_loader, device=args.device)
    
    # Print results
    print("\n" + "=" * 70)
    print("Calibration Results")
    print("=" * 70)
    print(f"\nLearned Temperature: {metrics['temperature']:.4f}")
    print(f"\nBefore Calibration:")
    print(f"  NLL:   {metrics['nll_before']:.4f}")
    print(f"  ECE:   {metrics['ece_before']:.4f}")
    print(f"  Brier: {metrics['brier_before']:.4f}")
    print(f"\nAfter Calibration:")
    print(f"  NLL:   {metrics['nll_after']:.4f}")
    print(f"  ECE:   {metrics['ece_after']:.4f}")
    print(f"  Brier: {metrics['brier_after']:.4f}")
    print(f"\nImprovement:")
    print(f"  ECE:   {(metrics['ece_before'] - metrics['ece_after']):.4f} ({(1 - metrics['ece_after']/metrics['ece_before'])*100:.1f}% reduction)")
    print(f"  Brier: {(metrics['brier_before'] - metrics['brier_after']):.4f}")
    
    # Save temperature scaler
    scaler_path = output_path / 'temperature_scaler.pth'
    torch.save({
        'temperature': temp_scaler.temperature.item(),
        'state_dict': temp_scaler.state_dict(),
        'metrics': metrics,
    }, scaler_path)
    print(f"\nSaved temperature scaler to: {scaler_path}")
    
    # Save metrics
    metrics_path = output_path / 'calibration_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")
    
    # Generate reliability diagrams
    print("\nGenerating reliability diagrams...")
    
    # Collect predictions
    model.eval()
    model = model.to(args.device)
    temp_scaler = temp_scaler.to(args.device)  # Move scaler to same device
    
    all_probs_before = []
    all_probs_after = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(args.device)
            logits = model(images)
            
            # Before calibration
            probs_before = F.softmax(logits, dim=1)
            
            # After calibration
            logits_scaled = temp_scaler(logits)
            probs_after = F.softmax(logits_scaled, dim=1)
            
            all_probs_before.append(probs_before[:, 1].cpu().numpy())  # Prob of tumor class
            all_probs_after.append(probs_after[:, 1].cpu().numpy())
            all_labels.append(labels.numpy())
    
    probs_before = np.concatenate(all_probs_before)
    probs_after = np.concatenate(all_probs_after)
    labels = np.concatenate(all_labels)
    
    # Plot before calibration
    fig_before = plot_reliability_diagram(
        probs_before,
        labels,
        title="Before Calibration",
        save_path=output_path / 'reliability_before.png'
    )
    print(f"  Saved: {output_path / 'reliability_before.png'}")
    
    # Plot after calibration
    fig_after = plot_reliability_diagram(
        probs_after,
        labels,
        title="After Temperature Scaling",
        save_path=output_path / 'reliability_after.png'
    )
    print(f"  Saved: {output_path / 'reliability_after.png'}")
    
    print("\n" + "=" * 70)
    print("✓ Calibration Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_path}")
    print("\nNext steps:")
    print("  1. Review reliability diagrams")
    print("  2. Use calibrated model in inference:")
    print("     - Load temperature scaler")
    print("     - Apply: probs = softmax(logits / temperature)")


if __name__ == "__main__":
    main()
