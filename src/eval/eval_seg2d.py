"""
Evaluation script for 2D brain tumor segmentation.

Computes metrics:
- Dice coefficient
- IoU (Jaccard index)
- Precision
- Recall
- F1 score

Generates visualizations and saves results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.brats2d_dataset import BraTS2DSliceDataset
from src.inference.infer_seg2d import SegmentationPredictor


def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calculate segmentation metrics.
    
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
    
    Returns:
        Dictionary of metrics
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # True positives, false positives, false negatives
    tp = np.sum(pred & target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    tn = np.sum(~pred & ~target)
    
    # Dice coefficient
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # IoU (Jaccard index)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Precision and Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # F1 score (same as Dice for binary)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Specificity
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def create_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create visualization overlay.
    
    Args:
        image: Input image (H, W)
        pred_mask: Predicted mask (H, W)
        gt_mask: Ground truth mask (H, W)
        alpha: Transparency for overlays
    
    Returns:
        RGB overlay image (H, W, 3)
    """
    # Normalize image to [0, 255]
    img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255)
    img_norm = img_norm.astype(np.uint8)
    
    # Convert to RGB
    overlay = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    
    # Create colored masks
    # Green for true positives
    # Red for false positives
    # Blue for false negatives
    
    tp = (pred_mask > 0) & (gt_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask > 0)
    
    # Apply colors
    overlay[tp] = (1 - alpha) * overlay[tp] + alpha * np.array([0, 255, 0])  # Green
    overlay[fp] = (1 - alpha) * overlay[fp] + alpha * np.array([255, 0, 0])  # Red
    overlay[fn] = (1 - alpha) * overlay[fn] + alpha * np.array([0, 0, 255])  # Blue
    
    return overlay.astype(np.uint8)


def evaluate_segmentation(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str = "outputs/seg/evaluation",
    device: str = "cuda",
    threshold: float = 0.5,
    batch_size: int = 16,
    num_workers: int = 4,
    save_visualizations: bool = True,
    max_visualizations: int = 20,
):
    """
    Evaluate segmentation model on validation/test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory with validation/test data
        output_dir: Directory to save results
        device: Device for inference
        threshold: Threshold for binarization
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        save_visualizations: Whether to save visualization overlays
        max_visualizations: Maximum number of visualizations to save
    """
    print("=" * 70)
    print("Segmentation Evaluation")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory
    if save_visualizations:
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    predictor = SegmentationPredictor(checkpoint_path, device, threshold)
    
    # Load dataset
    print(f"\nLoading dataset from {data_dir}...")
    dataset = BraTS2DSliceDataset(data_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"  Dataset size: {len(dataset)} slices")
    
    # Evaluate
    print("\nEvaluating...")
    all_metrics = []
    vis_count = 0
    
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Batches")):
        # Predict
        images_np = images.numpy()
        masks_np = masks.numpy()
        
        result = predictor.predict_batch(images_np, return_prob=True)
        pred_masks = result['masks']
        pred_probs = result['probs']
        
        # Calculate metrics for each sample in batch
        for i in range(len(images)):
            pred_mask = pred_masks[i]
            gt_mask = masks_np[i].squeeze()  # Remove channel dim
            
            metrics = calculate_metrics(pred_mask, gt_mask)
            metrics['batch_idx'] = batch_idx
            metrics['sample_idx'] = i
            all_metrics.append(metrics)
            
            # Save visualization
            if save_visualizations and vis_count < max_visualizations:
                image = images_np[i].squeeze()  # Remove channel dim
                overlay = create_overlay(image, pred_mask, gt_mask)
                
                # Create figure
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                # Original image
                axes[0].imshow(image, cmap='gray')
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                # Ground truth
                axes[1].imshow(image, cmap='gray')
                axes[1].imshow(gt_mask, cmap='Reds', alpha=0.5)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Prediction
                axes[2].imshow(image, cmap='gray')
                axes[2].imshow(pred_mask, cmap='Greens', alpha=0.5)
                axes[2].set_title(f'Prediction (Dice: {metrics["dice"]:.3f})')
                axes[2].axis('off')
                
                # Overlay (TP=Green, FP=Red, FN=Blue)
                axes[3].imshow(overlay)
                axes[3].set_title('Overlay (TP/FP/FN)')
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(vis_dir / f"sample_{vis_count:04d}.png", dpi=100, bbox_inches='tight')
                plt.close()
                
                vis_count += 1
    
    # Aggregate metrics
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    metrics_summary = {}
    for key in ['dice', 'iou', 'precision', 'recall', 'f1', 'specificity']:
        values = [m[key] for m in all_metrics]
        metrics_summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
        }
    
    print(f"\nMetrics (n={len(all_metrics)} slices):")
    print(f"  Dice:        {metrics_summary['dice']['mean']:.4f} ± {metrics_summary['dice']['std']:.4f}")
    print(f"  IoU:         {metrics_summary['iou']['mean']:.4f} ± {metrics_summary['iou']['std']:.4f}")
    print(f"  Precision:   {metrics_summary['precision']['mean']:.4f} ± {metrics_summary['precision']['std']:.4f}")
    print(f"  Recall:      {metrics_summary['recall']['mean']:.4f} ± {metrics_summary['recall']['std']:.4f}")
    print(f"  F1:          {metrics_summary['f1']['mean']:.4f} ± {metrics_summary['f1']['std']:.4f}")
    print(f"  Specificity: {metrics_summary['specificity']['mean']:.4f} ± {metrics_summary['specificity']['std']:.4f}")
    
    # Save detailed results
    results = {
        'checkpoint': str(checkpoint_path),
        'data_dir': str(data_dir),
        'num_samples': len(all_metrics),
        'threshold': threshold,
        'summary': metrics_summary,
        'per_sample': all_metrics,
    }
    
    results_file = output_path / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    if save_visualizations:
        print(f"Visualizations saved to: {vis_dir}")
        print(f"  Saved {vis_count} visualization(s)")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, key in enumerate(['dice', 'iou', 'precision', 'recall', 'f1', 'specificity']):
        values = [m[key] for m in all_metrics]
        axes[idx].hist(values, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].axvline(metrics_summary[key]['mean'], color='red', linestyle='--', 
                         label=f'Mean: {metrics_summary[key]["mean"]:.3f}')
        axes[idx].set_xlabel(key.capitalize())
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{key.capitalize()} Distribution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics distribution plot saved to: {output_path / 'metrics_distribution.png'}")
    
    print("\n" + "=" * 70)
    print("[OK] Evaluation Complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with validation/test data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/seg/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarization",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=20,
        help="Maximum number of visualizations to save",
    )
    
    args = parser.parse_args()
    
    evaluate_segmentation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_visualizations=args.max_visualizations,
    )


if __name__ == "__main__":
    main()
