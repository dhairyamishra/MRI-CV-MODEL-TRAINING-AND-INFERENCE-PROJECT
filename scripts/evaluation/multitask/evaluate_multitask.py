"""
Evaluation script for multi-task model (Phase 2.3).

Evaluates both segmentation and classification performance on test set.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.multi_task_model import create_multi_task_model
from src.data.multi_source_dataset import MultiSourceDataset


def custom_collate_fn(batch):
    """Custom collate function to handle None values in masks."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['cls'] for item in batch], dtype=torch.long)
    sources = [item['source'] for item in batch]
    has_masks = torch.tensor([item['has_mask'] for item in batch], dtype=torch.bool)
    
    # For masks, collect non-None values
    masks_list = []
    for item in batch:
        if item['mask'] is not None:
            masks_list.append(item['mask'])
    
    # Stack masks if any exist
    if masks_list:
        masks = torch.stack(masks_list)
    else:
        masks = None
    
    return {
        'image': images,
        'mask': masks,
        'cls': labels,
        'source': sources,
        'has_mask': has_masks,
    }


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Dice score for segmentation."""
    pred = (torch.sigmoid(pred) > threshold).float()
    
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice.item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate IoU (Jaccard) score for segmentation."""
    pred = (torch.sigmoid(pred) > threshold).float()
    
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate multi-task model on test set.
    
    Returns:
        Dictionary with segmentation and classification metrics
    """
    model.eval()
    
    # Segmentation metrics
    seg_dice_scores = []
    seg_iou_scores = []
    
    # Classification metrics
    all_cls_labels = []
    all_cls_preds = []
    all_cls_probs = []
    
    print("\nEvaluating model...")
    for batch in tqdm(dataloader, desc="Evaluation"):
        images = batch['image'].to(device)
        cls_labels = batch['cls'].to(device)
        has_masks = batch['has_mask'].to(device)
        
        # Forward pass
        outputs = model(images, do_seg=True, do_cls=True)
        seg_pred = outputs['seg']
        cls_pred = outputs['cls']
        
        # Classification metrics (all samples)
        cls_probs = F.softmax(cls_pred, dim=1)
        cls_pred_labels = torch.argmax(cls_pred, dim=1)
        
        all_cls_labels.extend(cls_labels.cpu().numpy())
        all_cls_preds.extend(cls_pred_labels.cpu().numpy())
        all_cls_probs.extend(cls_probs[:, 1].cpu().numpy())  # Probability of tumor class
        
        # Segmentation metrics (only samples with masks)
        if batch['mask'] is not None and has_masks.any():
            seg_targets = batch['mask'].to(device)
            mask_indices = has_masks.nonzero(as_tuple=True)[0]
            
            if len(mask_indices) > 0:
                seg_pred_masked = seg_pred[mask_indices]
                
                # Calculate Dice and IoU
                dice = calculate_dice_score(seg_pred_masked, seg_targets)
                iou = calculate_iou(seg_pred_masked, seg_targets)
                
                seg_dice_scores.append(dice)
                seg_iou_scores.append(iou)
    
    # Compute segmentation metrics
    seg_metrics = {
        'dice': np.mean(seg_dice_scores) if seg_dice_scores else 0.0,
        'dice_std': np.std(seg_dice_scores) if seg_dice_scores else 0.0,
        'iou': np.mean(seg_iou_scores) if seg_iou_scores else 0.0,
        'iou_std': np.std(seg_iou_scores) if seg_iou_scores else 0.0,
        'n_samples': len(seg_dice_scores),
    }
    
    # Compute classification metrics
    all_cls_labels = np.array(all_cls_labels)
    all_cls_preds = np.array(all_cls_preds)
    all_cls_probs = np.array(all_cls_probs)
    
    cls_metrics = {
        'accuracy': accuracy_score(all_cls_labels, all_cls_preds),
        'precision': precision_score(all_cls_labels, all_cls_preds, zero_division=0),
        'recall': recall_score(all_cls_labels, all_cls_preds, zero_division=0),
        'f1': f1_score(all_cls_labels, all_cls_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_cls_labels, all_cls_probs) if len(np.unique(all_cls_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(all_cls_labels, all_cls_preds).tolist(),
        'n_samples': len(all_cls_labels),
    }
    
    return {
        'segmentation': seg_metrics,
        'classification': cls_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-brats", type=str, default="data/processed/brats2d/test", help="BraTS test directory")
    parser.add_argument("--test-kaggle", type=str, default="data/processed/kaggle/test", help="Kaggle test directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default="results/multitask_evaluation.json", help="Output JSON file")
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Create model
    print("\nCreating model...")
    model = create_multi_task_model(
        in_channels=model_config.get('in_channels', 1),
        cls_num_classes=model_config.get('num_classes', 2),
        base_filters=model_config.get('base_filters', 32),
        depth=model_config.get('depth', 3),
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[OK] Model loaded successfully")
    
    # Print model info
    params = model.get_num_params()
    print(f"\nModel parameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Seg Decoder: {params['seg_decoder']:,}")
    print(f"  Cls Head: {params['cls_head']:,}")
    print(f"  Total: {params['total']:,}")
    
    # Load test dataset
    print(f"\nLoading test datasets...")
    test_dataset = MultiSourceDataset(
        brats_dir=args.test_brats,
        kaggle_dir=args.test_kaggle,
        transform=None,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print("\n📊 SEGMENTATION METRICS (BraTS samples)")
    print("-" * 80)
    seg = results['segmentation']
    print(f"  Dice Score:  {seg['dice']:.4f} ± {seg['dice_std']:.4f}")
    print(f"  IoU Score:   {seg['iou']:.4f} ± {seg['iou_std']:.4f}")
    print(f"  Samples:     {seg['n_samples']}")
    
    print("\n🎯 CLASSIFICATION METRICS (All samples)")
    print("-" * 80)
    cls = results['classification']
    print(f"  Accuracy:    {cls['accuracy']:.4f}")
    print(f"  Precision:   {cls['precision']:.4f}")
    print(f"  Recall:      {cls['recall']:.4f}")
    print(f"  F1 Score:    {cls['f1']:.4f}")
    print(f"  ROC-AUC:     {cls['roc_auc']:.4f}")
    print(f"  Samples:     {cls['n_samples']}")
    
    print("\n📈 CONFUSION MATRIX")
    print("-" * 80)
    cm = np.array(cls['confusion_matrix'])
    print(f"                Predicted")
    print(f"              No Tumor  Tumor")
    print(f"Actual No Tumor  {cm[0, 0]:4d}    {cm[0, 1]:4d}")
    print(f"       Tumor     {cm[1, 0]:4d}    {cm[1, 1]:4d}")
    
    # Combined metric
    combined_metric = (seg['dice'] + cls['accuracy']) / 2.0
    print(f"\n🎯 COMBINED METRIC: {combined_metric:.4f}")
    print("   (Average of Dice and Accuracy)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results['metadata'] = {
        'checkpoint': args.checkpoint,
        'test_brats': args.test_brats,
        'test_kaggle': args.test_kaggle,
        'device': str(device),
        'model_params': params,
        'combined_metric': combined_metric,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
