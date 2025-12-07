"""
Comprehensive comparison script for all training phases.

Compares:
- Phase 2.1: Segmentation-only baseline
- Phase 2.2: Classification-only baseline  
- Phase 2.3: Joint multi-task model

Generates side-by-side metrics and visualizations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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
    roc_auc_score,
)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.multi_task_model import create_multi_task_model
from src.data.multi_source_dataset import MultiSourceDataset


def custom_collate_fn(batch):
    """Custom collate function to handle None values in masks."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['cls'] for item in batch], dtype=torch.long)
    sources = [item['source'] for item in batch]
    has_masks = torch.tensor([item['has_mask'] for item in batch], dtype=torch.bool)
    
    masks_list = []
    for item in batch:
        if item['mask'] is not None:
            masks_list.append(item['mask'])
    
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
    """Calculate IoU score for segmentation."""
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
def evaluate_phase(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    phase_name: str,
    do_seg: bool = True,
    do_cls: bool = True,
) -> Dict:
    """Evaluate a single phase model."""
    model.eval()
    
    # Segmentation metrics
    seg_dice_scores = []
    seg_iou_scores = []
    
    # Classification metrics
    all_cls_labels = []
    all_cls_preds = []
    all_cls_probs = []
    
    print(f"\nEvaluating {phase_name}...")
    for batch in tqdm(dataloader, desc=f"{phase_name}"):
        images = batch['image'].to(device)
        cls_labels = batch['cls'].to(device)
        has_masks = batch['has_mask'].to(device)
        
        # Forward pass
        outputs = model(images, do_seg=do_seg, do_cls=do_cls)
        
        # Classification metrics
        if do_cls and 'cls' in outputs:
            cls_pred = outputs['cls']
            cls_probs = F.softmax(cls_pred, dim=1)
            cls_pred_labels = torch.argmax(cls_pred, dim=1)
            
            all_cls_labels.extend(cls_labels.cpu().numpy())
            all_cls_preds.extend(cls_pred_labels.cpu().numpy())
            all_cls_probs.extend(cls_probs[:, 1].cpu().numpy())
        
        # Segmentation metrics
        if do_seg and 'seg' in outputs and batch['mask'] is not None and has_masks.any():
            seg_pred = outputs['seg']
            seg_targets = batch['mask'].to(device)
            mask_indices = has_masks.nonzero(as_tuple=True)[0]
            
            if len(mask_indices) > 0:
                seg_pred_masked = seg_pred[mask_indices]
                
                dice = calculate_dice_score(seg_pred_masked, seg_targets)
                iou = calculate_iou(seg_pred_masked, seg_targets)
                
                seg_dice_scores.append(dice)
                seg_iou_scores.append(iou)
    
    # Compile results
    results = {'phase': phase_name}
    
    # Segmentation metrics
    if do_seg and seg_dice_scores:
        results['segmentation'] = {
            'dice': np.mean(seg_dice_scores),
            'dice_std': np.std(seg_dice_scores),
            'iou': np.mean(seg_iou_scores),
            'iou_std': np.std(seg_iou_scores),
            'n_samples': len(seg_dice_scores),
        }
    else:
        results['segmentation'] = None
    
    # Classification metrics
    if do_cls and all_cls_labels:
        all_cls_labels = np.array(all_cls_labels)
        all_cls_preds = np.array(all_cls_preds)
        all_cls_probs = np.array(all_cls_probs)
        
        results['classification'] = {
            'accuracy': accuracy_score(all_cls_labels, all_cls_preds),
            'precision': precision_score(all_cls_labels, all_cls_preds, zero_division=0),
            'recall': recall_score(all_cls_labels, all_cls_preds, zero_division=0),
            'f1': f1_score(all_cls_labels, all_cls_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_cls_labels, all_cls_probs) if len(np.unique(all_cls_labels)) > 1 else 0.0,
            'n_samples': len(all_cls_labels),
        }
    else:
        results['classification'] = None
    
    return results


def print_comparison_table(results: List[Dict]):
    """Print beautiful comparison table."""
    print("\n" + "=" * 100)
    print("PHASE COMPARISON RESULTS")
    print("=" * 100)
    
    # Segmentation comparison
    print("\n📊 SEGMENTATION METRICS")
    print("-" * 100)
    print(f"{'Phase':<20} {'Dice Score':<20} {'IoU Score':<20} {'Samples':<15}")
    print("-" * 100)
    
    for result in results:
        phase = result['phase']
        seg = result['segmentation']
        if seg:
            print(f"{phase:<20} {seg['dice']:.4f} ± {seg['dice_std']:.4f}    "
                  f"{seg['iou']:.4f} ± {seg['iou_std']:.4f}    {seg['n_samples']:<15}")
        else:
            print(f"{phase:<20} {'N/A':<20} {'N/A':<20} {'N/A':<15}")
    
    # Classification comparison
    print("\n🎯 CLASSIFICATION METRICS")
    print("-" * 100)
    print(f"{'Phase':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 100)
    
    for result in results:
        phase = result['phase']
        cls = result['classification']
        if cls:
            print(f"{phase:<20} {cls['accuracy']:.4f}      {cls['precision']:.4f}      "
                  f"{cls['recall']:.4f}      {cls['f1']:.4f}      {cls['roc_auc']:.4f}")
        else:
            print(f"{phase:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    # Improvements
    print("\n📈 IMPROVEMENTS (Phase 2.3 vs Baselines)")
    print("-" * 100)
    
    phase_21 = next((r for r in results if '2.1' in r['phase']), None)
    phase_22 = next((r for r in results if '2.2' in r['phase']), None)
    phase_23 = next((r for r in results if '2.3' in r['phase']), None)
    
    if phase_21 and phase_23 and phase_21['segmentation'] and phase_23['segmentation']:
        dice_21 = phase_21['segmentation']['dice']
        dice_23 = phase_23['segmentation']['dice']
        improvement = ((dice_23 - dice_21) / dice_21) * 100
        print(f"Segmentation Dice: {dice_21:.4f} → {dice_23:.4f} ({improvement:+.1f}%)")
    
    if phase_22 and phase_23 and phase_22['classification'] and phase_23['classification']:
        acc_22 = phase_22['classification']['accuracy']
        acc_23 = phase_23['classification']['accuracy']
        improvement = ((acc_23 - acc_22) / acc_22) * 100
        print(f"Classification Acc: {acc_22:.4f} → {acc_23:.4f} ({improvement:+.1f}%)")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Compare all training phases")
    parser.add_argument("--phase-21-checkpoint", type=str, 
                       default="checkpoints/multitask_seg_warmup/best_model.pth",
                       help="Phase 2.1 checkpoint")
    parser.add_argument("--phase-22-checkpoint", type=str,
                       default="checkpoints/multitask_cls_head/best_model.pth",
                       help="Phase 2.2 checkpoint")
    parser.add_argument("--phase-23-checkpoint", type=str,
                       default="checkpoints/multitask_joint/best_model.pth",
                       help="Phase 2.3 checkpoint")
    parser.add_argument("--test-brats", type=str, default="data/processed/brats2d/test")
    parser.add_argument("--test-kaggle", type=str, default="data/processed/kaggle/test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default="results/phase_comparison.json")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\nLoading test datasets...")
    test_dataset = MultiSourceDataset(
        brats_dir=args.test_brats,
        kaggle_dir=args.test_kaggle,
        transform=None,
    )
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )
    
    results = []
    
    # Evaluate Phase 2.1 (Segmentation only)
    if Path(args.phase_21_checkpoint).exists():
        print(f"\n{'='*100}")
        print("PHASE 2.1: Segmentation Warm-up")
        print(f"{'='*100}")
        
        checkpoint = torch.load(args.phase_21_checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        model = create_multi_task_model(
            in_channels=model_config.get('in_channels', 1),
            cls_num_classes=model_config.get('num_classes', 2),
            base_filters=model_config.get('base_filters', 32),
            depth=model_config.get('depth', 3),
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        result = evaluate_phase(model, test_loader, device, "Phase 2.1 (Seg Only)", 
                               do_seg=True, do_cls=False)
        results.append(result)
    
    # Evaluate Phase 2.2 (Classification only)
    if Path(args.phase_22_checkpoint).exists():
        print(f"\n{'='*100}")
        print("PHASE 2.2: Classification Head")
        print(f"{'='*100}")
        
        checkpoint = torch.load(args.phase_22_checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        model = create_multi_task_model(
            in_channels=model_config.get('in_channels', 1),
            cls_num_classes=model_config.get('num_classes', 2),
            base_filters=model_config.get('base_filters', 32),
            depth=model_config.get('depth', 3),
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        result = evaluate_phase(model, test_loader, device, "Phase 2.2 (Cls Only)",
                               do_seg=False, do_cls=True)
        results.append(result)
    
    # Evaluate Phase 2.3 (Joint multi-task)
    if Path(args.phase_23_checkpoint).exists():
        print(f"\n{'='*100}")
        print("PHASE 2.3: Joint Fine-Tuning")
        print(f"{'='*100}")
        
        checkpoint = torch.load(args.phase_23_checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        model = create_multi_task_model(
            in_channels=model_config.get('in_channels', 1),
            cls_num_classes=model_config.get('num_classes', 2),
            base_filters=model_config.get('base_filters', 32),
            depth=model_config.get('depth', 3),
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        result = evaluate_phase(model, test_loader, device, "Phase 2.3 (Multi-Task)",
                               do_seg=True, do_cls=True)
        results.append(result)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
