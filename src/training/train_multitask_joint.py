"""
Phase 2.3: Joint Fine-Tuning Training Script

This script implements joint training of segmentation and classification tasks with:
1. Unfrozen encoder (all parameters trainable)
2. Alternating batches (BraTS for both tasks, Kaggle for classification only)
3. Combined loss function (L_seg + λ_cls * L_cls)
4. Differential learning rates (encoder vs decoder/cls_head)
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.multi_task_model import create_multi_task_model
from src.data.multi_source_dataset import MultiSourceDataset
from src.training.multi_task_losses import create_multi_task_loss


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


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


def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate classification accuracy."""
    pred_labels = torch.argmax(pred, dim=1)
    correct = (pred_labels == target).sum().item()
    total = target.size(0)
    return correct / total


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


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_dice = 0.0
    total_acc = 0.0
    n_seg_batches = 0
    n_cls_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        images = batch['image'].to(device)
        cls_targets = batch['cls'].to(device)
        has_masks = batch['has_mask'].to(device)
        
        # Get masks for samples that have them
        seg_targets = None
        if batch['mask'] is not None:
            seg_targets = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            # Forward pass - always compute both tasks
            outputs = model(images, do_seg=True, do_cls=True)
            seg_pred = outputs['seg']
            cls_pred = outputs['cls']
            
            # Compute loss
            loss, loss_dict = criterion(
                seg_pred=seg_pred,
                cls_pred=cls_pred,
                seg_target=seg_targets,
                cls_target=cls_targets,
                has_mask=has_masks,
            )
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Update metrics
        total_loss += loss_dict['total_loss']
        total_cls_loss += loss_dict['cls_loss']
        n_cls_batches += 1
        
        if loss_dict['n_seg_samples'] > 0:
            total_seg_loss += loss_dict['seg_loss']
            n_seg_batches += 1
            
            # Calculate Dice score for segmentation samples
            with torch.no_grad():
                seg_indices = has_masks.nonzero(as_tuple=True)[0]
                if len(seg_indices) > 0:
                    dice = calculate_dice_score(
                        seg_pred[seg_indices],
                        seg_targets
                    )
                    total_dice += dice
        
        # Calculate accuracy
        with torch.no_grad():
            acc = calculate_accuracy(cls_pred, cls_targets)
            total_acc += acc
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'seg': f"{loss_dict['seg_loss']:.4f}" if loss_dict['n_seg_samples'] > 0 else "N/A",
            'cls': f"{loss_dict['cls_loss']:.4f}",
            'acc': f"{acc:.4f}",
        })
    
    # Calculate epoch averages
    metrics = {
        'loss': total_loss / len(train_loader),
        'seg_loss': total_seg_loss / n_seg_batches if n_seg_batches > 0 else 0.0,
        'cls_loss': total_cls_loss / n_cls_batches,
        'dice': total_dice / n_seg_batches if n_seg_batches > 0 else 0.0,
        'acc': total_acc / len(train_loader),
    }
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_dice = 0.0
    total_acc = 0.0
    n_seg_batches = 0
    n_cls_batches = 0
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for batch in pbar:
        images = batch['image'].to(device)
        cls_targets = batch['cls'].to(device)
        has_masks = batch['has_mask'].to(device)
        
        seg_targets = None
        if batch['mask'] is not None:
            seg_targets = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(images, do_seg=True, do_cls=True)
        seg_pred = outputs['seg']
        cls_pred = outputs['cls']
        
        # Compute loss
        loss, loss_dict = criterion(
            seg_pred=seg_pred,
            cls_pred=cls_pred,
            seg_target=seg_targets,
            cls_target=cls_targets,
            has_mask=has_masks,
        )
        
        # Update metrics
        total_loss += loss_dict['total_loss']
        total_cls_loss += loss_dict['cls_loss']
        n_cls_batches += 1
        
        if loss_dict['n_seg_samples'] > 0:
            total_seg_loss += loss_dict['seg_loss']
            n_seg_batches += 1
            
            # Calculate Dice score
            seg_indices = has_masks.nonzero(as_tuple=True)[0]
            if len(seg_indices) > 0:
                dice = calculate_dice_score(
                    seg_pred[seg_indices],
                    seg_targets
                )
                total_dice += dice
        
        # Calculate accuracy
        acc = calculate_accuracy(cls_pred, cls_targets)
        total_acc += acc
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'dice': f"{total_dice/max(n_seg_batches, 1):.4f}",
            'acc': f"{total_acc/(pbar.n+1):.4f}",
        })
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'seg_loss': total_seg_loss / n_seg_batches if n_seg_batches > 0 else 0.0,
        'cls_loss': total_cls_loss / n_cls_batches,
        'dice': total_dice / n_seg_batches if n_seg_batches > 0 else 0.0,
        'acc': total_acc / len(val_loader),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 2.3: Joint Fine-Tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--init-from", type=str, required=True, help="Path to Phase 2.2 checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/multitask_joint", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=== Loading Datasets ===")
    # Load datasets
    train_dataset = MultiSourceDataset(
        brats_dir=config['data']['brats_train_dir'],
        kaggle_dir=config['data']['kaggle_train_dir'],
        transform=None,
    )
    
    val_dataset = MultiSourceDataset(
        brats_dir=config['data']['brats_val_dir'],
        kaggle_dir=config['data']['kaggle_val_dir'],
        transform=None,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn,
    )
    
    print("=== Creating Multi-Task Model ===")
    # Create model
    model = create_multi_task_model(
        in_channels=config['model']['in_channels'],
        cls_num_classes=config['model']['num_classes'],
        base_filters=config['model']['base_filters'],
        depth=config['model']['depth'],
    ).to(device)
    
    # Load Phase 2.2 checkpoint
    print(f"\n=== Loading Phase 2.2 Checkpoint: {args.init_from} ===")
    checkpoint = torch.load(args.init_from, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[OK] Loaded Phase 2.2 checkpoint\n")
    
    # Unfreeze all parameters
    print("=== Unfreezing All Parameters ===")
    for param in model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Create loss function
    criterion = create_multi_task_loss(config).to(device)
    
    # Create optimizer with differential learning rates
    print("=== Setting Up Differential Learning Rates ===")
    encoder_lr = config['training']['encoder_lr']
    decoder_cls_lr = config['training']['decoder_cls_lr']
    
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.seg_decoder.parameters(), 'lr': decoder_cls_lr},
        {'params': model.cls_head.parameters(), 'lr': decoder_cls_lr},
    ]
    
    print(f"Encoder LR: {encoder_lr}")
    print(f"Decoder + Cls Head LR: {decoder_cls_lr}\n")
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=config['training']['weight_decay'])
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6,
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Training loop
    start_epoch = 0
    best_val_metric = 0.0
    patience_counter = 0
    patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping'].get('min_delta', 0.0)
    
    if args.resume:
        print(f"=== Resuming from {args.resume} ===")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint.get('best_val_metric', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resuming from epoch {start_epoch}\n")
    
    print("=== Starting Training ===")
    print(f"Training for {config['training']['epochs']} epochs")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Mixed precision: {config['training']['mixed_precision']}")
    if config['training']['early_stopping']['enabled']:
        print(f"Early stopping: enabled (patience={patience}, min_delta={min_delta})")
    print()
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print("=" * 60)
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=config['training']['mixed_precision'],
            grad_clip=config['training']['grad_clip'],
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, "
              f"Acc: {train_metrics['acc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, "
              f"Acc: {val_metrics['acc']:.4f}")
        
        # Combined metric (average of Dice and Accuracy)
        val_metric = (val_metrics['dice'] + val_metrics['acc']) / 2.0
        print(f"Combined metric: {val_metric:.4f} (best: {best_val_metric:.4f})")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_metric': best_val_metric,
            'patience_counter': patience_counter,
            'config': config,
        }
        
        # Save last checkpoint
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'last_model.pth'))
        
        # Save best checkpoint and update early stopping
        if val_metric > best_val_metric + min_delta:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"[OK] Saved best model (metric: {best_val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{patience})")
        
        print(f"Saved checkpoint to {args.checkpoint_dir}/last_model.pth\n")
        
        # Early stopping check
        if config['training']['early_stopping']['enabled']:
            if patience_counter >= patience:
                print(f"⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"  No improvement for {patience} consecutive epochs")
                print(f"  Best validation metric: {best_val_metric:.4f}")
                break
    
    print("=== Training Complete ===")
    print(f"Best validation metric: {best_val_metric:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("\nNext: Evaluate on test set and compare with baseline models")


if __name__ == "__main__":
    main()
