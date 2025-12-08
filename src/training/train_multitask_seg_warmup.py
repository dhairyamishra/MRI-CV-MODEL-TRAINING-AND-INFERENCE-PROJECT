"""
Phase 2.1: Segmentation Warm-Up Training for Multi-Task Model.

Trains encoder + decoder on BraTS segmentation task to initialize
shared encoder with good features before adding classification head.

This is Stage 1 of the 3-stage training strategy:
1. Seg warm-up (this script) - BraTS only, seg task only
2. Cls head training - BraTS + Kaggle, frozen encoder
3. Joint fine-tuning - Both datasets, both tasks, unfrozen encoder
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.brats2d_dataset import BraTS2DSliceDataset
from src.models.multi_task_model import create_multi_task_model
from src.training.losses import get_loss_function


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Predictions (B, 1, H, W) - probabilities
        target: Ground truth (B, 1, H, W) - binary
        threshold: Threshold for binarizing predictions
    
    Returns:
        Dice score
    """
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    
    if union == 0:
        return 1.0  # Both empty
    
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice.item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred: Predictions (B, 1, H, W) - probabilities
        target: Ground truth (B, 1, H, W) - binary
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score
    """
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0  # Both empty
    
    iou = intersection / (union + 1e-8)
    return iou.item()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # BraTS2DSliceDataset returns (image, mask) tuples
        images, masks = batch
        images = images.to(device)  # (B, 1, H, W)
        masks = masks.to(device)    # (B, 1, H, W)
        
        # Note: masks are already (B, 1, H, W), no need to unsqueeze
        
        optimizer.zero_grad()
        
        # Forward pass (segmentation only)
        if scaler is not None:
            with autocast():
                output = model(images, do_seg=True, do_cls=False)
                logits = output['seg']  # (B, 1, H, W)
                loss = criterion(logits, masks)
        else:
            output = model(images, do_seg=True, do_cls=False)
            logits = output['seg']
            loss = criterion(logits, masks)
        
        # Backward pass
        if scaler is not None:
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
        
        # Calculate metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            dice = calculate_dice_score(probs, masks)
            iou = calculate_iou(probs, masks)
        
        # Update running metrics
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}',
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_batches
    epoch_dice = running_dice / num_batches
    epoch_iou = running_iou / num_batches
    
    return {
        'loss': epoch_loss,
        'dice': epoch_dice,
        'iou': epoch_iou,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            # BraTS2DSliceDataset returns (image, mask) tuples
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            
            # Note: masks are already (B, 1, H, W), no need to unsqueeze
            
            # Forward pass (segmentation only)
            output = model(images, do_seg=True, do_cls=False)
            logits = output['seg']
            loss = criterion(logits, masks)
            
            # Calculate metrics
            probs = torch.sigmoid(logits)
            dice = calculate_dice_score(probs, masks)
            iou = calculate_iou(probs, masks)
            
            # Update running metrics
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}',
            })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_batches
    epoch_dice = running_dice / num_batches
    epoch_iou = running_iou / num_batches
    
    return {
        'loss': epoch_loss,
        'dice': epoch_dice,
        'iou': epoch_iou,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path,
    config: dict,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2.1: Segmentation Warm-Up Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/multitask_seg_warmup",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset = BraTS2DSliceDataset(
        data_dir=config['paths']['train_dir'],
        transform=None,  # TODO: Add augmentation transforms
    )
    val_dataset = BraTS2DSliceDataset(
        data_dir=config['paths']['val_dir'],
        transform=None,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Create model
    print("\n=== Creating Multi-Task Model ===")
    model = create_multi_task_model(
        in_channels=config['model']['in_channels'],
        seg_out_channels=config['model']['out_channels'],
        cls_num_classes=2,  # Binary classification
        base_filters=config['model']['base_filters'],
        depth=config['model']['depth'],
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    loss_config = config['loss']
    loss_kwargs = {
        'dice_weight': loss_config.get('dice_weight', 0.5),
        'bce_weight': loss_config.get('bce_weight', 0.5),
        'smooth': loss_config.get('smooth', 1.0),
    }
    criterion = get_loss_function(
        loss_name=loss_config['name'],
        **loss_kwargs
    )
    
    # Optimizer
    if config['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
        )
    elif config['optimizer']['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
        )
    elif config['optimizer']['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['optimizer']['weight_decay'],
            nesterov=config['optimizer'].get('nesterov', False),
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']['name']}")
    
    # Scheduler
    scheduler = None
    if config['scheduler']['name'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler']['T_max'],
            eta_min=config['scheduler']['eta_min'],
        )
    elif config['scheduler']['name'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['scheduler']['step_size'],
            gamma=config['scheduler']['gamma'],
        )
    elif config['scheduler']['name'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['scheduler']['mode'],
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience'],
            threshold=config['scheduler']['threshold'],
            min_lr=config['scheduler']['min_lr'],
        )
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training']['use_amp'] else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        print(f"\n=== Resuming from {args.resume} ===")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['metrics'].get('dice', 0.0)
        print(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")
    
    # Training loop
    print("\n=== Starting Training ===")
    print(f"Training for {config['training']['epochs']} epochs")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['optimizer']['lr']}")
    print(f"Mixed precision: {config['training']['use_amp']}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    patience_counter = 0
    patience = config['training']['early_stopping']['patience']
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
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
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['dice'])
            else:
                scheduler.step()
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_path=checkpoint_dir / "best_model.pth",
                config=config,
            )
            print(f"[OK] New best model! Dice: {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save last checkpoint
        if config['training']['save_last']:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_path=checkpoint_dir / "last_model.pth",
                config=config,
            )
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered after {patience} epochs without improvement")
                break
    
    print("\n=== Training Complete ===")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\nNext: Use this checkpoint to initialize encoder for Phase 2.2 (Classification Head Training)")


if __name__ == "__main__":
    main()
