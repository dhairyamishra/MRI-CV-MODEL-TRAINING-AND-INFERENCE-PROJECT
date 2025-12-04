"""
Training script for 2D brain tumor segmentation with U-Net.

Supports:
- Multiple loss functions (Dice, BCE, combined, Tversky, Focal)
- Mixed precision training (AMP)
- Early stopping
- Learning rate scheduling
- W&B logging
- Checkpoint management
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
from src.models.unet2d import create_unet
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
    
    pbar = tqdm(train_loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            dice = calculate_dice_score(probs, masks)
            iou = calculate_iou(probs, masks)
        
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
        })
    
    num_batches = len(train_loader)
    return {
        'loss': running_loss / num_batches,
        'dice': running_dice / num_batches,
        'iou': running_iou / num_batches,
    }


@torch.no_grad()
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
    
    pbar = tqdm(val_loader, desc="Validation")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Calculate metrics
        probs = torch.sigmoid(outputs)
        dice = calculate_dice_score(probs, masks)
        iou = calculate_iou(probs, masks)
        
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
        })
    
    num_batches = len(val_loader)
    return {
        'loss': running_loss / num_batches,
        'dice': running_dice / num_batches,
        'iou': running_iou / num_batches,
    }


def train(config_path: str):
    """Main training function."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("SliceWise - 2D Segmentation Training")
    print("=" * 70)
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create directories
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = BraTS2DSliceDataset(config['paths']['train_dir'])
    val_dataset = BraTS2DSliceDataset(config['paths']['val_dir'])
    
    print(f"  Train: {len(train_dataset)} slices")
    print(f"  Val:   {len(val_dataset)} slices")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
    )
    
    # Create model
    print("\nCreating model...")
    model = create_unet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_filters=config['model']['base_filters'],
        depth=config['model']['depth'],
        bilinear=config['model']['use_bilinear'],
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Create loss function
    loss_config = config['loss']
    loss_name = loss_config['name']
    
    # Build kwargs based on loss type
    loss_kwargs = {}
    if loss_name == 'dice':
        loss_kwargs['smooth'] = loss_config.get('smooth', 1.0)
    elif loss_name == 'bce':
        pass  # No additional params
    elif loss_name == 'dice_bce':
        loss_kwargs['dice_weight'] = loss_config.get('dice_weight', 0.5)
        loss_kwargs['bce_weight'] = loss_config.get('bce_weight', 0.5)
        loss_kwargs['smooth'] = loss_config.get('smooth', 1.0)
    elif loss_name == 'tversky':
        loss_kwargs['alpha'] = loss_config.get('alpha', 0.5)
        loss_kwargs['beta'] = loss_config.get('beta', 0.5)
        loss_kwargs['smooth'] = loss_config.get('smooth', 1.0)
    elif loss_name == 'focal':
        loss_kwargs['alpha'] = loss_config.get('focal_alpha', 0.25)
        loss_kwargs['gamma'] = loss_config.get('focal_gamma', 2.0)
    
    criterion = get_loss_function(loss_name, **loss_kwargs)
    print(f"  Loss: {loss_name}")
    
    # Create optimizer
    opt_config = config['optimizer']
    if opt_config['name'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas'],
            eps=opt_config['eps'],
        )
    elif opt_config['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas'],
            eps=opt_config['eps'],
        )
    elif opt_config['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            momentum=opt_config['momentum'],
            nesterov=opt_config['nesterov'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['name']}")
    
    print(f"  Optimizer: {opt_config['name']} (lr={opt_config['lr']})")
    
    # Create scheduler
    sched_config = config['scheduler']
    scheduler = None
    if sched_config['name'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max'],
            eta_min=sched_config['eta_min'],
        )
    elif sched_config['name'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma'],
        )
    elif sched_config['name'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_config['mode'],
            factor=sched_config['factor'],
            patience=sched_config['patience'],
            threshold=sched_config['threshold'],
        )
    
    if scheduler:
        print(f"  Scheduler: {sched_config['name']}")
    
    # Mixed precision
    scaler = GradScaler() if config['training']['use_amp'] else None
    if scaler:
        print("  Mixed precision: Enabled")
    
    # Initialize W&B
    wandb_run = None
    if config['logging']['use_wandb']:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging']['wandb_entity'],
                name=config['experiment']['name'],
                config=config,
                tags=config['experiment']['tags'],
            )
            print("  W&B logging: Enabled")
        except ImportError:
            print("  W&B logging: Disabled (wandb not installed)")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, config['training']['grad_clip']
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # Log to W&B
        if wandb_run:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/dice': train_metrics['dice'],
                'train/iou': train_metrics['iou'],
                'val/loss': val_metrics['loss'],
                'val/dice': val_metrics['dice'],
                'val/iou': val_metrics['iou'],
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['dice'])
            else:
                scheduler.step()
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            
            if config['training']['save_best']:
                checkpoint_path = checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': best_val_dice,
                    'config': config,
                }, checkpoint_path)
                print(f"  Saved best model (Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if patience_counter >= config['training']['early_stopping']['patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    # Save final model
    if config['training']['save_last']:
        checkpoint_path = checkpoint_dir / 'last_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_metrics['dice'],
            'config': config,
        }, checkpoint_path)
        print(f"\nSaved final model")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    if wandb_run:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for brain tumor segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/seg2d_baseline.yaml",
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
