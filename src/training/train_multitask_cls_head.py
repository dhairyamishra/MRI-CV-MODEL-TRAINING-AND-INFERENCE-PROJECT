"""
Phase 2.2: Classification Head Training for Multi-Task Model.

Trains classification head with FROZEN encoder on BraTS + Kaggle data.
Encoder is initialized from Phase 2.1 segmentation warm-up checkpoint.

This is Stage 2 of the 3-stage training strategy:
1. Seg warm-up - BraTS only, seg task only (Phase 2.1)
2. Cls head training (this script) - BraTS + Kaggle, frozen encoder
3. Joint fine-tuning - Both datasets, both tasks, unfrozen encoder (Phase 2.3)
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

from src.data.multi_source_dataset import MultiSourceDataset
from src.models.multi_task_model import create_multi_task_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        pred: Predictions (B, 2) - logits
        target: Ground truth (B,) - class indices
    
    Returns:
        Accuracy
    """
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == target).sum().item()
    total = target.size(0)
    return correct / total


def custom_collate_fn(batch):
    """
    Custom collate function to handle None values in masks.
    
    For classification-only training, Kaggle samples have mask=None.
    This function filters out None masks and creates proper batches.
    """
    # Separate components
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['cls'] for item in batch], dtype=torch.long)
    sources = [item['source'] for item in batch]
    has_masks = [item['has_mask'] for item in batch]
    
    # For masks, only stack non-None values (not used in classification training)
    masks = [item['mask'] for item in batch if item['mask'] is not None]
    if masks:
        masks = torch.stack(masks)
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
    scaler: Optional[GradScaler],
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)  # (B, 1, H, W)
        labels = batch['cls'].to(device)    # (B,)
        
        optimizer.zero_grad()
        
        # Forward pass (classification only)
        if scaler is not None:
            with autocast():
                output = model(images, do_seg=False, do_cls=True)
                logits = output['cls']  # (B, 2)
                loss = criterion(logits, labels)
        else:
            output = model(images, do_seg=False, do_cls=True)
            logits = output['cls']
            loss = criterion(logits, labels)
        
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
            acc = calculate_accuracy(logits, labels)
        
        # Update running metrics
        running_loss += loss.item()
        running_acc += acc
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}',
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_batches
    epoch_acc = running_acc / num_batches
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
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
    running_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['cls'].to(device)
            
            # Forward pass (classification only)
            output = model(images, do_seg=False, do_cls=True)
            logits = output['cls']
            loss = criterion(logits, labels)
            
            # Calculate metrics
            acc = calculate_accuracy(logits, labels)
            
            # Update running metrics
            running_loss += loss.item()
            running_acc += acc
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}',
            })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_batches
    epoch_acc = running_acc / num_batches
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
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
    parser = argparse.ArgumentParser(description="Phase 2.2: Classification Head Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--encoder-init", type=str, required=True,
                        help="Path to Phase 2.1 checkpoint for encoder initialization")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/multitask_cls_head",
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
    train_dataset = MultiSourceDataset(
        brats_dir=config['paths'].get('brats_train_dir'),
        kaggle_dir=config['paths'].get('kaggle_train_dir'),
        transform=None,  # TODO: Add augmentation support
    )
    val_dataset = MultiSourceDataset(
        brats_dir=config['paths'].get('brats_val_dir'),
        kaggle_dir=config['paths'].get('kaggle_val_dir'),
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
        collate_fn=custom_collate_fn,  # Use custom collate function
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn,  # Use custom collate function
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
    
    # Load encoder weights from Phase 2.1
    print(f"\n=== Loading Encoder from {args.encoder_init} ===")
    checkpoint = torch.load(args.encoder_init, map_location=device)
    
    # Load only encoder and decoder weights (not cls head)
    model_state = checkpoint['model_state_dict']
    encoder_state = {k: v for k, v in model_state.items() if k.startswith('encoder.')}
    decoder_state = {k: v for k, v in model_state.items() if k.startswith('decoder.')}
    
    # Load encoder and decoder
    model.load_state_dict(encoder_state, strict=False)
    model.load_state_dict(decoder_state, strict=False)
    print(f"✓ Loaded encoder and decoder from Phase 2.1 checkpoint")
    
    # Freeze encoder
    print("\n=== Freezing Encoder ===")
    model.freeze_encoder()
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Loss function (weighted cross-entropy for class imbalance)
    if 'class_weights' in config['loss']:
        class_weights = torch.tensor(config['loss']['class_weights']).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted CrossEntropyLoss with weights: {config['loss']['class_weights']}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")
    
    # Optimizer (only for classification head)
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    
    if config['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params_list,
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
        )
    elif config['optimizer']['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
        )
    elif config['optimizer']['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            trainable_params_list,
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
    best_acc = 0.0
    if args.resume:
        print(f"\n=== Resuming from {args.resume} ===")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['metrics'].get('acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")
    
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
                scheduler.step(val_metrics['acc'])
            else:
                scheduler.step()
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")
        
        # Save best model
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_path=checkpoint_dir / "best_model.pth",
                config=config,
            )
            print(f"✓ New best model! Acc: {best_acc:.4f}")
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
    print(f"Best validation Accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\nNext: Use this checkpoint for Phase 2.3 (Joint Fine-Tuning)")


if __name__ == "__main__":
    main()
