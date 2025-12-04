"""
Training script for brain tumor classification.

This script implements the complete training loop with:
- W&B logging
- Early stopping
- Checkpointing
- Mixed precision training
- Learning rate scheduling
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import create_classifier
from src.data.kaggle_mri_dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be limited.")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ClassificationTrainer:
    """
    Trainer class for brain tumor classification.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        self.set_seed(self.config['seed'])
        
        # Setup device
        self.device = self._setup_device()
        print(f"Using device: {self.device}")
        
        # Create output directories
        self._create_directories()
        
        # Initialize W&B
        self.use_wandb = self.config['logging']['use_wandb'] and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()
        
        # Create model
        self.model = self._create_model()
        
        # Watch model with wandb if enabled
        if self.use_wandb:
            wandb.watch(self.model, log='all', log_freq=100)
        
        # Setup loss function
        self.criterion = self._setup_loss()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision training
        self.use_amp = self.config['training']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        if self.config['hardware']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif self.config['hardware']['benchmark']:
            torch.backends.cudnn.benchmark = True
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        device_config = self.config['hardware']['device']
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                device = torch.device('cpu')
            else:
                gpu_id = self.config['hardware']['gpu_id']
                device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
        
        return device
    
    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.config['checkpoint']['save_dir'],
            self.config['paths']['output_dir'],
            self.config['paths']['log_dir'],
            self.config['paths']['results_dir'],
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config['logging']['wandb_project'],
            entity=self.config['logging'].get('wandb_entity'),
            config=self.config,
            name=f"cls_{self.config['model']['name']}_{self.config['seed']}"
        )
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        # Get transforms
        if self.config['data']['augmentation']['use_augmentation']:
            aug_strength = self.config['data']['augmentation']['augmentation_strength']
            if aug_strength == 'light':
                from src.data.transforms import get_light_train_transforms
                train_transform = get_light_train_transforms()
            elif aug_strength == 'strong':
                from src.data.transforms import get_strong_train_transforms
                train_transform = get_strong_train_transforms()
            else:
                train_transform = get_train_transforms()
        else:
            train_transform = None
        
        val_transform = get_val_transforms()
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            train_transform=train_transform,
            val_transform=val_transform
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def _create_model(self) -> nn.Module:
        """Create and initialize model."""
        model = create_classifier(
            model_name=self.config['model']['name'],
            pretrained=self.config['model']['pretrained'],
            num_classes=self.config['model']['num_classes'],
            dropout=self.config['model']['dropout'],
            freeze_backbone=self.config['model']['freeze_backbone']
        )
        
        model = model.to(self.device)
        
        print(f"Model: {self.config['model']['name']}")
        print(f"Total parameters: {model.get_num_total_params():,}")
        print(f"Trainable parameters: {model.get_num_trainable_params():,}")
        
        return model
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        loss_config = self.config['training']['loss']
        
        if loss_config['name'] == 'focal':
            criterion = FocalLoss(
                alpha=loss_config['alpha'],
                gamma=loss_config['gamma']
            )
        else:
            # Cross entropy with optional class weights
            if self.config['training']['use_class_weights']:
                # Calculate class weights from training data
                class_counts = np.bincount([label for _, label in self.train_loader.dataset])
                class_weights = 1.0 / class_counts
                class_weights = class_weights / class_weights.sum()
                class_weights = torch.FloatTensor(class_weights).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        
        return criterion
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        opt_config = self.config['training']['optimizer']
        
        if opt_config['name'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        elif opt_config['name'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        elif opt_config['name'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=0.9,
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[object]:
        """Setup learning rate scheduler."""
        if not self.config['training']['scheduler']['use_scheduler']:
            return None
        
        sched_config = self.config['training']['scheduler']
        
        if sched_config['name'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config['eta_min']
            )
        elif sched_config['name'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_config['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=sched_config['patience'],
                factor=sched_config['factor']
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_config['name']}")
        
        return scheduler
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.optimizer.step()
            
            running_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to W&B
            if self.use_wandb and batch_idx % self.config['logging']['log_frequency'] == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of tumor class
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_auc = roc_auc_score(all_labels, all_probs)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        metrics = {
            'loss': val_loss,
            'accuracy': val_acc,
            'roc_auc': val_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return val_loss, metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (metric: {self.best_metric:.4f})")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        keep_last_n = self.config['checkpoint']['keep_last_n']
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50 + "\n")
        
        num_epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping_patience']
        val_metric = self.config['validation']['metric']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            
            # Validate
            if (epoch + 1) % self.config['validation']['val_frequency'] == 0:
                val_loss, val_metrics = self.validate()
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_metrics)
                
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Acc: {val_metrics['accuracy']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")
                print(f"Val F1: {val_metrics['f1']:.4f} | Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
                
                # Log to W&B
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train/loss': train_loss,
                        'train/accuracy': train_acc,
                        'val/loss': val_loss,
                        **{f'val/{k}': v for k, v in val_metrics.items()}
                    })
                
                # Check if best model
                current_metric = val_metrics[val_metric]
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                if self.config['validation']['save_best_only']:
                    if is_best:
                        self.save_checkpoint(is_best=True)
                else:
                    self.save_checkpoint(is_best=is_best)
                
                # Early stopping
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[val_metric])
                else:
                    self.scheduler.step()
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best {val_metric}: {self.best_metric:.4f}")
        print("="*50 + "\n")
        
        if self.use_wandb:
            wandb.finish()


def train_classifier(config_path: str):
    """
    Main function to train the classifier.
    
    Args:
        config_path: Path to configuration YAML file
    
    Example:
        >>> train_classifier('configs/config_cls.yaml')
    """
    trainer = ClassificationTrainer(config_path)
    trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train brain tumor classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_cls.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    train_classifier(args.config)
