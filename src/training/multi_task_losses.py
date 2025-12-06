"""
Multi-task loss functions for joint segmentation and classification training.

This module implements combined loss functions that handle:
1. BraTS samples: Both segmentation and classification losses
2. Kaggle samples: Classification loss only (no masks available)

The combined loss is weighted to balance the two tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits of shape (B, 1, H, W)
            target: Ground truth masks of shape (B, H, W) or (B, 1, H, W)
        
        Returns:
            Dice loss (scalar)
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Ensure target has same shape as pred
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), -1)  # (B, H*W)
        target = target.view(target.size(0), -1)  # (B, H*W)
        
        # Calculate Dice coefficient
        intersection = (pred * target).sum(dim=1)  # (B,)
        union = pred.sum(dim=1) + target.sum(dim=1)  # (B,)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return mean Dice loss
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Dice + BCE loss for segmentation.
    
    This is commonly used in medical image segmentation as it combines:
    - Dice loss: Good for class imbalance
    - BCE loss: Pixel-wise classification
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        """
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits of shape (B, 1, H, W)
            target: Ground truth masks of shape (B, H, W) or (B, 1, H, W)
        
        Returns:
            Combined loss (scalar)
        """
        # Ensure target has same shape as pred for BCE
        if target.dim() == 3:
            target_bce = target.unsqueeze(1).float()  # (B, H, W) -> (B, 1, H, W)
        else:
            target_bce = target.float()
        
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target_bce)
        
        return self.dice_weight * dice + self.bce_weight * bce


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for joint segmentation and classification training.
    
    Handles two types of batches:
    1. BraTS: Has both masks and labels -> L_seg + λ_cls * L_cls
    2. Kaggle: Has labels only -> λ_cls * L_cls
    """
    
    def __init__(
        self,
        seg_loss_type: str = 'dice_bce',
        cls_loss_type: str = 'ce',
        lambda_cls: float = 1.0,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            seg_loss_type: Type of segmentation loss ('dice', 'bce', 'dice_bce')
            cls_loss_type: Type of classification loss ('ce', 'focal')
            lambda_cls: Weight for classification loss
            dice_weight: Weight for Dice loss (if using dice_bce)
            bce_weight: Weight for BCE loss (if using dice_bce)
            class_weights: Optional class weights for classification loss
        """
        super().__init__()
        self.lambda_cls = lambda_cls
        
        # Segmentation loss
        if seg_loss_type == 'dice':
            self.seg_loss_fn = DiceLoss()
        elif seg_loss_type == 'bce':
            self.seg_loss_fn = nn.BCEWithLogitsLoss()
        elif seg_loss_type == 'dice_bce':
            self.seg_loss_fn = CombinedSegmentationLoss(dice_weight, bce_weight)
        else:
            raise ValueError(f"Unknown seg_loss_type: {seg_loss_type}")
        
        # Classification loss
        if cls_loss_type == 'ce':
            self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif cls_loss_type == 'focal':
            # Focal loss not implemented yet, fall back to CE
            self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unknown cls_loss_type: {cls_loss_type}")
    
    def forward(
        self,
        seg_pred: Optional[torch.Tensor],
        cls_pred: torch.Tensor,
        seg_target: Optional[torch.Tensor],
        cls_target: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.
        
        Args:
            seg_pred: Segmentation predictions (B, 1, H, W) or None
            cls_pred: Classification predictions (B, num_classes)
            seg_target: Segmentation targets (B, H, W) or None
            cls_target: Classification targets (B,)
            has_mask: Boolean tensor indicating which samples have masks (B,)
        
        Returns:
            total_loss: Combined loss (scalar)
            loss_dict: Dictionary with individual loss components
        """
        batch_size = cls_pred.size(0)
        device = cls_pred.device
        
        # Initialize loss components
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {
            'seg_loss': 0.0,
            'cls_loss': 0.0,
            'total_loss': 0.0,
            'n_seg_samples': 0,
            'n_cls_samples': batch_size,
        }
        
        # Classification loss (all samples)
        cls_loss = self.cls_loss_fn(cls_pred, cls_target)
        total_loss += self.lambda_cls * cls_loss
        loss_dict['cls_loss'] = cls_loss.item()
        
        # Segmentation loss (only samples with masks)
        if seg_pred is not None and seg_target is not None and has_mask.any():
            # Filter samples that have masks
            mask_indices = has_mask.nonzero(as_tuple=True)[0]
            
            if len(mask_indices) > 0:
                # seg_pred needs filtering (full batch), seg_target is already filtered
                seg_pred_masked = seg_pred[mask_indices]
                # seg_target is already the stacked masks, no indexing needed
                
                seg_loss = self.seg_loss_fn(seg_pred_masked, seg_target)
                total_loss += seg_loss
                
                loss_dict['seg_loss'] = seg_loss.item()
                loss_dict['n_seg_samples'] = len(mask_indices)
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


def create_multi_task_loss(config: dict) -> MultiTaskLoss:
    """
    Factory function to create multi-task loss from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        MultiTaskLoss instance
    """
    loss_config = config.get('loss', {})
    
    # Get class weights if specified
    class_weights = None
    if 'class_weights' in loss_config:
        class_weights = torch.tensor(loss_config['class_weights'], dtype=torch.float32)
    
    return MultiTaskLoss(
        seg_loss_type=loss_config.get('seg_loss_type', 'dice_bce'),
        cls_loss_type=loss_config.get('cls_loss_type', 'ce'),
        lambda_cls=loss_config.get('lambda_cls', 1.0),
        dice_weight=loss_config.get('dice_weight', 0.5),
        bce_weight=loss_config.get('bce_weight', 0.5),
        class_weights=class_weights,
    )
