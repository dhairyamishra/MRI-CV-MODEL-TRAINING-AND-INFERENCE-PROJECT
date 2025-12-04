"""
Loss functions for medical image segmentation.

Implements various loss functions commonly used for brain tumor segmentation:
- Dice Loss
- Binary Cross-Entropy (BCE) Loss
- Combined Dice + BCE Loss
- Tversky Loss (handles class imbalance)
- Focal Loss (focuses on hard examples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice coefficient measures overlap between prediction and ground truth.
    Dice Loss = 1 - Dice Coefficient
    
    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits or probabilities
            target: Ground truth (B, C, H, W) or (B, H, W) for binary
        
        Returns:
            Dice loss value
        """
        # Apply sigmoid for binary, softmax for multi-class
        if pred.shape[1] == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.softmax(pred, dim=1)
        
        # Handle target shape
        if target.dim() == 3:  # (B, H, W)
            target = target.unsqueeze(1)  # (B, 1, H, W)
        
        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), pred.size(1), -1)  # (B, C, H*W)
        target = target.view(target.size(0), target.size(1), -1)  # (B, C, H*W)
        
        # Calculate Dice coefficient
        intersection = (pred * target).sum(dim=2)  # (B, C)
        union = pred.sum(dim=2) + target.sum(dim=2)  # (B, C)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with logits.
    
    Wrapper around nn.BCEWithLogitsLoss for consistency.
    
    Args:
        pos_weight: Weight for positive class (default: None)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 1, H, W) - logits
            target: Ground truth (B, 1, H, W) or (B, H, W)
        
        Returns:
            BCE loss value
        """
        if target.dim() == 3:
            target = target.unsqueeze(1).float()
        
        return self.bce(pred, target)


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss.
    
    Combines the benefits of both losses:
    - Dice: Good for class imbalance, focuses on overlap
    - BCE: Pixel-wise classification, stable gradients
    
    Args:
        dice_weight: Weight for Dice loss (default: 0.5)
        bce_weight: Weight for BCE loss (default: 0.5)
        smooth: Smoothing factor for Dice (default: 1.0)
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = BCELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, C, H, W) or (B, H, W)
        
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling class imbalance.
    
    Generalization of Dice loss with parameters alpha and beta
    to control false positives and false negatives.
    
    Args:
        alpha: Weight for false positives (default: 0.5)
        beta: Weight for false negatives (default: 0.5)
        smooth: Smoothing factor (default: 1.0)
        
    Note:
        - alpha = beta = 0.5: Equivalent to Dice Loss
        - alpha < beta: Penalize false negatives more (recall focus)
        - alpha > beta: Penalize false positives more (precision focus)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
    ):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, C, H, W) or (B, H, W)
        
        Returns:
            Tversky loss value
        """
        # Apply sigmoid/softmax
        if pred.shape[1] == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.softmax(pred, dim=1)
        
        # Handle target shape
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Flatten
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        
        # True positives, false positives, false negatives
        tp = (pred * target).sum(dim=2)
        fp = (pred * (1 - target)).sum(dim=2)
        fn = ((1 - pred) * target).sum(dim=2)
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return (1 - tversky).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    
    Args:
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, C, H, W) or (B, H, W)
        
        Returns:
            Focal loss value
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target.float(), reduction='none'
        )
        
        # Probability
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal loss
        focal_loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(
    loss_name: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function by name.
    
    Args:
        loss_name: Name of loss function
            - 'dice': Dice Loss
            - 'bce': Binary Cross-Entropy Loss
            - 'dice_bce': Combined Dice + BCE Loss
            - 'tversky': Tversky Loss
            - 'focal': Focal Loss
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function module
    
    Examples:
        >>> # Dice loss
        >>> loss_fn = get_loss_function('dice', smooth=1.0)
        
        >>> # Combined loss
        >>> loss_fn = get_loss_function('dice_bce', dice_weight=0.7, bce_weight=0.3)
        
        >>> # Tversky for recall focus
        >>> loss_fn = get_loss_function('tversky', alpha=0.3, beta=0.7)
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'bce':
        return BCELoss(**kwargs)
    elif loss_name == 'dice_bce':
        return DiceBCELoss(**kwargs)
    elif loss_name == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Choose from: dice, bce, dice_bce, tversky, focal"
        )


if __name__ == "__main__":
    # Test loss functions
    print("Testing Segmentation Loss Functions...")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    pred = torch.randn(batch_size, 1, 256, 256)  # Logits
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    print(f"\nInput shapes:")
    print(f"  Predictions: {pred.shape}")
    print(f"  Target:      {target.shape}")
    
    # Test 1: Dice Loss
    print("\n1. Dice Loss")
    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Requires grad: {loss.requires_grad}")
    
    # Test 2: BCE Loss
    print("\n2. BCE Loss")
    bce_loss = BCELoss()
    loss = bce_loss(pred, target)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test 3: Combined Dice + BCE
    print("\n3. Dice + BCE Loss")
    combined_loss = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    loss = combined_loss(pred, target)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test 4: Tversky Loss
    print("\n4. Tversky Loss")
    tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)  # Recall focus
    loss = tversky_loss(pred, target)
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Config: alpha=0.3 (FP weight), beta=0.7 (FN weight)")
    
    # Test 5: Focal Loss
    print("\n5. Focal Loss")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(pred, target)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test 6: Multi-class segmentation
    print("\n6. Multi-class Segmentation (4 classes)")
    pred_multi = torch.randn(batch_size, 4, 256, 256)
    target_multi = torch.randint(0, 2, (batch_size, 4, 256, 256)).float()
    
    dice_loss_multi = DiceLoss()
    loss = dice_loss_multi(pred_multi, target_multi)
    print(f"   Predictions: {pred_multi.shape}")
    print(f"   Target:      {target_multi.shape}")
    print(f"   Loss value:  {loss.item():.4f}")
    
    # Test 7: Backward pass
    print("\n7. Testing Backward Pass")
    pred_test = torch.randn(2, 1, 128, 128, requires_grad=True)
    target_test = torch.randint(0, 2, (2, 1, 128, 128)).float()
    
    loss_fn = DiceBCELoss()
    loss = loss_fn(pred_test, target_test)
    loss.backward()
    
    print(f"   Loss computed: {loss.item():.4f}")
    print(f"   Gradient computed: {'✓' if pred_test.grad is not None else '✗'}")
    print(f"   Gradient shape: {pred_test.grad.shape}")
    
    # Test 8: Factory function
    print("\n8. Testing Factory Function")
    loss_names = ['dice', 'bce', 'dice_bce', 'tversky', 'focal']
    
    for name in loss_names:
        loss_fn = get_loss_function(name)
        loss = loss_fn(pred, target)
        print(f"   {name:12s}: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
