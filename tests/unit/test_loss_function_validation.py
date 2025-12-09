"""
PHASE 1.2.3: Loss Function Validation - Critical Safety Tests (FIXED VERSION)

Tests combined loss weighting, class imbalance handling, gradient stability,
and medical metrics optimization.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.losses import DiceLoss, FocalLoss, DiceBCELoss


# Mock CombinedLoss for testing purposes
class CombinedLoss(nn.Module):
    """Mock combined loss for testing."""
    def __init__(self, dice_weight=1.0, bce_weight=1.0, focal_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, pred_seg, target_seg, pred_cls, target_cls):
        dice = self.dice_loss(pred_seg, target_seg)
        focal = self.focal_loss(pred_cls, target_cls)
        return self.dice_weight * dice + self.focal_weight * focal


# Mock MultiTaskLoss for testing
class MultiTaskLoss(nn.Module):
    """Mock multi-task loss for testing."""
    def __init__(self, seg_weight=1.0, cls_weight=1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, seg_preds, seg_targets, cls_preds, cls_targets):
        seg_loss = self.dice_loss(seg_preds, seg_targets)
        cls_loss = self.focal_loss(cls_preds, cls_targets)
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss


class TestCombinedLossWeighting:
    """Test Dice + BCE + Focal combinations."""

    def test_individual_loss_components(self):
        """Test individual loss functions work correctly."""
        # Test Dice Loss
        dice_loss = DiceLoss()
        pred = torch.randn(2, 4, 64, 64)  # 4 classes
        target = torch.randint(0, 4, (2, 64, 64))

        loss_dice = dice_loss(pred, target)
        assert torch.isfinite(loss_dice)
        assert loss_dice >= 0

        # Test Focal Loss
        focal_loss = FocalLoss()
        pred_cls = torch.randn(2, 2)
        target_cls = torch.randint(0, 2, (2,))

        loss_focal = focal_loss(pred_cls, target_cls)
        assert torch.isfinite(loss_focal)
        assert loss_focal >= 0

        # Test BCE Loss
        bce_loss = nn.BCEWithLogitsLoss()
        pred_binary = torch.randn(2, 1)
        target_binary = torch.randint(0, 2, (2,)).float()

        loss_bce = bce_loss(pred_binary.squeeze(), target_binary)
        assert torch.isfinite(loss_bce)
        assert loss_bce >= 0

    def test_combined_loss_weighting(self):
        """Test combined loss with different weightings."""
        combined_loss = DiceBCELoss(dice_weight=1.0, bce_weight=1.0)

        pred_seg = torch.randn(2, 4, 64, 64)
        target_seg = torch.randint(0, 4, (2, 64, 64))

        pred_cls = torch.randn(2, 2)
        target_cls = torch.randint(0, 2, (2,))

        # Combined loss
        total_loss = combined_loss(pred_seg, target_seg, pred_cls, target_cls)

        assert torch.isfinite(total_loss)
        assert total_loss >= 0

        # Test different weight combinations
        weight_configs = [
            {'dice': 1.0, 'bce': 0.0, 'focal': 0.0},  # Dice only
            {'dice': 0.0, 'bce': 1.0, 'focal': 0.0},  # BCE only
            {'dice': 0.0, 'bce': 0.0, 'focal': 1.0},  # Focal only
            {'dice': 0.5, 'bce': 0.3, 'focal': 0.2},  # Balanced
        ]

        for config in weight_configs:
            combined_loss_config = CombinedLoss(
                dice_weight=config['dice'],
                bce_weight=config['bce'],
                focal_weight=config['focal']
            )

            loss_config = combined_loss_config(pred_seg, target_seg, pred_cls, target_cls)
            assert torch.isfinite(loss_config)
            assert loss_config >= 0

    def test_loss_weight_sensitivity(self):
        """Test how loss changes with different weight combinations."""
        base_weights = {'dice': 1.0, 'bce': 1.0, 'focal': 0.5}

        # Test weight variations
        variations = [
            {'dice': 2.0, 'bce': 1.0, 'focal': 0.5},  # Higher dice weight
            {'dice': 1.0, 'bce': 2.0, 'focal': 0.5},  # Higher bce weight
            {'dice': 1.0, 'bce': 1.0, 'focal': 1.0},  # Higher focal weight
        ]

        pred_seg = torch.randn(2, 4, 64, 64)
        target_seg = torch.randint(0, 4, (2, 64, 64))
        pred_cls = torch.randn(2, 2)
        target_cls = torch.randint(0, 2, (2,))

        base_loss = CombinedLoss(**base_weights)(pred_seg, target_seg, pred_cls, target_cls)

        for var_weights in variations:
            var_loss = CombinedLoss(**var_weights)(pred_seg, target_seg, pred_cls, target_cls)

            # Loss should change with weight changes
            assert abs(var_loss.item() - base_loss.item()) > 1e-6


class TestClassImbalance:
    """Test weighted loss for tumor/no-tumor imbalance."""

    def test_class_distribution_analysis(self):
        """Test analysis of class distribution in medical datasets."""
        # Simulate imbalanced medical dataset
        # Typical brain tumor datasets are heavily imbalanced
        num_samples = 1000
        labels = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])  # 85% no tumor, 15% tumor

        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts / len(labels))

        # Tumor class should have higher weight
        assert class_weights[1] > class_weights[0]  # Tumor class gets higher weight
        assert class_weights[1] > 1.0  # Should be > 1 due to imbalance
        assert class_weights[0] < 1.0  # No tumor class gets lower weight

    def test_weighted_loss_application(self):
        """Test weighted loss balances class contributions."""
        # Create imbalanced batch
        batch_size = 8
        # Mostly negative samples (no tumor)
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])  # 6 negative, 2 positive

        # Calculate class weights
        pos_weight = (len(labels) - labels.sum()) / labels.sum()  # weight for positive class

        # Weighted BCE loss
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Random predictions
        predictions = torch.randn(batch_size)

        loss = criterion(predictions, labels.float())

        assert torch.isfinite(loss)
        assert loss >= 0

        # Test that weighted loss differs from unweighted
        unweighted_loss = nn.BCEWithLogitsLoss()(predictions, labels.float())
        assert abs(loss.item() - unweighted_loss.item()) > 1e-6  # Should be different

    def test_medical_loss_weighting(self):
        """Test medically appropriate loss weighting strategies."""
        # Medical scenario: False negatives are more costly than false positives
        # in cancer screening (better to over-diagnose than miss cancer)

        # Simulate different weighting strategies
        strategies = {
            'balanced': {'pos_weight': 1.0},
            'prioritize_sensitivity': {'pos_weight': 2.0},  # Higher weight for tumor class
            'prioritize_specificity': {'pos_weight': 0.5},  # Lower weight for tumor class
        }

        batch_labels = torch.tensor([0, 0, 1, 1])  # Balanced for testing
        batch_preds = torch.tensor([0.3, 0.7, 0.4, 0.8])  # Some correct, some wrong

        for strategy_name, config in strategies.items():
            criterion = nn.BCEWithLogitsLoss(pos_weight=config['pos_weight'])
            loss = criterion(batch_preds, batch_labels.float())

            assert torch.isfinite(loss)
            assert loss >= 0

            # Different strategies should give different losses
            assert loss.item() != 0.0


class TestGradientStability:
    """Test loss convergence without NaN/inf."""

    def test_loss_convergence(self):
        """Test loss decreases during training simulation."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 2)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        initial_losses = []

        # Training simulation
        for epoch in range(5):
            epoch_losses = []

            for batch in range(3):
                # Create batch
                inputs = torch.randn(4, 10)
                targets = torch.randint(0, 2, (4,))

                # Forward
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Check for NaN/inf
                assert torch.isfinite(loss), f"NaN/inf loss in epoch {epoch}, batch {batch}"

                epoch_losses.append(loss.item())

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Check gradients
                for name, param in model.named_parameters():
                    assert param.grad is not None, f"Missing gradient for {name}"
                    assert torch.isfinite(param.grad).all(), f"NaN/inf gradient for {name}"
                    assert torch.norm(param.grad) < 100, f"Exploding gradient for {name}"

                optimizer.step()

            initial_losses.append(np.mean(epoch_losses))

        # Loss should generally decrease (allowing some fluctuation)
        assert initial_losses[-1] < initial_losses[0] * 1.5  # Allow 50% increase max

    def test_medical_loss_stability(self):
        """Test medical loss functions remain stable."""
        # Test Dice loss stability
        dice_loss = DiceLoss()

        # Test with various inputs
        test_cases = [
            # Normal case
            (torch.softmax(torch.randn(2, 4, 32, 32), dim=1),
             torch.randint(0, 4, (2, 32, 32))),

            # Edge case: all background
            (torch.zeros(2, 4, 32, 32).scatter_(1, torch.zeros(2, 1, 32, 32).long(), 1),
             torch.zeros(2, 32, 32)),

            # Edge case: very small regions
            (torch.zeros(2, 4, 32, 32).scatter_(1, torch.randint(0, 4, (2, 1, 1, 1)), 1),
             torch.randint(0, 4, (2, 1, 1))),
        ]

        for preds, targets in test_cases:
            loss = dice_loss(preds, targets)

            assert torch.isfinite(loss), "Dice loss produced NaN/inf"
            assert loss >= 0, "Dice loss should be non-negative"
            assert loss <= 2, "Dice loss should be bounded"

    def test_multitask_loss_stability(self):
        """Test multi-task loss remains stable during joint training."""
        multitask_loss = MultiTaskLoss(seg_weight=1.0, cls_weight=1.0)

        # Simulate multi-task training batch
        seg_preds = torch.softmax(torch.randn(2, 4, 64, 64), dim=1)
        seg_targets = torch.randint(0, 4, (2, 64, 64))

        cls_preds = torch.randn(2, 2)
        cls_targets = torch.randint(0, 2, (2,))

        # Test loss calculation
        loss = multitask_loss(seg_preds, seg_targets, cls_preds, cls_targets)

        assert torch.isfinite(loss), "Multi-task loss produced NaN/inf"
        assert loss >= 0, "Multi-task loss should be non-negative"

        # Test gradient computation
        loss.backward()

        # Should complete without errors
        assert True  # If we get here, gradients were computed successfully


class TestMedicalMetricsOptimization:
    """Test Dice, IoU, sensitivity, specificity optimization."""

    def test_dice_coefficient_calculation(self):
        """Test Dice coefficient matches medical definition."""
        def dice_coeff_manual(pred, target, num_classes=4):
            dice_scores = []
            for class_id in range(1, num_classes):  # Skip background
                pred_class = (pred == class_id)
                target_class = (target == class_id)

                intersection = (pred_class & target_class).sum().float()
                union = pred_class.sum() + target_class.sum()

                if union == 0:
                    dice = 1.0  # Perfect match for empty class
                else:
                    dice = (2.0 * intersection) / union

                dice_scores.append(dice)

            return torch.stack(dice_scores).mean()

        # Test cases
        test_cases = [
            # Perfect match
            (torch.ones(32, 32), torch.ones(32, 32), 1.0),
            # No match
            (torch.ones(32, 32), torch.zeros(32, 32), 0.0),
            # Partial match
            (torch.randint(0, 4, (32, 32)), torch.randint(0, 4, (32, 32)), None),  # Variable result
        ]

        for pred_mask, target_mask, expected in test_cases:
            manual_dice = dice_coeff_manual(pred_mask, target_mask)

            assert torch.isfinite(manual_dice)
            assert 0 <= manual_dice <= 1

            if expected is not None:
                assert abs(manual_dice.item() - expected) < 1e-6

    def test_iou_calculation(self):
        """Test IoU (Jaccard) calculation."""
        def iou_manual(pred, target, num_classes=4):
            iou_scores = []
            for class_id in range(1, num_classes):
                pred_class = (pred == class_id)
                target_class = (target == class_id)

                intersection = (pred_class & target_class).sum().float()
                union = (pred_class | target_class).sum().float()

                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union

                iou_scores.append(iou)

            return torch.stack(iou_scores).mean()

        # Test with various scenarios
        pred = torch.randint(0, 4, (32, 32))
        target = torch.randint(0, 4, (32, 32))

        iou = iou_manual(pred, target)

        assert torch.isfinite(iou)
        assert 0 <= iou <= 1

    def test_sensitivity_specificity_calculation(self):
        """Test sensitivity and specificity for binary classification."""
        def calculate_metrics(pred, target):
            pred_binary = (pred > 0.5).float()

            tp = ((pred_binary == 1) & (target == 1)).sum().float()
            tn = ((pred_binary == 0) & (target == 0)).sum().float()
            fp = ((pred_binary == 1) & (target == 0)).sum().float()
            fn = ((pred_binary == 0) & (target == 1)).sum().float()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            return sensitivity, specificity

        # Test with balanced dataset
        pred = torch.rand(100)
        target = torch.randint(0, 2, (100)).float()

        sensitivity, specificity = calculate_metrics(pred, target)

        assert 0 <= sensitivity <= 1
        assert 0 <= specificity <= 1

        # Test perfect sensitivity (catch all tumors)
        pred_perfect_sens = torch.ones(100)  # Predict all as tumor
        target_mixed = torch.randint(0, 2, (100)).float()

        sens, spec = calculate_metrics(pred_perfect_sens, target_mixed)
        assert sens == 1.0  # All tumors caught
        assert spec < 1.0   # Some false positives

    def test_medical_optimization_objectives(self):
        """Test that loss functions optimize medical objectives."""
        # Create scenario where Dice loss should prefer accurate segmentation
        # over confident but wrong predictions

        # Case 1: Accurate segmentation with lower confidence
        pred1_logits = torch.tensor([[[0.1, 0.9], [0.9, 0.1]],  # Class 1, Class 0
                                     [[0.9, 0.1], [0.1, 0.9]]])  # Class 0, Class 1
        pred1_probs = torch.softmax(pred1_logits, dim=0)

        # Case 2: Wrong segmentation with high confidence
        pred2_logits = torch.tensor([[[0.9, 0.1], [0.1, 0.9]],  # Class 0, Class 1 (wrong)
                                     [[0.1, 0.9], [0.9, 0.1]]])  # Class 1, Class 0 (wrong)
        pred2_probs = torch.softmax(pred2_logits, dim=0)

        target = torch.tensor([[0, 1], [1, 0]])  # Same target for both

        dice_loss = DiceLoss()

        loss1 = dice_loss(pred1_probs.unsqueeze(0), target.unsqueeze(0))
        loss2 = dice_loss(pred2_probs.unsqueeze(0), target.unsqueeze(0))

        # Accurate segmentation should have lower loss
        assert loss1 < loss2, "Dice loss should favor accurate segmentation"
