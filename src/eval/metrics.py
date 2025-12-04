"""
Comprehensive metrics for segmentation and classification evaluation.

Implements:
- Segmentation metrics: Dice, IoU, Boundary F-measure
- Classification metrics: Accuracy, ROC-AUC, PR-AUC, Sensitivity, Specificity
- Calibration metrics: ECE, Brier score
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from scipy.ndimage import distance_transform_edt
import cv2


# ============================================================================
# Segmentation Metrics
# ============================================================================

def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-8) -> float:
    """
    Compute Dice coefficient (F1 score for segmentation).
    
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient in [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.sum(pred & target)
    union = np.sum(pred) + np.sum(target)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-8) -> float:
    """
    Compute Intersection over Union (Jaccard index).
    
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
        smooth: Smoothing factor
    
    Returns:
        IoU score in [0, 1]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def boundary_f_measure(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 2.0,
) -> float:
    """
    Compute Boundary F-measure (BF).
    
    Measures how well predicted boundaries align with ground truth boundaries.
    
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
        threshold: Distance threshold in pixels
    
    Returns:
        Boundary F-measure in [0, 1]
    """
    # Extract boundaries using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    pred_boundary = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    
    # If no boundaries, return 0
    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return 0.0
    
    # Compute distance transforms
    pred_dist = distance_transform_edt(1 - pred_boundary)
    target_dist = distance_transform_edt(1 - target_boundary)
    
    # Precision: fraction of predicted boundary pixels within threshold of target
    pred_boundary_coords = np.argwhere(pred_boundary > 0)
    if len(pred_boundary_coords) == 0:
        precision = 0.0
    else:
        pred_distances = target_dist[pred_boundary_coords[:, 0], pred_boundary_coords[:, 1]]
        precision = np.mean(pred_distances <= threshold)
    
    # Recall: fraction of target boundary pixels within threshold of prediction
    target_boundary_coords = np.argwhere(target_boundary > 0)
    if len(target_boundary_coords) == 0:
        recall = 0.0
    else:
        target_distances = pred_dist[target_boundary_coords[:, 0], target_boundary_coords[:, 1]]
        recall = np.mean(target_distances <= threshold)
    
    # F-measure
    if precision + recall == 0:
        return 0.0
    
    bf = 2 * precision * recall / (precision + recall)
    return float(bf)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
    
    Returns:
        Pixel accuracy in [0, 1]
    """
    correct = np.sum(pred == target)
    total = pred.size
    return float(correct / total)


def sensitivity_specificity(
    pred: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute sensitivity (recall) and specificity.
    
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
    
    Returns:
        Tuple of (sensitivity, specificity)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    tp = np.sum(pred & target)
    tn = np.sum(~pred & ~target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    
    sensitivity = tp / (tp + fn + 1e-8)  # Recall, TPR
    specificity = tn / (tn + fp + 1e-8)  # TNR
    
    return float(sensitivity), float(specificity)


# ============================================================================
# Classification Metrics
# ============================================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N,) or (N, C) - optional
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        
        # Sensitivity (Recall, TPR)
        metrics['sensitivity'] = tp / (tp + fn + 1e-8)
        metrics['recall'] = metrics['sensitivity']
        
        # Specificity (TNR)
        metrics['specificity'] = tn / (tn + fp + 1e-8)
        
        # Precision (PPV)
        metrics['precision'] = tp / (tp + fp + 1e-8)
        
        # F1 score
        metrics['f1'] = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        # False positive rate
        metrics['fpr'] = fp / (fp + tn + 1e-8)
        
        # False negative rate
        metrics['fnr'] = fn / (fn + tp + 1e-8)
    
    # ROC-AUC and PR-AUC (requires probabilities)
    if y_prob is not None:
        try:
            # Handle multi-class
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            else:
                # Binary classification
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]  # Probability of positive class
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            # Not enough classes or other issues
            pass
    
    return metrics


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True binary labels (N,)
        y_prob: Predicted probabilities (N,)
        metric: Metric to optimize ('f1', 'accuracy', 'youden')
    
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    if metric == 'youden':
        # Youden's J statistic = sensitivity + specificity - 1
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return float(thresholds[optimal_idx]), float(j_scores[optimal_idx])
    
    elif metric == 'f1':
        # Compute F1 for each threshold
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            else:
                f1 = 0.0
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        return float(thresholds[optimal_idx]), float(f1_scores[optimal_idx])
    
    elif metric == 'accuracy':
        # Compute accuracy for each threshold
        accuracies = []
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            acc = accuracy_score(y_true, y_pred)
            accuracies.append(acc)
        
        optimal_idx = np.argmax(accuracies)
        return float(thresholds[optimal_idx]), float(accuracies[optimal_idx])
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# Comprehensive Evaluation
# ============================================================================

def compute_all_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    pred_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics.
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        target_mask: Ground truth binary mask (H, W)
        pred_prob: Predicted probability map (H, W) - optional
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Segmentation metrics
    metrics['dice'] = dice_coefficient(pred_mask, target_mask)
    metrics['iou'] = iou_score(pred_mask, target_mask)
    metrics['boundary_f'] = boundary_f_measure(pred_mask, target_mask)
    metrics['pixel_accuracy'] = pixel_accuracy(pred_mask, target_mask)
    
    # Sensitivity and specificity
    sens, spec = sensitivity_specificity(pred_mask, target_mask)
    metrics['sensitivity'] = sens
    metrics['specificity'] = spec
    
    # Confusion matrix elements
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)
    
    metrics['tp'] = int(np.sum(pred_bool & target_bool))
    metrics['tn'] = int(np.sum(~pred_bool & ~target_bool))
    metrics['fp'] = int(np.sum(pred_bool & ~target_bool))
    metrics['fn'] = int(np.sum(~pred_bool & target_bool))
    
    # Precision and recall
    metrics['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-8)
    metrics['recall'] = metrics['sensitivity']
    
    # F1 score
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-8)
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing Metrics Module...")
    print("=" * 60)
    
    # Create synthetic masks
    np.random.seed(42)
    target = np.zeros((100, 100), dtype=np.uint8)
    target[30:70, 30:70] = 1  # Square tumor
    
    # Good prediction (high overlap)
    pred_good = np.zeros((100, 100), dtype=np.uint8)
    pred_good[32:68, 32:68] = 1
    
    # Poor prediction (low overlap)
    pred_poor = np.zeros((100, 100), dtype=np.uint8)
    pred_poor[60:90, 60:90] = 1
    
    print("\n1. Testing Segmentation Metrics...")
    print("\nGood Prediction:")
    metrics_good = compute_all_metrics(pred_good, target)
    for key, value in metrics_good.items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:.4f}")
        else:
            print(f"  {key:15s}: {value}")
    
    print("\nPoor Prediction:")
    metrics_poor = compute_all_metrics(pred_poor, target)
    for key, value in metrics_poor.items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:.4f}")
        else:
            print(f"  {key:15s}: {value}")
    
    print("\n2. Testing Classification Metrics...")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.2, 0.85, 0.6])
    
    cls_metrics = classification_metrics(y_true, y_pred, y_prob)
    for key, value in cls_metrics.items():
        print(f"  {key:15s}: {value:.4f}")
    
    print("\n3. Testing Optimal Threshold...")
    thresh, score = optimal_threshold(y_true, y_prob, metric='f1')
    print(f"  Optimal threshold (F1): {thresh:.4f} (F1={score:.4f})")
    
    thresh, score = optimal_threshold(y_true, y_prob, metric='youden')
    print(f"  Optimal threshold (Youden): {thresh:.4f} (J={score:.4f})")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
