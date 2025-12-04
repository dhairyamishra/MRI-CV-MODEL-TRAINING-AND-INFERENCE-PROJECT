"""
Evaluation script for brain tumor classification.

This script evaluates a trained classifier and generates:
- Accuracy, ROC-AUC, PR-AUC metrics
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Per-class metrics
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from tqdm import tqdm
from typing import Dict, Tuple, List
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import create_classifier
from src.data.kaggle_mri_dataset import create_dataloaders
from src.data.transforms import get_val_transforms


class ClassifierEvaluator:
    """
    Evaluator for brain tumor classification models.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to configuration YAML file
            checkpoint_path: Path to model checkpoint
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create results directory
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Create data loaders
        _, _, self.test_loader = create_dataloaders(
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            val_transform=get_val_transforms()
        )
        
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        # Create model
        model = create_classifier(
            model_name=self.config['model']['name'],
            pretrained=False,
            num_classes=self.config['model']['num_classes'],
            dropout=self.config['model']['dropout']
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best metric: {checkpoint['best_metric']:.4f}")
        
        return model
    
    @torch.no_grad()
    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Run predictions on test set.
        
        Returns:
            all_labels: Ground truth labels
            all_preds: Predicted labels
            all_probs: Predicted probabilities
            all_ids: Sample IDs
        """
        all_labels = []
        all_preds = []
        all_probs = []
        all_ids = []
        
        print("\nRunning predictions...")
        sample_idx = 0
        for images, labels in tqdm(self.test_loader):
            images = images.to(self.device)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of tumor class
            # Generate IDs since dataset doesn't return them
            batch_ids = [f"sample_{sample_idx + i}" for i in range(len(labels))]
            all_ids.extend(batch_ids)
            sample_idx += len(labels)
        
        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            all_ids
        )
    
    def compute_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            labels: Ground truth labels
            preds: Predicted labels
            probs: Predicted probabilities
        
        Returns:
            metrics: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'roc_auc': roc_auc_score(labels, probs),
            'pr_auc': average_precision_score(labels, probs),
        }
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = metrics['sensitivity']
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                              (metrics['precision'] + metrics['recall']) \
                              if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        save_path: Path
    ):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved confusion matrix to {save_path}")
    
    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_path: Path
    ):
        """Plot and save ROC curve."""
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved ROC curve to {save_path}")
    
    def plot_pr_curve(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_path: Path
    ):
        """Plot and save Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved PR curve to {save_path}")
    
    def save_predictions(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
        ids: List[str],
        save_path: Path
    ):
        """Save predictions to CSV."""
        import pandas as pd
        
        df = pd.DataFrame({
            'id': ids,
            'true_label': labels,
            'predicted_label': preds,
            'tumor_probability': probs,
            'correct': labels == preds
        })
        
        df.to_csv(save_path, index=False)
        print(f"✓ Saved predictions to {save_path}")
    
    def save_metrics(self, metrics: Dict[str, float], save_path: Path):
        """Save metrics to JSON."""
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"✓ Saved metrics to {save_path}")
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way."""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
        print(f"F1 Score:     {metrics['f1_score']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print("="*50 + "\n")
    
    def evaluate(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*50)
        print("EVALUATING CLASSIFIER")
        print("="*50 + "\n")
        
        # Get predictions
        labels, preds, probs, ids = self.predict()
        
        # Compute metrics
        metrics = self.compute_metrics(labels, preds, probs)
        self.print_metrics(metrics)
        
        # Save metrics
        if self.config['evaluation']['compute_metrics']:
            self.save_metrics(metrics, self.results_dir / 'metrics.json')
        
        # Save predictions
        if self.config['evaluation']['save_predictions']:
            self.save_predictions(
                labels, preds, probs, ids,
                self.results_dir / 'predictions.csv'
            )
        
        # Plot confusion matrix
        if self.config['evaluation']['save_confusion_matrix']:
            self.plot_confusion_matrix(
                labels, preds,
                self.results_dir / 'confusion_matrix.png'
            )
        
        # Plot ROC curve
        if self.config['evaluation']['save_roc_curve']:
            self.plot_roc_curve(
                labels, probs,
                self.results_dir / 'roc_curve.png'
            )
        
        # Plot PR curve
        if self.config['evaluation']['save_pr_curve']:
            self.plot_pr_curve(
                labels, probs,
                self.results_dir / 'pr_curve.png'
            )
        
        print("\n✓ Evaluation complete!")
        print(f"Results saved to: {self.results_dir}")
        
        return metrics


def evaluate_classifier(config_path: str, checkpoint_path: str) -> Dict[str, float]:
    """
    Evaluate a trained classifier.
    
    Args:
        config_path: Path to configuration YAML file
        checkpoint_path: Path to model checkpoint
    
    Returns:
        metrics: Dictionary of evaluation metrics
    
    Example:
        >>> metrics = evaluate_classifier(
        ...     'configs/config_cls.yaml',
        ...     'checkpoints/cls/best_model.pth'
        ... )
    """
    evaluator = ClassifierEvaluator(config_path, checkpoint_path)
    metrics = evaluator.evaluate()
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate brain tumor classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_cls.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/cls/best_model.pth',
        help='Path to model checkpoint'
    )
    
    args = parser.parse_args()
    
    evaluate_classifier(args.config, args.checkpoint)
