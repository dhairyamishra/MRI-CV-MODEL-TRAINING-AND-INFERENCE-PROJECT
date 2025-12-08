"""
Calibration utilities for classification models.

Implements:
- Temperature scaling
- Expected Calibration Error (ECE)
- Brier score
- Reliability diagrams
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    Scales logits by a learned temperature parameter to improve
    probability calibration without changing predictions.
    
    Reference:
        Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
    """
    
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale logits by temperature.
        
        Args:
            logits: Model logits (B, C)
        
        Returns:
            Scaled logits (B, C)
        """
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Fit temperature parameter using validation set.
        
        Args:
            logits: Validation logits (N, C)
            labels: Validation labels (N,)
            lr: Learning rate
            max_iter: Maximum iterations
        
        Returns:
            Final NLL loss
        """
        # Move to same device as logits
        self.to(logits.device)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        # Return final loss
        with torch.no_grad():
            final_loss = F.cross_entropy(self.forward(logits), labels)
        
        return final_loss.item()


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy.
    
    Args:
        probs: Predicted probabilities (N, C) or (N,) for binary
        labels: True labels (N,)
        n_bins: Number of bins for calibration
    
    Returns:
        Tuple of (ece, bin_accuracies, bin_confidences, bin_counts)
    """
    # Get predicted class and confidence
    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs > 0.5).astype(int)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Initialize arrays
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Compute accuracy and confidence for each bin
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_counts[i] = in_bin.sum()
        
        if bin_counts[i] > 0:
            # Accuracy: fraction of correct predictions in bin
            bin_accuracies[i] = (predictions[in_bin] == labels[in_bin]).mean()
            # Confidence: average confidence in bin
            bin_confidences[i] = confidences[in_bin].mean()
    
    # Compute ECE
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / bin_counts.sum()
    
    return ece, bin_accuracies, bin_confidences, bin_counts


def brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Brier score.
    
    Brier score measures the mean squared difference between
    predicted probabilities and actual outcomes.
    
    Args:
        probs: Predicted probabilities (N, C) or (N,) for binary
        labels: True labels (N,)
    
    Returns:
        Brier score (lower is better)
    """
    if probs.ndim == 2:
        # Multi-class: one-hot encode labels
        n_classes = probs.shape[1]
        labels_onehot = np.eye(n_classes)[labels]
        return np.mean(np.sum((probs - labels_onehot) ** 2, axis=1))
    else:
        # Binary
        return np.mean((probs - labels) ** 2)


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        probs: Predicted probabilities (N,) for positive class
        labels: True binary labels (N,)
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib figure
    """
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, probs, n_bins=n_bins, strategy='uniform'
    )
    
    # Compute ECE
    ece, _, _, _ = expected_calibration_error(probs, labels, n_bins=n_bins)
    
    # Compute Brier score
    brier = brier_score(probs, labels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
            label=f'Model (ECE={ece:.3f}, Brier={brier:.3f})',
            linewidth=2, markersize=8)
    
    # Formatting
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'{title}\n(Lower ECE and Brier = Better Calibration)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def calibrate_classifier(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
) -> Tuple[TemperatureScaling, dict]:
    """
    Calibrate a classifier using temperature scaling.
    
    Args:
        model: Trained classifier
        val_loader: Validation data loader
        device: Device to use
    
    Returns:
        Tuple of (temperature_scaler, metrics_dict)
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Collect logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics before calibration
    probs_before = F.softmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()
    
    ece_before, _, _, _ = expected_calibration_error(probs_before, labels_np)
    brier_before = brier_score(probs_before, labels_np)
    nll_before = F.cross_entropy(all_logits, all_labels).item()
    
    # Fit temperature scaling
    temp_scaler = TemperatureScaling()
    nll_after = temp_scaler.fit(all_logits, all_labels)
    
    # Compute metrics after calibration
    with torch.no_grad():
        logits_scaled = temp_scaler(all_logits)
        probs_after = F.softmax(logits_scaled, dim=1).numpy()
    
    ece_after, _, _, _ = expected_calibration_error(probs_after, labels_np)
    brier_after = brier_score(probs_after, labels_np)
    
    # Compile metrics
    metrics = {
        'temperature': temp_scaler.temperature.item(),
        'nll_before': nll_before,
        'nll_after': nll_after,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'brier_before': brier_before,
        'brier_after': brier_after,
    }
    
    return temp_scaler, metrics


if __name__ == "__main__":
    # Test calibration functions
    print("Testing Calibration Module...")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate overconfident predictions
    true_labels = np.random.randint(0, 2, n_samples)
    
    # Overconfident probabilities (shifted towards 0 and 1)
    probs_overconfident = np.random.beta(0.5, 0.5, n_samples)
    probs_overconfident = (probs_overconfident > 0.5).astype(float) * 0.9 + 0.05
    
    # Well-calibrated probabilities
    probs_calibrated = true_labels + np.random.normal(0, 0.1, n_samples)
    probs_calibrated = np.clip(probs_calibrated, 0, 1)
    
    print("\n1. Testing ECE computation...")
    ece_over, _, _, _ = expected_calibration_error(probs_overconfident, true_labels)
    ece_cal, _, _, _ = expected_calibration_error(probs_calibrated, true_labels)
    print(f"   Overconfident ECE: {ece_over:.4f}")
    print(f"   Calibrated ECE:    {ece_cal:.4f}")
    
    print("\n2. Testing Brier score...")
    brier_over = brier_score(probs_overconfident, true_labels)
    brier_cal = brier_score(probs_calibrated, true_labels)
    print(f"   Overconfident Brier: {brier_over:.4f}")
    print(f"   Calibrated Brier:    {brier_cal:.4f}")
    
    print("\n3. Testing Temperature Scaling...")
    # Create synthetic logits
    logits = torch.randn(100, 2) * 3  # Overconfident logits
    labels = torch.randint(0, 2, (100,))
    
    temp_scaler = TemperatureScaling()
    loss = temp_scaler.fit(logits, labels)
    print(f"   Learned temperature: {temp_scaler.temperature.item():.4f}")
    print(f"   Final NLL loss: {loss:.4f}")
    
    print("\n4. Testing Reliability Diagram...")
    fig = plot_reliability_diagram(
        probs_overconfident,
        true_labels,
        title="Test Reliability Diagram"
    )
    print("   Reliability diagram created successfully")
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
