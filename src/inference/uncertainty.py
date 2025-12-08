"""
Uncertainty estimation for segmentation models.

Implements:
- Monte Carlo (MC) Dropout
- Test-Time Augmentation (TTA)
- Uncertainty quantification
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F


def enable_dropout(model: nn.Module):
    """
    Enable dropout layers during inference for MC Dropout.
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


def disable_dropout(model: nn.Module):
    """
    Disable dropout layers (return to normal inference).
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.eval()


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Runs multiple stochastic forward passes with dropout enabled
    to estimate prediction uncertainty.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        device: str = 'cuda',
    ):
        """
        Args:
            model: Segmentation model with dropout layers
            n_samples: Number of MC samples
            device: Device to run on
        """
        self.model = model
        self.n_samples = n_samples
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        image: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            image: Input image (1, C, H, W) or (C, H, W)
        
        Returns:
            Dictionary containing:
                - 'mean': Mean probability map (H, W)
                - 'std': Standard deviation map (H, W)
                - 'samples': All probability samples (N, H, W)
        """
        # Ensure correct shape
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dim
        
        image = image.to(self.device)
        
        # Enable dropout
        enable_dropout(self.model)
        
        # Collect samples
        samples = []
        for _ in range(self.n_samples):
            logits = self.model(image)
            probs = torch.sigmoid(logits)
            samples.append(probs.squeeze().cpu().numpy())
        
        # Disable dropout
        disable_dropout(self.model)
        
        # Stack samples
        samples = np.stack(samples, axis=0)  # (N, H, W)
        
        # Compute statistics
        mean_prob = samples.mean(axis=0)
        std_prob = samples.std(axis=0)
        
        return {
            'mean': mean_prob,
            'std': std_prob,
            'samples': samples,
        }


class TTAPredictor:
    """
    Test-Time Augmentation for uncertainty estimation.
    
    Applies multiple augmentations at test time and averages predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        augmentations: Optional[List[str]] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            model: Segmentation model
            augmentations: List of augmentations to apply
                          Options: 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'
            device: Device to run on
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Default augmentations
        if augmentations is None:
            augmentations = ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
        
        self.augmentations = augmentations
    
    def _apply_augmentation(
        self,
        image: torch.Tensor,
        aug_type: str,
    ) -> torch.Tensor:
        """Apply augmentation to image."""
        if aug_type == 'original':
            return image
        elif aug_type == 'hflip':
            return torch.flip(image, dims=[-1])  # Flip width
        elif aug_type == 'vflip':
            return torch.flip(image, dims=[-2])  # Flip height
        elif aug_type == 'rot90':
            return torch.rot90(image, k=1, dims=[-2, -1])
        elif aug_type == 'rot180':
            return torch.rot90(image, k=2, dims=[-2, -1])
        elif aug_type == 'rot270':
            return torch.rot90(image, k=3, dims=[-2, -1])
        else:
            raise ValueError(f"Unknown augmentation: {aug_type}")
    
    def _reverse_augmentation(
        self,
        output: torch.Tensor,
        aug_type: str,
    ) -> torch.Tensor:
        """Reverse augmentation on output."""
        if aug_type == 'original':
            return output
        elif aug_type == 'hflip':
            return torch.flip(output, dims=[-1])
        elif aug_type == 'vflip':
            return torch.flip(output, dims=[-2])
        elif aug_type == 'rot90':
            return torch.rot90(output, k=-1, dims=[-2, -1])  # Rotate back
        elif aug_type == 'rot180':
            return torch.rot90(output, k=-2, dims=[-2, -1])
        elif aug_type == 'rot270':
            return torch.rot90(output, k=-3, dims=[-2, -1])
        else:
            raise ValueError(f"Unknown augmentation: {aug_type}")
    
    @torch.no_grad()
    def predict_with_tta(
        self,
        image: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with Test-Time Augmentation.
        
        Args:
            image: Input image (1, C, H, W) or (C, H, W)
        
        Returns:
            Dictionary containing:
                - 'mean': Mean probability map (H, W)
                - 'std': Standard deviation map (H, W)
                - 'samples': All probability samples (N, H, W)
        """
        # Ensure correct shape
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dim
        
        image = image.to(self.device)
        
        # Collect predictions for each augmentation
        samples = []
        
        for aug_type in self.augmentations:
            # Apply augmentation
            aug_image = self._apply_augmentation(image, aug_type)
            
            # Predict
            logits = self.model(aug_image)
            probs = torch.sigmoid(logits)
            
            # Reverse augmentation
            probs = self._reverse_augmentation(probs, aug_type)
            
            samples.append(probs.squeeze().cpu().numpy())
        
        # Stack samples
        samples = np.stack(samples, axis=0)  # (N, H, W)
        
        # Compute statistics
        mean_prob = samples.mean(axis=0)
        std_prob = samples.std(axis=0)
        
        return {
            'mean': mean_prob,
            'std': std_prob,
            'samples': samples,
        }


class EnsemblePredictor:
    """
    Ensemble prediction combining MC Dropout and TTA.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_mc_samples: int = 5,
        use_tta: bool = True,
        device: str = 'cuda',
    ):
        """
        Args:
            model: Segmentation model
            n_mc_samples: Number of MC Dropout samples
            use_tta: Whether to use TTA
            device: Device to run on
        """
        self.mc_predictor = MCDropoutPredictor(model, n_mc_samples, device)
        
        if use_tta:
            self.tta_predictor = TTAPredictor(model, device=device)
        else:
            self.tta_predictor = None
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        image: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with combined MC Dropout and TTA.
        
        Args:
            image: Input image (1, C, H, W) or (C, H, W)
        
        Returns:
            Dictionary containing:
                - 'mean': Mean probability map (H, W)
                - 'std': Standard deviation map (H, W)
                - 'epistemic': Epistemic uncertainty (model uncertainty)
                - 'aleatoric': Aleatoric uncertainty (data uncertainty)
        """
        all_samples = []
        
        # MC Dropout samples
        mc_result = self.mc_predictor.predict_with_uncertainty(image)
        all_samples.append(mc_result['samples'])
        
        # TTA samples (if enabled)
        if self.tta_predictor is not None:
            tta_result = self.tta_predictor.predict_with_tta(image)
            all_samples.append(tta_result['samples'])
        
        # Combine all samples
        all_samples = np.concatenate(all_samples, axis=0)
        
        # Compute statistics
        mean_prob = all_samples.mean(axis=0)
        total_std = all_samples.std(axis=0)
        
        # Epistemic uncertainty (variance across MC samples)
        epistemic = mc_result['std']
        
        # Aleatoric uncertainty (variance across TTA)
        if self.tta_predictor is not None:
            aleatoric = tta_result['std']
        else:
            aleatoric = np.zeros_like(mean_prob)
        
        return {
            'mean': mean_prob,
            'std': total_std,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
        }


if __name__ == "__main__":
    # Test uncertainty estimation
    print("Testing Uncertainty Estimation...")
    print("=" * 60)
    
    # Create a simple model with dropout
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.dropout = nn.Dropout2d(0.5)
            self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.dropout(x)
            x = self.conv2(x)
            return x
    
    model = SimpleUNet()
    
    # Create test image
    test_image = torch.randn(1, 1, 64, 64)
    
    print("\n1. Testing MC Dropout...")
    mc_predictor = MCDropoutPredictor(model, n_samples=10, device='cpu')
    mc_result = mc_predictor.predict_with_uncertainty(test_image)
    print(f"   Mean shape: {mc_result['mean'].shape}")
    print(f"   Std shape: {mc_result['std'].shape}")
    print(f"   Samples shape: {mc_result['samples'].shape}")
    print(f"   Mean uncertainty: {mc_result['std'].mean():.4f}")
    
    print("\n2. Testing TTA...")
    tta_predictor = TTAPredictor(model, device='cpu')
    tta_result = tta_predictor.predict_with_tta(test_image)
    print(f"   Mean shape: {tta_result['mean'].shape}")
    print(f"   Std shape: {tta_result['std'].shape}")
    print(f"   Samples shape: {tta_result['samples'].shape}")
    print(f"   Mean uncertainty: {tta_result['std'].mean():.4f}")
    
    print("\n3. Testing Ensemble...")
    ensemble = EnsemblePredictor(model, n_mc_samples=5, use_tta=True, device='cpu')
    ensemble_result = ensemble.predict_with_uncertainty(test_image)
    print(f"   Mean shape: {ensemble_result['mean'].shape}")
    print(f"   Total std: {ensemble_result['std'].mean():.4f}")
    print(f"   Epistemic: {ensemble_result['epistemic'].mean():.4f}")
    print(f"   Aleatoric: {ensemble_result['aleatoric'].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
