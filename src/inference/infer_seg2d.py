"""
Inference utility for 2D brain tumor segmentation.

Provides functions for:
- Single slice prediction
- Batch prediction
- Probability map and binary mask generation
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet2d import create_unet


class SegmentationPredictor:
    """
    Predictor class for brain tumor segmentation.
    
    Loads a trained U-Net model and performs inference on MRI slices.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on ('cuda' or 'cpu')
        threshold: Threshold for binarizing probability maps (default: 0.5)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        threshold: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Create model
        self.model = create_unet(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            base_filters=model_config.get('base_filters', 64),
            depth=model_config.get('depth', 4),
            bilinear=model_config.get('use_bilinear', True),
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Threshold: {self.threshold}")
    
    @torch.no_grad()
    def predict_slice(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_prob: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predict segmentation for a single slice.
        
        Args:
            image: Input image (H, W) or (1, H, W) or (B, 1, H, W)
            return_prob: If True, return probability map
        
        Returns:
            Dictionary containing:
                - 'mask': Binary segmentation mask (H, W)
                - 'prob': Probability map (H, W) - if return_prob=True
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Ensure correct shape (B, C, H, W)
        if image.dim() == 2:  # (H, W)
            image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif image.dim() == 3:  # (1, H, W) or (C, H, W)
            image = image.unsqueeze(0)  # (1, C, H, W)
        
        # Move to device
        image = image.to(self.device)
        
        # Forward pass
        logits = self.model(image)
        probs = torch.sigmoid(logits)
        
        # Binarize
        mask = (probs > self.threshold).float()
        
        # Convert to numpy and remove batch/channel dims
        mask_np = mask.squeeze().cpu().numpy()
        
        result = {'mask': mask_np}
        
        if return_prob:
            prob_np = probs.squeeze().cpu().numpy()
            result['prob'] = prob_np
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: Union[np.ndarray, torch.Tensor],
        return_prob: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predict segmentation for a batch of slices.
        
        Args:
            images: Batch of images (B, 1, H, W) or (B, H, W)
            return_prob: If True, return probability maps
        
        Returns:
            Dictionary containing:
                - 'masks': Binary segmentation masks (B, H, W)
                - 'probs': Probability maps (B, H, W) - if return_prob=True
        """
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        # Ensure correct shape (B, C, H, W)
        if images.dim() == 3:  # (B, H, W)
            images = images.unsqueeze(1)  # (B, 1, H, W)
        
        # Move to device
        images = images.to(self.device)
        
        # Forward pass
        logits = self.model(images)
        probs = torch.sigmoid(logits)
        
        # Binarize
        masks = (probs > self.threshold).float()
        
        # Convert to numpy and remove channel dim
        masks_np = masks.squeeze(1).cpu().numpy()
        
        result = {'masks': masks_np}
        
        if return_prob:
            probs_np = probs.squeeze(1).cpu().numpy()
            result['probs'] = probs_np
        
        return result
    
    @torch.no_grad()
    def predict_dataloader(
        self,
        dataloader: DataLoader,
        return_prob: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict segmentation for entire dataset via DataLoader.
        
        Args:
            dataloader: PyTorch DataLoader
            return_prob: If True, return probability maps
        
        Returns:
            Dictionary containing:
                - 'masks': All binary masks (N, H, W)
                - 'probs': All probability maps (N, H, W) - if return_prob=True
        """
        all_masks = []
        all_probs = [] if return_prob else None
        
        for batch in dataloader:
            # Handle both (images, masks) and (images,) formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            # Predict
            result = self.predict_batch(images, return_prob=return_prob)
            
            all_masks.append(result['masks'])
            if return_prob:
                all_probs.append(result['probs'])
        
        # Concatenate all batches
        masks_np = np.concatenate(all_masks, axis=0)
        
        result = {'masks': masks_np}
        
        if return_prob:
            probs_np = np.concatenate(all_probs, axis=0)
            result['probs'] = probs_np
        
        return result


def predict_slice(
    image: Union[np.ndarray, torch.Tensor],
    checkpoint_path: str,
    device: str = 'cuda',
    threshold: float = 0.5,
    return_prob: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to predict a single slice.
    
    Args:
        image: Input image (H, W) or (1, H, W)
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        threshold: Threshold for binarization
        return_prob: If True, return probability map
    
    Returns:
        Dictionary with 'mask' and optionally 'prob'
    
    Example:
        >>> image = np.random.randn(256, 256)
        >>> result = predict_slice(image, 'checkpoints/seg/best_model.pth')
        >>> mask = result['mask']
        >>> prob = result['prob']
    """
    predictor = SegmentationPredictor(checkpoint_path, device, threshold)
    return predictor.predict_slice(image, return_prob)


def predict_batch(
    images: Union[np.ndarray, torch.Tensor],
    checkpoint_path: str,
    device: str = 'cuda',
    threshold: float = 0.5,
    return_prob: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to predict a batch of slices.
    
    Args:
        images: Batch of images (B, H, W) or (B, 1, H, W)
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        threshold: Threshold for binarization
        return_prob: If True, return probability maps
    
    Returns:
        Dictionary with 'masks' and optionally 'probs'
    
    Example:
        >>> images = np.random.randn(10, 256, 256)
        >>> result = predict_batch(images, 'checkpoints/seg/best_model.pth')
        >>> masks = result['masks']  # (10, 256, 256)
    """
    predictor = SegmentationPredictor(checkpoint_path, device, threshold)
    return predictor.predict_batch(images, return_prob)


if __name__ == "__main__":
    # Test the predictor
    print("Testing SegmentationPredictor...")
    
    # Create dummy checkpoint for testing
    import tempfile
    from src.models.unet2d import UNet2D
    
    model = UNet2D(in_channels=1, out_channels=1)
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'model': {
                    'in_channels': 1,
                    'out_channels': 1,
                    'base_filters': 64,
                    'depth': 4,
                    'use_bilinear': True,
                }
            }
        }, checkpoint_path)
    
    # Test single slice prediction
    print("\n1. Testing single slice prediction...")
    image = np.random.randn(256, 256).astype(np.float32)
    
    predictor = SegmentationPredictor(checkpoint_path, device='cpu')
    result = predictor.predict_slice(image)
    
    print(f"   Input shape: {image.shape}")
    print(f"   Mask shape: {result['mask'].shape}")
    print(f"   Prob shape: {result['prob'].shape}")
    print(f"   Mask unique values: {np.unique(result['mask'])}")
    print(f"   Prob range: [{result['prob'].min():.3f}, {result['prob'].max():.3f}]")
    
    # Test batch prediction
    print("\n2. Testing batch prediction...")
    images = np.random.randn(4, 256, 256).astype(np.float32)
    
    result = predictor.predict_batch(images)
    
    print(f"   Input shape: {images.shape}")
    print(f"   Masks shape: {result['masks'].shape}")
    print(f"   Probs shape: {result['probs'].shape}")
    
    # Test convenience functions
    print("\n3. Testing convenience functions...")
    result = predict_slice(image, checkpoint_path, device='cpu')
    print(f"   predict_slice: mask shape = {result['mask'].shape}")
    
    result = predict_batch(images, checkpoint_path, device='cpu')
    print(f"   predict_batch: masks shape = {result['masks'].shape}")
    
    # Cleanup
    Path(checkpoint_path).unlink()
    
    print("\n✓ All tests passed!")
