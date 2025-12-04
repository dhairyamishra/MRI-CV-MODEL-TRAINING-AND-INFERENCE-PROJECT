"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

This module provides Grad-CAM visualization for explaining classifier predictions.
Grad-CAM highlights the regions of the input image that are most important for
the model's decision.

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (ICCV 2017)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional, List
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import create_classifier
from src.data.kaggle_mri_dataset import create_dataloaders
from src.data.transforms import get_val_transforms


class GradCAM:
    """
    Grad-CAM implementation for CNNs.
    
    Generates heatmaps showing which regions of the input image
    contribute most to the model's prediction.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model
            target_layer: The target convolutional layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input tensor of shape (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
        
        Returns:
            cam: Grad-CAM heatmap of shape (H, W) with values in [0, 1]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)[0].item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_overlay(
        self,
        input_image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Generate overlay of Grad-CAM heatmap on input image.
        
        Args:
            input_image: Original image (H, W) or (H, W, 1)
            cam: Grad-CAM heatmap (H, W)
            alpha: Blending factor for overlay
            colormap: OpenCV colormap for heatmap
        
        Returns:
            overlay: RGB overlay image (H, W, 3)
        """
        # Ensure input_image is 2D
        if input_image.ndim == 3:
            input_image = input_image.squeeze()
        
        # Resize CAM to match input image size
        h, w = input_image.shape
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert CAM to heatmap
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize input image to [0, 255]
        img_normalized = ((input_image - input_image.min()) / 
                         (input_image.max() - input_image.min() + 1e-8) * 255)
        img_uint8 = np.uint8(img_normalized)
        
        # Convert grayscale to RGB
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        
        # Blend images
        overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


def generate_gradcam_visualizations(
    config_path: str,
    checkpoint_path: str,
    num_samples: int = 16,
    save_dir: Optional[str] = None
):
    """
    Generate Grad-CAM visualizations for sample images.
    
    Args:
        config_path: Path to configuration YAML file
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    
    Example:
        >>> generate_gradcam_visualizations(
        ...     'configs/config_cls.yaml',
        ...     'checkpoints/cls/best_model.pth',
        ...     num_samples=16
        ... )
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    if save_dir is None:
        save_dir = config['gradcam']['save_dir']
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = create_classifier(
        model_name=config['model']['name'],
        pretrained=False,
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Get target layer for Grad-CAM
    target_layer = model.get_cam_target_layer()
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Create data loader
    _, _, test_loader = create_dataloaders(
        batch_size=1,
        num_workers=0,
        val_transform=get_val_transforms()
    )
    
    # Generate visualizations
    print(f"\nGenerating Grad-CAM visualizations for {num_samples} samples...")
    
    samples_processed = 0
    correct_samples = 0
    incorrect_samples = 0
    
    for images, labels, ids in tqdm(test_loader):
        if samples_processed >= num_samples:
            break
        
        images = images.to(device)
        label = labels.item()
        
        # Get prediction
        with torch.no_grad():
            output = model(images)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)[0].item()
            confidence = probs[0, pred].item()
        
        # Generate Grad-CAM
        cam = grad_cam.generate_cam(images, target_class=pred)
        
        # Get original image
        original_image = images[0, 0].cpu().numpy()
        
        # Generate overlay
        overlay = grad_cam.generate_overlay(original_image, cam, alpha=0.5)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')
        
        # Add prediction info
        class_names = ['No Tumor', 'Tumor']
        pred_label = class_names[pred]
        true_label = class_names[label]
        is_correct = pred == label
        
        title_color = 'green' if is_correct else 'red'
        fig.suptitle(
            f'True: {true_label} | Predicted: {pred_label} (conf: {confidence:.3f})',
            fontsize=14,
            color=title_color,
            fontweight='bold'
        )
        
        # Save figure
        status = 'correct' if is_correct else 'incorrect'
        if is_correct:
            filename = f'gradcam_correct_{correct_samples:03d}_{ids[0]}.png'
            correct_samples += 1
        else:
            filename = f'gradcam_incorrect_{incorrect_samples:03d}_{ids[0]}.png'
            incorrect_samples += 1
        
        plt.tight_layout()
        plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        samples_processed += 1
    
    print(f"\n✓ Generated {samples_processed} Grad-CAM visualizations")
    print(f"  - Correct predictions: {correct_samples}")
    print(f"  - Incorrect predictions: {incorrect_samples}")
    print(f"  - Saved to: {save_dir}")


def visualize_single_image(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    save_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM for a single image.
    
    Args:
        model: Trained model
        image: Input image tensor (1, 1, H, W)
        device: Compute device
        save_path: Optional path to save visualization
    
    Returns:
        cam: Grad-CAM heatmap
        overlay: Overlay image
    """
    model.eval()
    image = image.to(device)
    
    # Get target layer
    target_layer = model.get_cam_target_layer()
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam = grad_cam.generate_cam(image)
    
    # Get original image
    original_image = image[0, 0].cpu().numpy()
    
    # Generate overlay
    overlay = grad_cam.generate_overlay(original_image, cam)
    
    # Optionally save
    if save_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return cam, overlay


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
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
    parser.add_argument(
        '--num_samples',
        type=int,
        default=16,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    generate_gradcam_visualizations(
        args.config,
        args.checkpoint,
        args.num_samples,
        args.save_dir
    )
