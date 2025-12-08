"""
Generate Grad-CAM visualizations for multi-task model.

Shows which regions of the MRI slice the model focuses on for classification.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.multi_task_model import create_multi_task_model
from src.data.multi_source_dataset import MultiSourceDataset


class GradCAM:
    """Grad-CAM implementation for multi-task model."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            x: Input tensor (1, C, H, W)
            class_idx: Target class index (default: predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        # Forward pass
        self.model.eval()
        output = self.model(x, do_seg=False, do_cls=True)
        logits = output['cls']
        
        # Get predicted class if not specified
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def create_overlay(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Create overlay of heatmap on image.
    
    Args:
        image: Original image (H, W) in [0, 1]
        heatmap: Grad-CAM heatmap (H, W) in [0, 1]
        alpha: Overlay transparency
        colormap: OpenCV colormap
    
    Returns:
        overlay: RGB image (H, W, 3)
    """
    # Convert image to RGB
    image_rgb = (image * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        colormap
    )
    
    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def visualize_sample(image, heatmap, pred_class, true_class, pred_prob, save_path):
    """
    Create visualization with original image, heatmap, and overlay.
    
    Args:
        image: Original image (H, W)
        heatmap: Grad-CAM heatmap (H, W)
        pred_class: Predicted class (0 or 1)
        true_class: True class (0 or 1)
        pred_prob: Prediction probability
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MRI Slice')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    overlay = create_overlay(image, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add prediction info
    class_names = ['No Tumor', 'Tumor']
    pred_name = class_names[pred_class]
    true_name = class_names[true_class]
    correct = '[OK]' if pred_class == true_class else '✗'
    
    fig.suptitle(
        f'{correct} Predicted: {pred_name} ({pred_prob:.2%}) | True: {true_name}',
        fontsize=14,
        fontweight='bold',
        color='green' if pred_class == true_class else 'red'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for multi-task model")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/multitask_joint/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--test-brats", type=str, default="data/processed/brats2d/test")
    parser.add_argument("--test-kaggle", type=str, default="data/processed/kaggle/test")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to visualize")
    parser.add_argument("--output-dir", type=str, default="visualizations/multitask_gradcam",
                       help="Output directory")
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Create model
    print("Creating model...")
    model = create_multi_task_model(
        in_channels=model_config.get('in_channels', 1),
        cls_num_classes=model_config.get('num_classes', 2),
        base_filters=model_config.get('base_filters', 32),
        depth=model_config.get('depth', 3),
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("[OK] Model loaded successfully")
    
    # Create Grad-CAM
    # Target the last convolutional layer in the encoder (bottleneck)
    # The bottleneck is the last down block's DoubleConv
    target_layer = model.encoder.down_blocks[-1].maxpool_conv[1].double_conv[-2]  # Last conv before ReLU
    gradcam = GradCAM(model, target_layer)
    print(f"[OK] Grad-CAM initialized on layer: {target_layer.__class__.__name__}")
    
    # Load test dataset
    print("\nLoading test datasets...")
    test_dataset = MultiSourceDataset(
        brats_dir=args.test_brats,
        kaggle_dir=args.test_kaggle,
        transform=None,
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select samples to visualize (stratified by class and correctness)
    print("\nSelecting samples...")
    correct_tumor = []
    correct_no_tumor = []
    incorrect_tumor = []
    incorrect_no_tumor = []
    
    # First pass: classify all samples
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        true_class = sample['cls']
        
        with torch.no_grad():
            output = model(image, do_seg=False, do_cls=True)
            logits = output['cls']
            pred_class = logits.argmax(dim=1).item()
        
        if pred_class == true_class:
            if true_class == 1:
                correct_tumor.append(idx)
            else:
                correct_no_tumor.append(idx)
        else:
            if true_class == 1:
                incorrect_tumor.append(idx)
            else:
                incorrect_no_tumor.append(idx)
    
    # Select balanced samples
    n_per_category = args.num_samples // 4
    selected_indices = (
        correct_tumor[:n_per_category] +
        correct_no_tumor[:n_per_category] +
        incorrect_tumor[:n_per_category] +
        incorrect_no_tumor[:n_per_category]
    )
    
    print(f"\nSelected {len(selected_indices)} samples:")
    print(f"  Correct Tumor: {min(n_per_category, len(correct_tumor))}")
    print(f"  Correct No Tumor: {min(n_per_category, len(correct_no_tumor))}")
    print(f"  Incorrect Tumor: {min(n_per_category, len(incorrect_tumor))}")
    print(f"  Incorrect No Tumor: {min(n_per_category, len(incorrect_no_tumor))}")
    
    # Generate Grad-CAM visualizations
    print("\nGenerating Grad-CAM visualizations...")
    for i, idx in enumerate(tqdm(selected_indices, desc="Grad-CAM")):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        true_class = sample['cls']
        source = sample['source']
        
        # Get prediction
        with torch.no_grad():
            output = model(image, do_seg=False, do_cls=True)
            logits = output['cls']
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        # Generate Grad-CAM
        heatmap = gradcam(image, class_idx=pred_class)
        
        # Get original image for visualization
        image_np = sample['image'].squeeze().cpu().numpy()
        
        # Normalize to [0, 1] for visualization
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        # Save visualization
        save_path = output_dir / f"gradcam_{i:03d}_{source}_pred{pred_class}_true{true_class}.png"
        visualize_sample(image_np, heatmap, pred_class, true_class, pred_prob, save_path)
    
    print(f"\n[OK] Saved {len(selected_indices)} Grad-CAM visualizations to: {output_dir}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("GRAD-CAM GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total visualizations: {len(selected_indices)}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.checkpoint}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
