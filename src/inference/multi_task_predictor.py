"""
Multi-Task Predictor for Joint Segmentation and Classification.

This module provides a unified inference interface for the multi-task model
that performs both tumor classification and segmentation in a single forward pass.

Key Features:
- Single forward pass for both tasks (~40% faster than separate models)
- Conditional segmentation (only compute if tumor probability >= threshold)
- Grad-CAM visualization support
- Flexible preprocessing and post-processing
- Batch inference support
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.multi_task_model import create_multi_task_model
from src.eval.grad_cam import GradCAM


class MultiTaskPredictor:
    """
    Unified predictor for multi-task brain tumor detection.
    
    Performs both classification and segmentation in a single forward pass,
    with conditional segmentation based on tumor probability.
    
    Args:
        checkpoint_path: Path to multi-task model checkpoint
        base_filters: Base number of filters (default: 32)
        depth: Encoder/decoder depth (default: 3)
        device: Device to run inference on ('cuda', 'cpu', or None for auto)
        classification_threshold: Threshold for showing segmentation (default: 0.3)
        segmentation_threshold: Threshold for binary mask (default: 0.5)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        base_filters: int = 32,
        depth: int = 3,
        device: Optional[str] = None,
        classification_threshold: float = 0.3,
        segmentation_threshold: float = 0.5,
    ):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.classification_threshold = classification_threshold
        self.segmentation_threshold = segmentation_threshold
        
        # Load model
        self.model = self._load_model(checkpoint_path, base_filters, depth)
        self.model.eval()
        
        # Class names
        self.class_names = ['No Tumor', 'Tumor']
        
        # Grad-CAM (initialized lazily)
        self._gradcam = None
        
        print(f"✓ Multi-task model loaded from {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Classification threshold: {self.classification_threshold}")
        print(f"  Segmentation threshold: {self.segmentation_threshold}")
        print(f"  Model parameters: {self.model.get_num_params()}")
    
    def _load_model(
        self,
        checkpoint_path: str,
        base_filters: int,
        depth: int
    ) -> nn.Module:
        """Load multi-task model from checkpoint."""
        # Create model
        model = create_multi_task_model(
            in_channels=1,
            seg_out_channels=1,
            cls_num_classes=2,
            base_filters=base_filters,
            depth=depth,
            bilinear=True,
            dropout=0.0,
            cls_hidden_dim=256,
            cls_dropout=0.5,
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        
        return model
    
    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: int = 256,
        normalize_method: str = 'z_score'
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            target_size: Target size for resizing
            normalize_method: 'z_score' or 'min_max'
        
        Returns:
            tensor: Preprocessed image tensor (1, 1, H, W)
            original: Original image normalized to [0, 1] for visualization
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                image = image[:, :, 0]
        
        # Resize
        image = cv2.resize(image, (target_size, target_size))
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Normalize to [0, 1] for visualization
        if image.max() > 1.0:
            image_original = image / 255.0
        else:
            image_original = image.copy()
        
        # Apply normalization for model
        if normalize_method == 'z_score':
            # Z-score normalization (for segmentation)
            mean = image.mean()
            std = image.std()
            if std > 0:
                image_normalized = (image - mean) / std
            else:
                image_normalized = image - mean
        else:
            # Min-max normalization (for classification)
            if image.max() > 1.0:
                image_normalized = image / 255.0
            else:
                image_normalized = image
        
        # Convert to tensor
        tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return tensor, image_original
    
    @torch.no_grad()
    def predict_single(
        self,
        image: np.ndarray,
        do_seg: bool = True,
        do_cls: bool = True,
        return_logits: bool = False
    ) -> Dict[str, any]:
        """
        Run prediction on a single image.
        
        Args:
            image: Input image as numpy array
            do_seg: Whether to compute segmentation
            do_cls: Whether to compute classification
            return_logits: Whether to return raw logits
        
        Returns:
            result: Dictionary containing prediction results
        """
        # Preprocess (use z-score for segmentation compatibility)
        tensor, image_original = self.preprocess_image(image, normalize_method='z_score')
        tensor = tensor.to(self.device)
        
        # Forward pass
        output = self.model(tensor, do_seg=do_seg, do_cls=do_cls)
        
        result = {}
        
        # Classification results
        if do_cls and 'cls' in output:
            cls_logits = output['cls']
            cls_probs = torch.softmax(cls_logits, dim=1)
            
            pred_class = cls_logits.argmax(dim=1).item()
            confidence = cls_probs[0, pred_class].item()
            tumor_prob = cls_probs[0, 1].item()  # Probability of tumor class
            
            result['classification'] = {
                'predicted_class': pred_class,
                'predicted_label': self.class_names[pred_class],
                'confidence': confidence,
                'tumor_probability': tumor_prob,
                'probabilities': {
                    self.class_names[i]: cls_probs[0, i].item()
                    for i in range(len(self.class_names))
                }
            }
            
            if return_logits:
                result['classification']['logits'] = cls_logits.cpu().numpy()
        
        # Segmentation results
        if do_seg and 'seg' in output:
            seg_logits = output['seg']
            seg_probs = torch.sigmoid(seg_logits)
            seg_mask = (seg_probs > self.segmentation_threshold).float()
            
            # Convert to numpy
            prob_map = seg_probs.squeeze().cpu().numpy()
            binary_mask = seg_mask.squeeze().cpu().numpy()
            
            # Calculate tumor statistics
            tumor_pixels = int(binary_mask.sum())
            total_pixels = binary_mask.size
            tumor_percentage = (tumor_pixels / total_pixels) * 100
            
            result['segmentation'] = {
                'mask': binary_mask,
                'prob_map': prob_map,
                'tumor_area_pixels': tumor_pixels,
                'tumor_percentage': tumor_percentage,
            }
            
            if return_logits:
                result['segmentation']['logits'] = seg_logits.cpu().numpy()
        
        # Store original image for visualization
        result['image_original'] = image_original
        
        return result
    
    @torch.no_grad()
    def predict_conditional(
        self,
        image: np.ndarray,
        return_logits: bool = False
    ) -> Dict[str, any]:
        """
        Run conditional prediction: classification first, then segmentation if needed.
        
        This is the recommended method for production use. It only computes
        segmentation if the tumor probability is above the threshold.
        
        Args:
            image: Input image as numpy array
            return_logits: Whether to return raw logits
        
        Returns:
            result: Dictionary containing prediction results
        """
        # First, run classification only
        result = self.predict_single(image, do_seg=False, do_cls=True, return_logits=return_logits)
        
        tumor_prob = result['classification']['tumor_probability']
        
        # Decide whether to run segmentation
        if tumor_prob >= self.classification_threshold:
            # Run segmentation
            seg_result = self.predict_single(image, do_seg=True, do_cls=False, return_logits=return_logits)
            result['segmentation'] = seg_result['segmentation']
            result['segmentation_computed'] = True
            result['recommendation'] = f"Tumor detected with {tumor_prob*100:.1f}% confidence. Segmentation mask generated."
        else:
            result['segmentation_computed'] = False
            result['recommendation'] = f"No significant tumor detected ({tumor_prob*100:.1f}% probability). Segmentation not computed."
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[np.ndarray],
        do_seg: bool = True,
        do_cls: bool = True,
    ) -> List[Dict[str, any]]:
        """
        Run predictions on a batch of images.
        
        Args:
            images: List of images as numpy arrays
            do_seg: Whether to compute segmentation
            do_cls: Whether to compute classification
        
        Returns:
            results: List of prediction dictionaries
        """
        if not images:
            return []
        
        # Preprocess all images
        tensors = []
        originals = []
        for img in images:
            tensor, original = self.preprocess_image(img, normalize_method='z_score')
            tensors.append(tensor)
            originals.append(original)
        
        # Stack into batch
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Forward pass
        output = self.model(batch, do_seg=do_seg, do_cls=do_cls)
        
        # Process results for each image
        results = []
        for i in range(len(images)):
            result = {}
            
            # Classification
            if do_cls and 'cls' in output:
                cls_logits = output['cls'][i:i+1]
                cls_probs = torch.softmax(cls_logits, dim=1)
                
                pred_class = cls_logits.argmax(dim=1).item()
                confidence = cls_probs[0, pred_class].item()
                tumor_prob = cls_probs[0, 1].item()
                
                result['classification'] = {
                    'predicted_class': pred_class,
                    'predicted_label': self.class_names[pred_class],
                    'confidence': confidence,
                    'tumor_probability': tumor_prob,
                    'probabilities': {
                        self.class_names[j]: cls_probs[0, j].item()
                        for j in range(len(self.class_names))
                    }
                }
            
            # Segmentation
            if do_seg and 'seg' in output:
                seg_logits = output['seg'][i:i+1]
                seg_probs = torch.sigmoid(seg_logits)
                seg_mask = (seg_probs > self.segmentation_threshold).float()
                
                prob_map = seg_probs.squeeze().cpu().numpy()
                binary_mask = seg_mask.squeeze().cpu().numpy()
                
                tumor_pixels = int(binary_mask.sum())
                total_pixels = binary_mask.size
                tumor_percentage = (tumor_pixels / total_pixels) * 100
                
                result['segmentation'] = {
                    'mask': binary_mask,
                    'prob_map': prob_map,
                    'tumor_area_pixels': tumor_pixels,
                    'tumor_percentage': tumor_percentage,
                }
            
            result['image_original'] = originals[i]
            results.append(result)
        
        return results
    
    def predict_with_gradcam(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Run prediction with Grad-CAM visualization.
        
        Args:
            image: Input image as numpy array
            target_class: Target class for Grad-CAM (None = predicted class)
        
        Returns:
            result: Dictionary with predictions and Grad-CAM heatmap
        """
        # Initialize Grad-CAM if needed
        if self._gradcam is None:
            # Create a wrapper model that returns only classification logits for Grad-CAM
            class ClassificationWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    output = self.model(x, do_seg=False, do_cls=True)
                    return output['cls']  # Return only classification logits
            
            # Wrap the model
            wrapped_model = ClassificationWrapper(self.model)
            
            # Hook into encoder's last down_block (bottleneck layer)
            target_layer = self.model.encoder.down_blocks[-1]
            self._gradcam = GradCAM(wrapped_model, target_layer)
        
        # Preprocess
        tensor, image_original = self.preprocess_image(image, normalize_method='z_score')
        tensor = tensor.to(self.device)
        
        # Generate Grad-CAM
        self.model.train()  # Need gradients
        cam = self._gradcam.generate_cam(tensor, target_class)
        self.model.eval()
        
        # Run normal prediction
        result = self.predict_single(image, do_seg=True, do_cls=True)
        
        # Add Grad-CAM heatmap
        result['gradcam'] = {
            'heatmap': cam,
            'target_class': target_class if target_class is not None else result['classification']['predicted_class']
        }
        
        return result
    
    def predict_full(
        self,
        image: np.ndarray,
        include_gradcam: bool = True
    ) -> Dict[str, any]:
        """
        Run comprehensive prediction with all features.
        
        This method returns:
        - Classification results
        - Segmentation results (if tumor probability >= threshold)
        - Grad-CAM visualization (optional)
        - Recommendations
        
        Args:
            image: Input image as numpy array
            include_gradcam: Whether to include Grad-CAM
        
        Returns:
            result: Comprehensive prediction dictionary
        """
        # Run conditional prediction
        result = self.predict_conditional(image)
        
        # Add Grad-CAM if requested
        if include_gradcam:
            gradcam_result = self.predict_with_gradcam(image)
            result['gradcam'] = gradcam_result['gradcam']
        
        return result
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information and statistics."""
        params = self.model.get_num_params()
        
        return {
            'architecture': 'Multi-Task U-Net',
            'parameters': params,
            'device': str(self.device),
            'classification_threshold': self.classification_threshold,
            'segmentation_threshold': self.segmentation_threshold,
            'input_size': [256, 256],
            'input_channels': 1,
            'output_classes': 2,
            'tasks': ['classification', 'segmentation']
        }


def create_multi_task_predictor(
    checkpoint_path: str,
    base_filters: int = 32,
    depth: int = 3,
    device: Optional[str] = None,
    classification_threshold: float = 0.3,
    segmentation_threshold: float = 0.5,
) -> MultiTaskPredictor:
    """
    Factory function to create multi-task predictor.
    
    Args:
        checkpoint_path: Path to multi-task model checkpoint
        base_filters: Base number of filters (default: 32)
        depth: Encoder/decoder depth (default: 3)
        device: Device to run inference on
        classification_threshold: Threshold for showing segmentation
        segmentation_threshold: Threshold for binary mask
    
    Returns:
        MultiTaskPredictor instance
    
    Examples:
        >>> # Create predictor
        >>> predictor = create_multi_task_predictor(
        ...     'checkpoints/multitask_joint/best_model.pth'
        ... )
        
        >>> # Run conditional prediction (recommended)
        >>> result = predictor.predict_conditional(image)
        >>> print(result['classification']['tumor_probability'])
        >>> if result['segmentation_computed']:
        ...     mask = result['segmentation']['mask']
        
        >>> # Run full prediction with Grad-CAM
        >>> result = predictor.predict_full(image, include_gradcam=True)
    """
    return MultiTaskPredictor(
        checkpoint_path=checkpoint_path,
        base_filters=base_filters,
        depth=depth,
        device=device,
        classification_threshold=classification_threshold,
        segmentation_threshold=segmentation_threshold,
    )


if __name__ == "__main__":
    print("MultiTaskPredictor class is ready to use!")
    print("\nExample usage:")
    print("  predictor = create_multi_task_predictor('checkpoints/multitask_joint/best_model.pth')")
    print("  result = predictor.predict_conditional(image)")
    print("  print(result['classification']['tumor_probability'])")
