"""
Multi-task predictor for simultaneous classification and segmentation.

This module provides the MultiTaskPredictor class that handles predictions
using the unified multi-task model (classification + segmentation).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import sys

# Add project root to path for brain mask utilities
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from app.backend.utils.image_processing import (
        compute_brain_mask_from_image,
        apply_brain_mask_to_prediction,
        detect_background_padding
    )
    BRAIN_MASK_AVAILABLE = True
except ImportError:
    BRAIN_MASK_AVAILABLE = False
    print("Warning: Brain mask utilities not available in multi-task predictor")

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
from src.models.model_config import ModelConfig
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
        base_filters: Optional[int] = None,
        depth: Optional[int] = None,
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
        
        print(f"[OK] Multi-task model loaded from {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Classification threshold: {self.classification_threshold}")
        print(f"  Segmentation threshold: {self.segmentation_threshold}")
        print(f"  Model parameters: {self.model.get_num_params()}")
    
    def _load_model(
        self,
        checkpoint_path: str,
        base_filters: Optional[int],
        depth: Optional[int]
    ) -> nn.Module:
        """Load multi-task model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint first to get config
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Try to get config from checkpoint
        model_config = {}
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            model_config = checkpoint['config']['model']
            print(f"[OK] Loaded model config from checkpoint")
        else:
            # Try to load config from checkpoint directory
            try:
                config = ModelConfig.from_checkpoint_dir(checkpoint_path.parent)
                model_config = {
                    'in_channels': config.in_channels,
                    'seg_out_channels': config.seg_out_channels,
                    'cls_num_classes': config.cls_num_classes,
                    'base_filters': config.base_filters,
                    'depth': config.depth,
                    'cls_hidden_dim': getattr(config, 'cls_hidden_dim', 256),
                    'cls_dropout': getattr(config, 'cls_dropout', 0.5),
                }
                print(f"[OK] Loaded model config from: {checkpoint_path.parent / 'model_config.json'}")
            except FileNotFoundError:
                print(f"⚠ No model config found, using defaults")
        
        # Get model parameters (allow override from arguments)
        in_channels = model_config.get('in_channels', 1)
        seg_out_channels = model_config.get('seg_out_channels', 1)
        cls_num_classes = model_config.get('cls_num_classes', 2)
        base_filters = base_filters if base_filters is not None else model_config.get('base_filters', 32)
        depth = depth if depth is not None else model_config.get('depth', 4)
        cls_hidden_dim = model_config.get('cls_hidden_dim', 256)
        cls_dropout = model_config.get('cls_dropout', 0.5)
        
        print(f"  Model architecture: base_filters={base_filters}, depth={depth}, cls_hidden_dim={cls_hidden_dim}")
        
        # Create model
        model = create_multi_task_model(
            in_channels=in_channels,
            seg_out_channels=seg_out_channels,
            cls_num_classes=cls_num_classes,
            base_filters=base_filters,
            depth=depth,
            cls_hidden_dim=cls_hidden_dim,
            cls_dropout=cls_dropout,
        )
        
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
            
            print(f"\n[DEBUG] === Multi-Task Segmentation Debug ===")
            print(f"[DEBUG] Image original range: [{image_original.min():.3f}, {image_original.max():.3f}]")
            print(f"[DEBUG] Model output - prob_map range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
            print(f"[DEBUG] Model output - binary_mask unique: {np.unique(binary_mask)}")
            print(f"[DEBUG] Model output - tumor pixels: {binary_mask.sum()}/{binary_mask.size} ({binary_mask.sum()/binary_mask.size*100:.1f}%)")
            
            # Fix inverted masks for Kaggle data using skull boundary detection
            if detect_background_padding(image_original):
                print(f"[DEBUG] Background padding detected - applying skull boundary detection...")
                
                # Detect skull boundary using circular Hough transform
                skull_mask = self._detect_skull_boundary(image_original)
                
                if skull_mask is not None:
                    print(f"[DEBUG] Skull boundary detected successfully")
                    
                    # Check if model output is inverted by comparing with skull mask
                    # The skull mask defines the brain region (inside = 1, outside = 0)
                    background_region = (skull_mask == 0)
                    brain_region = (skull_mask == 1)
                    
                    if background_region.sum() > 0 and brain_region.sum() > 0:
                        avg_prob_background = prob_map[background_region].mean()
                        avg_prob_brain = prob_map[brain_region].mean()
                        avg_mask_background = binary_mask[background_region].mean()
                        avg_mask_brain = binary_mask[brain_region].mean()
                        
                        print(f"[DEBUG] Model predictions:")
                        print(f"[DEBUG]   - Avg prob in background: {avg_prob_background:.3f}")
                        print(f"[DEBUG]   - Avg prob in brain: {avg_prob_brain:.3f}")
                        print(f"[DEBUG]   - Avg mask in background: {avg_mask_background:.3f}")
                        print(f"[DEBUG]   - Avg mask in brain: {avg_mask_brain:.3f}")
                        
                        # Check if binary mask is inverted
                        # If more 1s in background than brain, it's inverted
                        if avg_mask_background > avg_mask_brain:
                            print(f"[INFO] ⚠️  Detected inverted mask - INVERTING output")
                            # Invert the predictions
                            prob_map = 1.0 - prob_map
                            binary_mask = 1.0 - binary_mask
                            print(f"[INFO] ✅ Mask inverted successfully")
                            print(f"[DEBUG] After inversion:")
                            print(f"[DEBUG]   - Avg mask in background: {(1.0 - avg_mask_background):.3f}")
                            print(f"[DEBUG]   - Avg mask in brain: {(1.0 - avg_mask_brain):.3f}")
                        else:
                            print(f"[DEBUG] Mask orientation looks correct")
                    
                    # Apply skull mask to zero out background
                    if BRAIN_MASK_AVAILABLE:
                        prob_map = apply_brain_mask_to_prediction(prob_map, skull_mask)
                        binary_mask = apply_brain_mask_to_prediction(binary_mask, skull_mask)
                    else:
                        prob_map = prob_map * skull_mask
                        binary_mask = binary_mask * skull_mask
                    print(f"[INFO] Skull boundary mask applied")
                else:
                    print(f"[DEBUG] Skull boundary detection failed - using fallback")
                    # Fallback to simple thresholding
                    image_u8 = (image_original * 255).astype(np.uint8)
                    simple_mask = (image_u8 > 30).astype(np.uint8)
                    
                    # Check for inversion
                    background_region = (simple_mask == 0)
                    brain_region = (simple_mask == 1)
                    if background_region.sum() > 0 and brain_region.sum() > 0:
                        avg_mask_background = binary_mask[background_region].mean()
                        avg_mask_brain = binary_mask[brain_region].mean()
                        if avg_mask_background > avg_mask_brain:
                            prob_map = 1.0 - prob_map
                            binary_mask = 1.0 - binary_mask
                            print(f"[INFO] Mask inverted (fallback)")
                    
                    if BRAIN_MASK_AVAILABLE:
                        prob_map = apply_brain_mask_to_prediction(prob_map, simple_mask)
                        binary_mask = apply_brain_mask_to_prediction(binary_mask, simple_mask)
                    else:
                        prob_map = prob_map * simple_mask
                        binary_mask = binary_mask * simple_mask
                    print(f"[INFO] Simple threshold mask applied (fallback)")
            else:
                print(f"[DEBUG] No background padding detected - skipping brain mask")
            
            print(f"[DEBUG] === End Debug ===\n")
            
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
        target_class: Optional[int] = None,
        use_brain_mask: bool = True
    ) -> Dict[str, any]:
        """
        Run prediction with Grad-CAM visualization.
        
        Args:
            image: Input image as numpy array
            target_class: Target class for Grad-CAM (None = predicted class)
            use_brain_mask: Whether to apply brain masking to eliminate background artifacts
        
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
        
        # Detect brain mask to eliminate background artifacts
        brain_mask = None
        if use_brain_mask:
            print("[INFO] 🧠 Detecting brain boundary for Grad-CAM masking...")
            brain_mask = self._detect_skull_boundary(image_original)
            
            if brain_mask is not None:
                print(f"[INFO] ✅ Brain mask detected (coverage: {brain_mask.sum()/brain_mask.size*100:.1f}%)")
                
                # Apply brain mask to input tensor BEFORE Grad-CAM computation
                # This zeros out background gradients
                brain_mask_tensor = torch.from_numpy(brain_mask).float().to(self.device)
                brain_mask_tensor = brain_mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                tensor_masked = tensor * brain_mask_tensor
                
                print("[INFO] 🎯 Applying brain mask to input (zeros out background gradients)")
            else:
                print("[INFO] ⚠️  Brain mask detection failed, using full image")
                tensor_masked = tensor
        else:
            tensor_masked = tensor
        
        # Generate Grad-CAM on masked input
        self.model.train()  # Need gradients
        cam = self._gradcam.generate_cam(tensor_masked, target_class)
        self.model.eval()
        
        # Apply brain mask to CAM output for clean visualization
        if use_brain_mask and brain_mask is not None:
            print("[INFO] 🎨 Masking Grad-CAM heatmap (removes background artifacts)")
            # Resize brain mask to match CAM dimensions (CAM is lower resolution from bottleneck)
            cam_h, cam_w = cam.shape
            if brain_mask.shape != cam.shape:
                brain_mask_resized = cv2.resize(brain_mask, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
                print(f"[DEBUG] Resized brain mask from {brain_mask.shape} to {brain_mask_resized.shape} to match CAM")
            else:
                brain_mask_resized = brain_mask
            cam = cam * brain_mask_resized
        
        # Run normal prediction
        result = self.predict_single(image, do_seg=True, do_cls=True)
        
        # Add Grad-CAM heatmap and metadata
        result['gradcam'] = {
            'heatmap': cam,
            'target_class': target_class if target_class is not None else result['classification']['predicted_class'],
            'brain_masked': brain_mask is not None if use_brain_mask else False
        }
        
        return result
    
    def predict_full(
        self,
        image: np.ndarray,
        include_gradcam: bool = True,
        use_brain_mask_for_gradcam: bool = True
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
            use_brain_mask_for_gradcam: Whether to apply brain masking to Grad-CAM
        
        Returns:
            result: Comprehensive prediction dictionary
        """
        # Run conditional prediction
        result = self.predict_conditional(image)
        
        # Add Grad-CAM if requested
        if include_gradcam:
            gradcam_result = self.predict_with_gradcam(image, use_brain_mask=use_brain_mask_for_gradcam)
            result['gradcam'] = gradcam_result['gradcam']
        
        return result
    
    def _detect_skull_boundary(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the skull boundary using circular contour detection.
        
        This finds the largest circular/elliptical contour in the image,
        which typically corresponds to the skull boundary in MRI scans.
        
        Args:
            image: Input image normalized to [0, 1]
        
        Returns:
            Binary mask with 1 inside skull, 0 outside, or None if detection fails
        """
        try:
            # Convert to uint8
            image_u8 = (image * 255).astype(np.uint8)
            
            # Apply threshold to get binary image
            # Use a low threshold to capture the brain region
            _, binary = cv2.threshold(image_u8, 30, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour (should be the skull)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if contour is large enough (at least 20% of image)
            contour_area = cv2.contourArea(largest_contour)
            image_area = image.shape[0] * image.shape[1]
            if contour_area < 0.2 * image_area:
                print(f"[DEBUG] Largest contour too small: {contour_area/image_area*100:.1f}% of image")
                return None
            
            # Create mask from contour
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
            
            # Apply additional morphological closing to ensure solid interior
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
            
            # Fill any remaining holes
            h, w = mask.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            mask_inv = 1 - mask
            cv2.floodFill(mask_inv, flood_mask, (0, 0), 1)
            mask = 1 - mask_inv
            
            coverage = mask.sum() / mask.size
            print(f"[DEBUG] Skull mask coverage: {coverage*100:.1f}%")
            
            # Sanity check: mask should cover 30-90% of image
            if coverage < 0.3 or coverage > 0.9:
                print(f"[DEBUG] Skull mask coverage out of range: {coverage*100:.1f}%")
                return None
            
            return mask
            
        except Exception as e:
            print(f"[DEBUG] Skull boundary detection failed: {e}")
            return None
    
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
    base_filters: Optional[int] = None,
    depth: Optional[int] = None,
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
