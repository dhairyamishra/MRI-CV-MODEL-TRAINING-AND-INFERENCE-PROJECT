"""
Inference utilities for brain tumor classification.

This module provides a simple interface for running predictions
on new MRI images.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import create_classifier


class ClassifierPredictor:
    """
    Predictor class for brain tumor classification.
    
    Provides a simple interface for loading a trained model
    and running predictions on new images.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "efficientnet",
        device: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_name: Model architecture name
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(checkpoint_path, model_name)
        self.model.eval()
        
        # Class names
        self.class_names = ['No Tumor', 'Tumor']
    
    def _load_model(self, checkpoint_path: str, model_name: str) -> nn.Module:
        """Load model from checkpoint."""
        # Create model
        model = create_classifier(
            model_name=model_name,
            pretrained=False,
            num_classes=2,
            dropout=0.3
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 256) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            target_size: Target size for resizing
        
        Returns:
            tensor: Preprocessed image tensor (1, 1, H, W)
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
        
        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return tensor
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_probabilities: bool = True
    ) -> Dict[str, any]:
        """
        Run prediction on an image.
        
        Args:
            image: Input image as numpy array
            return_probabilities: Whether to return class probabilities
        
        Returns:
            result: Dictionary containing prediction results
        """
        # Preprocess
        tensor = self.preprocess_image(image)
        tensor = tensor.to(self.device)
        
        # Forward pass
        output = self.model(tensor)
        
        # Get probabilities
        probs = torch.softmax(output, dim=1)
        
        # Get prediction
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        
        # Prepare result
        result = {
            'predicted_class': pred_class,
            'predicted_label': self.class_names[pred_class],
            'confidence': confidence,
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.class_names[i]: probs[0, i].item()
                for i in range(len(self.class_names))
            }
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: list,
        return_probabilities: bool = True
    ) -> list:
        """
        Run predictions on a batch of images.
        
        Args:
            images: List of images as numpy arrays
            return_probabilities: Whether to return class probabilities
        
        Returns:
            results: List of prediction dictionaries
        """
        # Handle empty batch
        if not images:
            return []
        
        # Preprocess all images
        tensors = [self.preprocess_image(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Forward pass
        outputs = self.model(batch)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        # Prepare results
        results = []
        for i in range(len(images)):
            pred_class = preds[i].item()
            confidence = probs[i, pred_class].item()
            
            result = {
                'predicted_class': pred_class,
                'predicted_label': self.class_names[pred_class],
                'confidence': confidence,
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    self.class_names[j]: probs[i, j].item()
                    for j in range(len(self.class_names))
                }
            
            results.append(result)
        
        return results
    
    def predict_from_path(
        self,
        image_path: str,
        return_probabilities: bool = True
    ) -> Dict[str, any]:
        """
        Run prediction on an image file.
        
        Args:
            image_path: Path to image file
            return_probabilities: Whether to return class probabilities
        
        Returns:
            result: Dictionary containing prediction results
        """
        # Load image
        image = np.array(Image.open(image_path))
        
        # Run prediction
        return self.predict(image, return_probabilities)


if __name__ == "__main__":
    # Test the predictor
    print("Testing ClassifierPredictor...")
    
    # Create a dummy image
    dummy_image = np.random.rand(256, 256).astype(np.float32)
    
    # Note: You need a trained model checkpoint to actually run this
    # predictor = ClassifierPredictor('checkpoints/cls/best_model.pth')
    # result = predictor.predict(dummy_image)
    # print(result)
    
    print("ClassifierPredictor class is ready to use!")
