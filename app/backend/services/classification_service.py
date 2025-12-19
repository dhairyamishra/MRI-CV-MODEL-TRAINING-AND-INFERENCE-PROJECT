"""
Classification service for SliceWise Backend.

This module provides the ClassificationService class that handles all
classification-related business logic including predictions, calibration,
and Grad-CAM visualization.

Extracted from main_v2.py classification endpoints (lines 457-568).
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
import time

from app.backend.services.model_loader import ModelManager
from app.backend.models.responses import ClassificationResponse, BatchResponse
from app.backend.utils.image_processing import preprocess_image_for_classification
from app.backend.utils.visualization import numpy_to_base64_png, create_gradcam_overlay
from src.eval.grad_cam import visualize_single_image


# ============================================================================
# Classification Service
# ============================================================================

class ClassificationService:
    """
    Service class for classification operations.
    
    Handles:
    - Single image classification
    - Batch classification
    - Calibration (temperature scaling)
    - Grad-CAM visualization
    
    Extracted from main_v2.py (lines 457-568).
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize classification service.
        
        Args:
            model_manager: ModelManager instance with loaded models
        """
        self.model_manager = model_manager
    
    # ========================================================================
    # Single Image Classification
    # ========================================================================
    
    async def classify_single(
        self,
        image_array: np.ndarray,
        return_gradcam: bool = False,
        apply_calibration: bool = True
    ) -> ClassificationResponse:
        """
        Classify a single MRI slice.
        
        Extracted from main_v2.py classify_slice() (lines 469-508).
        
        Args:
            image_array: Input image as numpy array
            return_gradcam: Whether to include Grad-CAM visualization
            apply_calibration: Whether to apply temperature scaling
        
        Returns:
            ClassificationResponse with prediction results
            
        Raises:
            ValueError: If no classification model available
        """
        # Use multi-task model if standalone classifier not available
        if not self.model_manager.classifier_loaded and self.model_manager.multitask_loaded:
            # Use multi-task model for classification
            if return_gradcam:
                # Use predict_with_gradcam for Grad-CAM visualization
                result = self.model_manager.multitask.predict_with_gradcam(
                    image_array,
                    use_brain_mask=True
                )
                # Create Grad-CAM overlay from heatmap
                gradcam_base64 = None
                if result.get('gradcam', {}).get('heatmap') is not None:
                    heatmap = result['gradcam']['heatmap']
                    # Normalize image to [0, 1] for overlay creation
                    image_normalized = image_array.astype(np.float32)
                    if image_normalized.max() > 1.0:
                        image_normalized = image_normalized / 255.0
                    # Create overlay
                    overlay = create_gradcam_overlay(image_normalized, heatmap, alpha=0.5)
                    gradcam_base64 = numpy_to_base64_png(overlay)
                
                return ClassificationResponse(
                    predicted_class=result['classification']['predicted_class'],
                    predicted_label=result['classification']['predicted_label'],
                    confidence=result['classification']['confidence'],
                    probabilities=result['classification']['probabilities'],
                    gradcam_overlay=gradcam_base64
                )
            else:
                # Use predict_conditional for regular classification
                result = self.model_manager.multitask.predict_conditional(
                    image_array,
                    return_logits=False
                )
                return ClassificationResponse(
                    predicted_class=result['classification']['predicted_class'],
                    predicted_label=result['classification']['predicted_label'],
                    confidence=result['classification']['confidence'],
                    probabilities=result['classification']['probabilities']
                )
        
        if not self.model_manager.classifier_loaded:
            raise ValueError("No classification model available")
        
        # Preprocess image
        preprocessed = preprocess_image_for_classification(image_array)
        
        # Run prediction
        result = self.model_manager.classifier.predict(
            preprocessed,
            return_probabilities=True
        )
        
        # Apply calibration if available and requested
        calibrated_probs = None
        if apply_calibration and self.model_manager.calibration_loaded:
            calibrated_probs = self._apply_calibration(image_array)
        
        # Generate Grad-CAM if requested
        gradcam_base64 = None
        if return_gradcam:
            gradcam_base64 = self._generate_gradcam(image_array)
        
        return ClassificationResponse(
            predicted_class=result['predicted_class'],
            predicted_label=result['predicted_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            calibrated_probabilities=calibrated_probs,
            gradcam_overlay=gradcam_base64
        )
    
    # ========================================================================
    # Batch Classification
    # ========================================================================
    
    async def classify_batch(
        self,
        images: List[np.ndarray],
        filenames: Optional[List[str]] = None,
        return_gradcam: bool = False
    ) -> BatchResponse:
        """
        Classify multiple MRI slices in a batch.
        
        Extracted from main_v2.py classify_batch() (lines 520-568).
        
        Args:
            images: List of input images as numpy arrays
            filenames: Optional list of filenames for results
            return_gradcam: Whether to include Grad-CAM (not recommended for large batches)
        
        Returns:
            BatchResponse with batch results and summary
            
        Raises:
            ValueError: If classifier not loaded
        """
        if not self.model_manager.classifier_loaded:
            raise ValueError("Classifier not loaded")
        
        start_time = time.time()
        results = []
        tumor_count = 0
        
        for idx, image_array in enumerate(images):
            # Preprocess
            preprocessed = preprocess_image_for_classification(image_array)
            
            # Predict
            result = self.model_manager.classifier.predict(
                preprocessed,
                return_probabilities=True
            )
            
            # Count tumors
            if result['predicted_label'] == 'Tumor':
                tumor_count += 1
            
            # Build result
            result_dict = {
                "filename": filenames[idx] if filenames else f"image_{idx}",
                "predicted_class": result['predicted_class'],
                "predicted_label": result['predicted_label'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities']
            }
            
            results.append(result_dict)
        
        processing_time = time.time() - start_time
        
        return BatchResponse(
            num_images=len(images),
            processing_time_seconds=processing_time,
            results=results,
            summary={
                "tumor_detected": tumor_count,
                "no_tumor": len(images) - tumor_count,
                "avg_time_per_image": processing_time / len(images) if images else 0,
                "tumor_percentage": (tumor_count / len(images) * 100) if images else 0
            }
        )
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _apply_calibration(self, image_array: np.ndarray) -> Dict[str, float]:
        """
        Apply temperature scaling calibration.
        
        Extracted from main_v2.py (lines 478-488).
        
        Args:
            image_array: Input image as numpy array
        
        Returns:
            Dictionary of calibrated probabilities
        """
        if not self.model_manager.calibration_loaded:
            return None
        
        # Preprocess
        preprocessed = preprocess_image_for_classification(image_array)
        tensor = self.model_manager.classifier.preprocess_image(preprocessed)
        
        # Get calibrated logits
        with torch.no_grad():
            logits = self.model_manager.classifier.model(tensor)
            calibrated_logits = self.model_manager.temperature_scaler(logits)
            calibrated_probs_tensor = torch.softmax(calibrated_logits, dim=1)[0]
            
            calibrated_probs = {
                name: float(calibrated_probs_tensor[i])
                for i, name in enumerate(self.model_manager.classifier.class_names)
            }
        
        return calibrated_probs
    
    def _generate_gradcam(self, image_array: np.ndarray) -> str:
        """
        Generate Grad-CAM visualization.
        
        Extracted from main_v2.py (lines 492-499).
        
        Args:
            image_array: Input image as numpy array
        
        Returns:
            Base64 encoded Grad-CAM overlay
        """
        # Preprocess
        preprocessed = preprocess_image_for_classification(image_array)
        tensor = self.model_manager.classifier.preprocess_image(preprocessed)
        
        # Generate Grad-CAM
        cam, overlay = visualize_single_image(
            self.model_manager.classifier.model,
            tensor,
            self.model_manager.classifier.device
        )
        
        # Encode to base64
        return numpy_to_base64_png(overlay)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def is_available(self) -> bool:
        """Check if classification service is available."""
        # Use standalone classifier if available, otherwise fall back to multi-task model
        return self.model_manager.classifier_loaded or self.model_manager.multitask_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get classifier model information."""
        if not self.is_available():
            return {}
        
        return {
            'architecture': self.model_manager.classifier.model_name,
            'num_classes': len(self.model_manager.classifier.class_names),
            'class_names': self.model_manager.classifier.class_names,
            'device': str(self.model_manager.classifier.device),
            'calibrated': self.model_manager.calibration_loaded
        }


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Test classification service
    print("Testing ClassificationService...")
    
    from app.backend.services.model_loader import get_model_manager
    
    # Get model manager
    manager = get_model_manager()
    manager.load_all_models()
    
    # Create service
    service = ClassificationService(manager)
    print(f"\n1. Created ClassificationService")
    print(f"   Available: {service.is_available()}")
    
    if service.is_available():
        # Test with dummy image
        print("\n2. Testing single classification:")
        test_image = np.random.rand(256, 256).astype(np.float32)
        
        import asyncio
        result = asyncio.run(service.classify_single(test_image))
        print(f"   Predicted: {result.predicted_label}")
        print(f"   Confidence: {result.confidence:.4f}")
        
        # Test batch
        print("\n3. Testing batch classification:")
        test_images = [np.random.rand(256, 256).astype(np.float32) for _ in range(3)]
        batch_result = asyncio.run(service.classify_batch(test_images))
        print(f"   Processed: {batch_result.num_images} images")
        print(f"   Time: {batch_result.processing_time_seconds:.4f}s")
    
    print("\n✅ ClassificationService test complete!")
