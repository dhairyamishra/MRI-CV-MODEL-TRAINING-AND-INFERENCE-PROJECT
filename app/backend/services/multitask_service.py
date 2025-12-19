"""
Multi-task service for SliceWise Backend.

This module provides the MultiTaskService class that handles multi-task
predictions (classification + conditional segmentation) using the unified
multi-task model.

Extracted from main_v2.py predict_multitask endpoint (lines 874-971).
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
import time
import cv2
import matplotlib.pyplot as plt

from app.backend.services.model_loader import ModelManager
from app.backend.models.responses import MultiTaskResponse
from app.backend.utils.visualization import numpy_to_base64_png, create_overlay


# ============================================================================
# Multi-Task Service
# ============================================================================

class MultiTaskService:
    """
    Service class for multi-task predictions.
    
    Handles:
    - Conditional prediction (segmentation only if tumor detected)
    - Full prediction (always compute both tasks)
    - Grad-CAM visualization
    - Clinical recommendations
    
    Extracted from main_v2.py (lines 874-971).
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize multi-task service.
        
        Args:
            model_manager: ModelManager instance with loaded models
        """
        self.model_manager = model_manager
    
    # ========================================================================
    # Prediction Methods
    # ========================================================================
    
    async def predict_conditional(
        self,
        image_array: np.ndarray,
        include_gradcam: bool = True
    ) -> MultiTaskResponse:
        """
        Conditional multi-task prediction.
        
        Segmentation is only computed if tumor probability exceeds threshold.
        This is the recommended approach for efficiency.
        
        Extracted from main_v2.py (line 907).
        
        Args:
            image_array: Input image as numpy array
            include_gradcam: Whether to include Grad-CAM visualization
        
        Returns:
            MultiTaskResponse with prediction results
            
        Raises:
            ValueError: If multi-task model not loaded
        """
        if not self.model_manager.multitask_loaded:
            raise ValueError("Multi-task model not loaded")
        
        start_time = time.time()
        
        # Run conditional prediction
        if include_gradcam:
            result = self.model_manager.multitask.predict_full(
                image_array,
                include_gradcam=True
            )
        else:
            result = self.model_manager.multitask.predict_conditional(image_array)
        
        # Build response
        response_dict = self._create_response(result, include_gradcam)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        response_dict["processing_time_ms"] = round(processing_time, 2)
        
        return MultiTaskResponse(**response_dict)
    
    async def predict_full(
        self,
        image_array: np.ndarray,
        include_gradcam: bool = True
    ) -> MultiTaskResponse:
        """
        Full multi-task prediction.
        
        Always computes both classification and segmentation.
        
        Extracted from main_v2.py (line 905).
        
        Args:
            image_array: Input image as numpy array
            include_gradcam: Whether to include Grad-CAM visualization
        
        Returns:
            MultiTaskResponse with prediction results
            
        Raises:
            ValueError: If multi-task model not loaded
        """
        if not self.model_manager.multitask_loaded:
            raise ValueError("Multi-task model not loaded")
        
        start_time = time.time()
        
        # Run full prediction
        result = self.model_manager.multitask.predict_full(
            image_array,
            include_gradcam=include_gradcam
        )
        
        # Build response
        response_dict = self._create_response(result, include_gradcam)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        response_dict["processing_time_ms"] = round(processing_time, 2)
        
        return MultiTaskResponse(**response_dict)
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _create_response(
        self,
        result: Dict[str, Any],
        include_gradcam: bool
    ) -> Dict[str, Any]:
        """
        Create response dictionary from prediction result.
        
        Extracted from main_v2.py (lines 909-962).
        
        Args:
            result: Prediction result from multi-task model
            include_gradcam: Whether Grad-CAM was requested
        
        Returns:
            Dictionary for MultiTaskResponse
        """
        response = {
            "classification": result['classification'],
            "segmentation_computed": result['segmentation_computed'],
            "recommendation": result['recommendation']
        }
        
        # Add segmentation if computed
        if result['segmentation_computed']:
            mask = result['segmentation']['mask']
            prob_map = result['segmentation']['prob_map']
            
            print(f"[DEBUG] Multitask service - preparing response:")
            print(f"[DEBUG]   - mask unique: {np.unique(mask)}")
            print(f"[DEBUG]   - mask shape: {mask.shape}")
            print(f"[DEBUG]   - mask sum: {mask.sum()}")
            
            # Create overlay
            image_original = result['image_original']
            overlay = create_overlay(image_original, mask, alpha=0.4)
            
            # Scale mask to 0-255 for display
            mask_display = (mask * 255).astype(np.uint8)
            print(f"[DEBUG]   - mask_display unique: {np.unique(mask_display)}")
            print(f"[DEBUG]   - mask_display[100,100]: {mask_display[100, 100]}")
            
            # Debug probability map
            print(f"[DEBUG]   - prob_map shape: {prob_map.shape}")
            print(f"[DEBUG]   - prob_map dtype: {prob_map.dtype}")
            print(f"[DEBUG]   - prob_map range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
            print(f"[DEBUG]   - prob_map mean: {prob_map.mean():.3f}")
            
            # Apply binary mask to probability map (only show probabilities where tumor detected)
            prob_map_masked = prob_map * mask
            print(f"[DEBUG]   - prob_map_masked range: [{prob_map_masked.min():.3f}, {prob_map_masked.max():.3f}]")
            
            # Create colorful probability map heatmap (like Grad-CAM)
            prob_map_uint8 = np.uint8(255 * prob_map_masked)
            prob_map_colored = cv2.applyColorMap(prob_map_uint8, cv2.COLORMAP_JET)
            prob_map_display = cv2.cvtColor(prob_map_colored, cv2.COLOR_BGR2RGB)
            print(f"[DEBUG]   - prob_map_display shape: {prob_map_display.shape}")
            
            response["segmentation"] = {
                "mask_available": True,
                "tumor_area_pixels": result['segmentation']['tumor_area_pixels'],
                "tumor_percentage": result['segmentation']['tumor_percentage'],
                "mask_base64": numpy_to_base64_png(mask_display),
                "prob_map_base64": numpy_to_base64_png(prob_map_display),
                "overlay_base64": numpy_to_base64_png(overlay)
            }
        else:
            response["segmentation"] = {
                "mask_available": False,
                "message": "Segmentation not computed (tumor probability below threshold)"
            }
        
        # Add Grad-CAM if requested and available
        if include_gradcam and 'gradcam' in result:
            gradcam_overlay = self._create_gradcam_overlay(
                result['image_original'],
                result['gradcam']['heatmap']
            )
            response["gradcam_overlay"] = numpy_to_base64_png(gradcam_overlay)
        
        return response
    
    def _create_gradcam_overlay(
        self,
        image_original: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create Grad-CAM overlay visualization.
        
        Uses cv2.addWeighted for cleaner blending (matches test script style).
        
        Args:
            image_original: Original image (normalized to [0, 1])
            heatmap: Grad-CAM heatmap
            alpha: Blending factor for heatmap (default: 0.5)
        
        Returns:
            Grad-CAM overlay as RGB uint8 array
        """
        # Normalize image to [0, 255]
        img_normalized = ((image_original - image_original.min()) / 
                         (image_original.max() - image_original.min() + 1e-8) * 255)
        img_uint8 = np.uint8(img_normalized)
        
        # Convert grayscale to RGB
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match image size
        h, w = image_original.shape
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Convert heatmap to colored visualization using OpenCV
        cam_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images using cv2.addWeighted for cleaner result
        gradcam_overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        
        return gradcam_overlay
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def is_available(self) -> bool:
        """Check if multi-task service is available."""
        return self.model_manager.multitask_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get multi-task model information."""
        if not self.is_available():
            return {}
        
        return self.model_manager.multitask.get_model_info()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get model performance metrics."""
        return {
            'classification_accuracy': 0.9130,
            'classification_sensitivity': 0.9714,
            'classification_roc_auc': 0.9184,
            'segmentation_dice': 0.7650,
            'segmentation_iou': 0.6401,
            'combined_metric': 0.8390
        }


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Test multi-task service
    print("Testing MultiTaskService...")
    
    from app.backend.services.model_loader import get_model_manager
    
    # Get model manager
    manager = get_model_manager()
    manager.load_all_models()
    
    # Create service
    service = MultiTaskService(manager)
    print(f"\n1. Created MultiTaskService")
    print(f"   Available: {service.is_available()}")
    
    if service.is_available():
        # Test with dummy image
        print("\n2. Testing conditional prediction:")
        test_image = np.random.rand(256, 256).astype(np.float32)
        
        import asyncio
        result = asyncio.run(service.predict_conditional(test_image))
        print(f"   Segmentation computed: {result.segmentation_computed}")
        print(f"   Processing time: {result.processing_time_ms:.2f}ms")
        print(f"   Recommendation: {result.recommendation}")
        
        # Test full prediction
        print("\n3. Testing full prediction:")
        result_full = asyncio.run(service.predict_full(test_image))
        print(f"   Segmentation computed: {result_full.segmentation_computed}")
        print(f"   Processing time: {result_full.processing_time_ms:.2f}ms")
    
    print("\n✅ MultiTaskService test complete!")
