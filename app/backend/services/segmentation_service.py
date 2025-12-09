"""
Segmentation service for SliceWise Backend.

This module provides the SegmentationService class that handles all
segmentation-related business logic including predictions, uncertainty
estimation, and post-processing.

Extracted from main_v2.py segmentation endpoints (lines 575-790).
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional
import time

from app.backend.services.model_loader import ModelManager
from app.backend.models.responses import SegmentationResponse, BatchResponse
from app.backend.utils.image_processing import preprocess_image_for_segmentation
from app.backend.utils.visualization import numpy_to_base64_png, create_overlay
from src.inference.postprocess import postprocess_mask
from app.backend.config.settings import settings


# ============================================================================
# Segmentation Service
# ============================================================================

class SegmentationService:
    """
    Service class for segmentation operations.
    
    Handles:
    - Single image segmentation
    - Batch segmentation
    - Uncertainty estimation (MC Dropout + TTA)
    - Post-processing (morphology, filtering)
    
    Extracted from main_v2.py (lines 575-790).
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize segmentation service.
        
        Args:
            model_manager: ModelManager instance with loaded models
        """
        self.model_manager = model_manager
    
    # ========================================================================
    # Single Image Segmentation
    # ========================================================================
    
    async def segment_single(
        self,
        image_array: np.ndarray,
        threshold: float = 0.5,
        apply_postprocessing: bool = True,
        min_object_size: int = 50,
        return_overlay: bool = True
    ) -> SegmentationResponse:
        """
        Segment a single MRI slice.
        
        Extracted from main_v2.py segment_slice() (lines 589-657).
        
        Args:
            image_array: Input image as numpy array
            threshold: Probability threshold for segmentation
            apply_postprocessing: Whether to apply morphological post-processing
            min_object_size: Minimum tumor area in pixels
            return_overlay: Whether to return overlay visualization
        
        Returns:
            SegmentationResponse with segmentation results
            
        Raises:
            ValueError: If no segmentation model available
        """
        # Use multi-task model if standalone segmentation not available
        if not self.model_manager.segmentation_loaded and self.model_manager.multitask_loaded:
            # Use multi-task model for segmentation (always compute segmentation)
            result = self.model_manager.multitask.predict_full(
                image_array,
                include_gradcam=False
            )
            
            if result['segmentation_computed'] and result['segmentation']:
                seg = result['segmentation']
                return SegmentationResponse(
                    has_tumor=seg['tumor_area_pixels'] > 0,
                    tumor_probability=seg['tumor_percentage'] / 100.0,
                    tumor_area_pixels=seg['tumor_area_pixels'],
                    num_components=1,  # Multi-task doesn't provide this
                    mask_base64=seg.get('mask_base64', ''),
                    probability_map_base64=seg.get('prob_map_base64'),
                    overlay_base64=seg.get('overlay_base64')
                )
            else:
                # No tumor detected
                return SegmentationResponse(
                    has_tumor=False,
                    tumor_probability=0.0,
                    tumor_area_pixels=0,
                    num_components=0,
                    mask_base64='',
                    probability_map_base64=None,
                    overlay_base64=None
                )
        
        if not self.model_manager.segmentation_loaded:
            raise ValueError("No segmentation model available")
        
        # Save original image for visualization (before z-score normalization)
        image_original = self._prepare_original_for_viz(image_array)
        
        # Preprocess image for segmentation (applies z-score normalization)
        preprocessed = preprocess_image_for_segmentation(image_array)
        
        # Run segmentation
        result = self.model_manager.segmentation.predict_slice(preprocessed)
        prob_map = result.get('prob', result['mask'])
        binary_mask = result['mask']
        
        # Apply post-processing if requested
        if apply_postprocessing:
            binary_mask, stats = self._apply_postprocessing(
                prob_map,
                threshold,
                min_object_size
            )
        else:
            binary_mask = (prob_map > threshold).astype(np.uint8)
            stats = {
                'num_components': np.unique(binary_mask).sum() - 1,
                'total_area': binary_mask.sum()
            }
        
        # Calculate metrics
        has_tumor = stats.get('final_pixels', stats.get('total_area', 0)) > 0
        tumor_probability = float(prob_map.max()) if has_tumor else 0.0
        
        # Create visualizations using ORIGINAL image (not z-score normalized)
        mask_base64 = numpy_to_base64_png(binary_mask * 255)
        prob_map_base64 = numpy_to_base64_png(prob_map)
        
        overlay_base64 = None
        if return_overlay:
            overlay = create_overlay(image_original, binary_mask)
            overlay_base64 = numpy_to_base64_png(overlay)
        
        return SegmentationResponse(
            has_tumor=has_tumor,
            tumor_probability=tumor_probability,
            tumor_area_pixels=int(stats.get('final_pixels', stats.get('total_area', 0))),
            tumor_area_mm2=None,  # Would need pixel spacing metadata
            num_components=int(stats.get('num_components', 0)),
            mask_base64=mask_base64,
            probability_map_base64=prob_map_base64,
            overlay_base64=overlay_base64
        )
    
    # ========================================================================
    # Uncertainty Estimation
    # ========================================================================
    
    async def segment_with_uncertainty(
        self,
        image_array: np.ndarray,
        threshold: float = 0.5,
        min_object_size: int = 50,
        mc_iterations: int = 10,
        use_tta: bool = True
    ) -> SegmentationResponse:
        """
        Segment with uncertainty estimation using MC Dropout and/or TTA.
        
        Extracted from main_v2.py segment_with_uncertainty() (lines 660-726).
        
        Args:
            image_array: Input image as numpy array
            threshold: Probability threshold
            min_object_size: Minimum tumor area in pixels
            mc_iterations: Number of MC Dropout iterations
            use_tta: Whether to use Test-Time Augmentation
        
        Returns:
            SegmentationResponse with uncertainty metrics
            
        Raises:
            ValueError: If segmentation model not loaded
        """
        if not self.model_manager.segmentation_loaded:
            raise ValueError("Segmentation model not loaded")
        
        # Save original image for visualization
        image_original = self._prepare_original_for_viz(image_array)
        
        # Preprocess image for segmentation
        preprocessed = preprocess_image_for_segmentation(image_array)
        
        # Run uncertainty estimation
        image_tensor = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).float()
        result = self.model_manager.uncertainty.predict_with_uncertainty(image_tensor)
        
        # Apply post-processing
        binary_mask, stats = postprocess_mask(
            result['mean'],
            threshold=threshold,
            min_object_size=min_object_size
        )
        
        # Create visualizations using ORIGINAL image
        mask_base64 = numpy_to_base64_png(binary_mask * 255)
        prob_map_base64 = numpy_to_base64_png((result['mean'] * 255).astype(np.uint8))
        uncertainty_base64 = numpy_to_base64_png((result['epistemic'] * 255).astype(np.uint8))
        overlay_base64 = numpy_to_base64_png(create_overlay(image_original, binary_mask))
        
        return SegmentationResponse(
            has_tumor=stats['total_area'] > 0,
            tumor_probability=float(result['mean'].max()),
            tumor_area_pixels=int(stats['total_area']),
            num_components=int(stats['num_components']),
            mask_base64=mask_base64,
            probability_map_base64=prob_map_base64,
            overlay_base64=overlay_base64,
            uncertainty_map_base64=uncertainty_base64,
            metrics={
                'epistemic_uncertainty': float(result['epistemic'].mean()),
                'aleatoric_uncertainty': float(result['aleatoric'].mean()),
                'total_uncertainty': float(result['std'].mean())
            }
        )
    
    # ========================================================================
    # Batch Segmentation
    # ========================================================================
    
    async def segment_batch(
        self,
        images: List[np.ndarray],
        filenames: Optional[List[str]] = None,
        threshold: float = 0.5,
        min_object_size: int = 50
    ) -> BatchResponse:
        """
        Segment multiple MRI slices in a batch.
        
        Extracted from main_v2.py segment_batch() (lines 729-790).
        
        Args:
            images: List of input images as numpy arrays
            filenames: Optional list of filenames for results
            threshold: Probability threshold
            min_object_size: Minimum tumor area in pixels
        
        Returns:
            BatchResponse with batch results and summary
            
        Raises:
            ValueError: If segmentation model not loaded
        """
        if not self.model_manager.segmentation_loaded:
            raise ValueError("Segmentation model not loaded")
        
        start_time = time.time()
        results = []
        tumor_count = 0
        total_tumor_area = 0
        
        for idx, image_array in enumerate(images):
            # Preprocess
            preprocessed = preprocess_image_for_segmentation(image_array)
            
            # Segment
            result = self.model_manager.segmentation.predict_slice(preprocessed)
            prob_map = result.get('prob', result['mask'])
            binary_mask = result['mask']
            binary_mask, stats = postprocess_mask(
                prob_map,
                threshold=threshold,
                min_object_size=min_object_size
            )
            
            has_tumor = stats['total_area'] > 0
            if has_tumor:
                tumor_count += 1
                total_tumor_area += stats['total_area']
            
            results.append({
                "filename": filenames[idx] if filenames else f"image_{idx}",
                "has_tumor": has_tumor,
                "tumor_area_pixels": int(stats['total_area']),
                "tumor_probability": float(prob_map.max())
            })
        
        processing_time = time.time() - start_time
        
        return BatchResponse(
            num_images=len(images),
            processing_time_seconds=processing_time,
            results=results,
            summary={
                "slices_with_tumor": tumor_count,
                "slices_without_tumor": len(images) - tumor_count,
                "total_tumor_area_pixels": total_tumor_area,
                "avg_tumor_area": total_tumor_area / tumor_count if tumor_count > 0 else 0,
                "avg_time_per_image": processing_time / len(images) if images else 0,
                "tumor_percentage": (tumor_count / len(images) * 100) if images else 0
            }
        )
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _prepare_original_for_viz(self, image_array: np.ndarray) -> np.ndarray:
        """
        Prepare original image for visualization (before z-score normalization).
        
        Args:
            image_array: Input image array
        
        Returns:
            Image normalized to [0, 1] for visualization
        """
        if len(image_array.shape) == 3:
            image_original = image_array[:, :, 0].astype(np.float32)
        else:
            image_original = image_array.astype(np.float32)
        
        if image_original.max() > 1.0:
            image_original = image_original / 255.0
        
        return image_original
    
    def _apply_postprocessing(
        self,
        prob_map: np.ndarray,
        threshold: float,
        min_object_size: int
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply morphological post-processing.
        
        Extracted from main_v2.py (lines 613-627).
        
        Args:
            prob_map: Probability map from segmentation
            threshold: Probability threshold
            min_object_size: Minimum object size in pixels
        
        Returns:
            Tuple of (binary_mask, statistics_dict)
        """
        binary_mask, stats = postprocess_mask(
            prob_map,
            threshold=threshold,
            min_object_size=min_object_size,
            fill_holes_size=settings.postprocessing.fill_holes_size,
            morphology_op=settings.postprocessing.morphology_op,
            morphology_kernel=settings.postprocessing.morphology_kernel
        )
        return binary_mask, stats
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def is_available(self) -> bool:
        """Check if segmentation service is available."""
        # Use standalone segmentation if available, otherwise fall back to multi-task model
        return self.model_manager.segmentation_loaded or self.model_manager.multitask_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get segmentation model information."""
        if not self.is_available():
            return {}
        
        return {
            'architecture': 'U-Net 2D',
            'parameters': '31.4M',
            'device': str(self.model_manager.segmentation.device),
            'uncertainty_available': self.model_manager.uncertainty is not None
        }


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Test segmentation service
    print("Testing SegmentationService...")
    
    from app.backend.services.model_loader import get_model_manager
    
    # Get model manager
    manager = get_model_manager()
    manager.load_all_models()
    
    # Create service
    service = SegmentationService(manager)
    print(f"\n1. Created SegmentationService")
    print(f"   Available: {service.is_available()}")
    
    if service.is_available():
        # Test with dummy image
        print("\n2. Testing single segmentation:")
        test_image = np.random.rand(256, 256).astype(np.float32)
        
        import asyncio
        result = asyncio.run(service.segment_single(test_image))
        print(f"   Has tumor: {result.has_tumor}")
        print(f"   Tumor area: {result.tumor_area_pixels} pixels")
        
        # Test batch
        print("\n3. Testing batch segmentation:")
        test_images = [np.random.rand(256, 256).astype(np.float32) for _ in range(3)]
        batch_result = asyncio.run(service.segment_batch(test_images))
        print(f"   Processed: {batch_result.num_images} images")
        print(f"   Time: {batch_result.processing_time_seconds:.4f}s")
    
    print("\n✅ SegmentationService test complete!")
