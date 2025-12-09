"""
Patient-level analysis service for SliceWise Backend.

This module provides the PatientService class that handles patient-level
analysis including volume estimation and aggregated metrics across
multiple MRI slices.

Extracted from main_v2.py analyze_patient_stack endpoint (lines 797-867).
"""

import numpy as np
from typing import List, Dict, Any
from app.backend.services.model_loader import ModelManager
from app.backend.services.segmentation_service import SegmentationService
from app.backend.models.responses import PatientAnalysisResponse
from app.backend.utils.image_processing import preprocess_image_for_segmentation
from src.inference.postprocess import postprocess_mask


# ============================================================================
# Patient Service
# ============================================================================

class PatientService:
    """
    Service class for patient-level analysis.
    
    Handles:
    - Multi-slice analysis
    - Volume estimation
    - Patient-level tumor detection
    - Aggregated metrics
    
    Extracted from main_v2.py (lines 797-867).
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        segmentation_service: SegmentationService
    ):
        """
        Initialize patient service.
        
        Args:
            model_manager: ModelManager instance with loaded models
            segmentation_service: SegmentationService for slice-level analysis
        """
        self.model_manager = model_manager
        self.segmentation_service = segmentation_service
    
    # ========================================================================
    # Patient Stack Analysis
    # ========================================================================
    
    async def analyze_stack(
        self,
        images: List[np.ndarray],
        patient_id: str,
        filenames: List[str] = None,
        threshold: float = 0.5,
        min_object_size: int = 50,
        slice_thickness_mm: float = 1.0,
        pixel_spacing_mm: float = 1.0
    ) -> PatientAnalysisResponse:
        """
        Analyze a stack of MRI slices for patient-level tumor detection.
        
        Extracted from main_v2.py analyze_patient_stack() (lines 811-864).
        
        Args:
            images: List of MRI slice images as numpy arrays
            patient_id: Patient identifier
            filenames: Optional list of filenames for each slice
            threshold: Probability threshold for segmentation
            min_object_size: Minimum tumor area in pixels
            slice_thickness_mm: Slice thickness for volume calculation
            pixel_spacing_mm: Pixel spacing in mm (assumes square pixels)
        
        Returns:
            PatientAnalysisResponse with patient-level results
            
        Raises:
            ValueError: If segmentation model not loaded
        """
        if not self.segmentation_service.is_available():
            raise ValueError("Segmentation model not loaded")
        
        slice_predictions = []
        affected_slices = 0
        total_tumor_volume_pixels = 0
        
        # Analyze each slice
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
                affected_slices += 1
                total_tumor_volume_pixels += stats['total_area']
            
            slice_predictions.append({
                "slice_index": idx,
                "filename": filenames[idx] if filenames else f"slice_{idx:03d}",
                "has_tumor": has_tumor,
                "tumor_area_pixels": int(stats['total_area']),
                "max_probability": float(prob_map.max())
            })
        
        # Patient-level decision: tumor present if any slice has tumor
        patient_has_tumor = affected_slices > 0
        
        # Estimate volume
        tumor_volume_mm3 = None
        if patient_has_tumor:
            tumor_volume_mm3 = self._calculate_volume(
                total_tumor_volume_pixels,
                pixel_spacing_mm,
                slice_thickness_mm
            )
        
        # Calculate patient-level metrics
        patient_metrics = self._aggregate_metrics(
            slice_predictions,
            affected_slices,
            len(images),
            total_tumor_volume_pixels
        )
        
        return PatientAnalysisResponse(
            patient_id=patient_id,
            num_slices=len(images),
            has_tumor=patient_has_tumor,
            tumor_volume_mm3=tumor_volume_mm3,
            affected_slices=affected_slices,
            slice_predictions=slice_predictions,
            patient_level_metrics=patient_metrics
        )
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _calculate_volume(
        self,
        total_area_pixels: int,
        pixel_spacing_mm: float,
        slice_thickness_mm: float
    ) -> float:
        """
        Calculate tumor volume in mm³.
        
        Extracted from main_v2.py (lines 847-850).
        
        Args:
            total_area_pixels: Total tumor area across all slices in pixels
            pixel_spacing_mm: Pixel spacing in mm (assumes square pixels)
            slice_thickness_mm: Slice thickness in mm
        
        Returns:
            Tumor volume in mm³
        """
        # Convert pixel area to mm²
        pixel_area_mm2 = pixel_spacing_mm * pixel_spacing_mm
        
        # Calculate volume: area × thickness
        tumor_volume_mm3 = total_area_pixels * pixel_area_mm2 * slice_thickness_mm
        
        return tumor_volume_mm3
    
    def _aggregate_metrics(
        self,
        slice_predictions: List[Dict[str, Any]],
        affected_slices: int,
        total_slices: int,
        total_tumor_area: int
    ) -> Dict[str, float]:
        """
        Calculate aggregated patient-level metrics.
        
        Extracted from main_v2.py (lines 859-863).
        
        Args:
            slice_predictions: List of per-slice predictions
            affected_slices: Number of slices with tumor
            total_slices: Total number of slices
            total_tumor_area: Total tumor area in pixels
        
        Returns:
            Dictionary of patient-level metrics
        """
        metrics = {
            "affected_slice_ratio": affected_slices / total_slices if total_slices > 0 else 0,
            "avg_tumor_area_per_slice": total_tumor_area / total_slices if total_slices > 0 else 0,
            "avg_tumor_area_per_affected_slice": total_tumor_area / affected_slices if affected_slices > 0 else 0,
            "max_tumor_area": max([p['tumor_area_pixels'] for p in slice_predictions]) if slice_predictions else 0,
            "max_probability": max([p['max_probability'] for p in slice_predictions]) if slice_predictions else 0,
            "avg_probability": sum([p['max_probability'] for p in slice_predictions]) / len(slice_predictions) if slice_predictions else 0
        }
        
        return metrics
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def is_available(self) -> bool:
        """Check if patient service is available."""
        return self.segmentation_service.is_available()
    
    def get_recommended_parameters(self) -> Dict[str, Any]:
        """Get recommended parameters for patient analysis."""
        return {
            'threshold': 0.5,
            'min_object_size': 50,
            'slice_thickness_mm': 1.0,
            'pixel_spacing_mm': 1.0,
            'min_slices_recommended': 10,
            'max_slices_per_analysis': 500
        }


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Test patient service
    print("Testing PatientService...")
    
    from app.backend.services.model_loader import get_model_manager
    from app.backend.services.segmentation_service import SegmentationService
    
    # Get model manager
    manager = get_model_manager()
    manager.load_all_models()
    
    # Create services
    seg_service = SegmentationService(manager)
    patient_service = PatientService(manager, seg_service)
    
    print(f"\n1. Created PatientService")
    print(f"   Available: {patient_service.is_available()}")
    
    if patient_service.is_available():
        # Test with dummy images
        print("\n2. Testing patient stack analysis:")
        test_images = [np.random.rand(256, 256).astype(np.float32) for _ in range(5)]
        
        import asyncio
        result = asyncio.run(patient_service.analyze_stack(
            images=test_images,
            patient_id="TEST_PATIENT_001"
        ))
        
        print(f"   Patient ID: {result.patient_id}")
        print(f"   Num slices: {result.num_slices}")
        print(f"   Has tumor: {result.has_tumor}")
        print(f"   Affected slices: {result.affected_slices}")
        if result.tumor_volume_mm3:
            print(f"   Tumor volume: {result.tumor_volume_mm3:.2f} mm³")
        
        # Show recommended parameters
        print("\n3. Recommended parameters:")
        params = patient_service.get_recommended_parameters()
        for key, value in params.items():
            print(f"   {key}: {value}")
    
    print("\n✅ PatientService test complete!")
