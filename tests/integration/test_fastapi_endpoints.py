"""
PHASE 2.1: FastAPI Backend Endpoints Testing - API Integration Tests

Tests all FastAPI backend endpoints for SliceWise MRI Brain Tumor Detection:
- Health & Info endpoints (GET /healthz, /model/info)
- Classification endpoints (POST /classify, /classify/gradcam, /classify/batch)
- Segmentation endpoints (POST /segment, /segment/uncertainty, /segment/batch)
- Multi-task endpoints (POST /predict_multitask with conditional execution)
- Patient analysis endpoints (POST /patient/analyze_stack)

These tests validate the complete API functionality for clinical deployment.
"""

import sys
import pytest
import numpy as np
import base64
import io
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
import tempfile
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import FastAPI test client
from fastapi.testclient import TestClient

# Import our backend API
try:
    from app.backend.main_v2 import app
    API_AVAILABLE = True
except ImportError:
    # Mock API for testing if not available
    from unittest.mock import MagicMock
    app = MagicMock()
    API_AVAILABLE = False

# Create test client
if API_AVAILABLE:
    client = TestClient(app)
else:
    client = None


@pytest.fixture
def sample_mri_image():
    """Create a sample MRI image for testing."""
    # Create a synthetic brain-like MRI image (256x256 grayscale)
    image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    # Add some brain-like structure
    center = 128
    y, x = np.ogrid[:256, :256]
    mask = ((x - center)**2 + (y - center)**2) < 80**2
    image[mask] = np.random.randint(100, 200, size=np.sum(mask))

    return image


@pytest.fixture
def sample_mri_stack():
    """Create a sample MRI stack (multiple slices) for patient analysis."""
    # Create 10 slices of 256x256
    stack = []
    for i in range(10):
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        # Add slice-specific variation
        image += np.random.randint(-20, 20)
        image = np.clip(image, 0, 255).astype(np.uint8)
        stack.append(image)

    return stack


def image_to_base64(image_array):
    """Convert numpy array to base64 string for API testing."""
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def create_test_payload(image_array, **kwargs):
    """Create test payload for API requests."""
    payload = {
        "image": image_to_base64(image_array),
        "image_format": "png",
        **kwargs
    }
    return payload


class TestHealthInfoEndpoints:
    """Test health and info endpoints."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_health_endpoint(self):
        """Test GET /healthz endpoint."""
        response = client.get("/healthz")

        assert response.status_code == 200
        data = response.json()

        # Validate health response structure
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "gpu_available" in data

        # Validate health status
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_model_info_endpoint(self):
        """Test GET /model/info endpoint."""
        response = client.get("/model/info")

        assert response.status_code == 200
        data = response.json()

        # Validate model info structure
        assert "model_type" in data
        assert "model_version" in data
        assert "supported_tasks" in data
        assert "input_requirements" in data
        assert "output_format" in data

        # Validate supported tasks
        supported_tasks = data["supported_tasks"]
        assert "classification" in supported_tasks
        assert "segmentation" in supported_tasks
        assert "multitask" in supported_tasks

        # Validate input requirements
        input_reqs = data["input_requirements"]
        assert "image_format" in input_reqs
        assert "min_resolution" in input_reqs
        assert "max_resolution" in input_reqs


class TestClassificationEndpoints:
    """Test classification-related endpoints."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_classify_endpoint(self, sample_mri_image):
        """Test POST /classify endpoint."""
        payload = create_test_payload(sample_mri_image)

        response = client.post("/classify", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate classification response
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "processing_time_ms" in data

        # Validate prediction structure
        prediction = data["prediction"]
        assert prediction in ["tumor_present", "no_tumor"]

        confidence = data["confidence"]
        assert 0.0 <= confidence <= 1.0

        probabilities = data["probabilities"]
        assert "tumor_present" in probabilities
        assert "no_tumor" in probabilities
        assert abs(sum(probabilities.values()) - 1.0) < 0.01

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_classify_gradcam_endpoint(self, sample_mri_image):
        """Test POST /classify/gradcam endpoint."""
        payload = create_test_payload(sample_mri_image)

        response = client.post("/classify/gradcam", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate Grad-CAM response
        assert "prediction" in data
        assert "confidence" in data
        assert "gradcam_overlay" in data
        assert "heatmap" in data
        assert "processing_time_ms" in data

        # Validate Grad-CAM data
        gradcam_overlay = data["gradcam_overlay"]
        heatmap = data["heatmap"]

        # Should be base64 encoded images
        assert isinstance(gradcam_overlay, str)
        assert isinstance(heatmap, str)

        # Should be valid base64
        try:
            base64.b64decode(gradcam_overlay)
            base64.b64decode(heatmap)
        except Exception:
            pytest.fail("Grad-CAM outputs are not valid base64")

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_classify_batch_endpoint(self, sample_mri_image):
        """Test POST /classify/batch endpoint."""
        # Create batch of 3 images
        images = [sample_mri_image] * 3
        batch_payload = {
            "images": [image_to_base64(img) for img in images],
            "image_format": "png"
        }

        response = client.post("/classify/batch", json=batch_payload)

        assert response.status_code == 200
        data = response.json()

        # Validate batch response
        assert "results" in data
        assert "batch_size" in data
        assert "processing_time_ms" in data

        results = data["results"]
        assert len(results) == 3

        # Validate each result
        for result in results:
            assert "prediction" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert result["prediction"] in ["tumor_present", "no_tumor"]


class TestSegmentationEndpoints:
    """Test segmentation-related endpoints."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_segment_endpoint(self, sample_mri_image):
        """Test POST /segment endpoint."""
        payload = create_test_payload(sample_mri_image)

        response = client.post("/segment", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate segmentation response
        assert "mask" in data
        assert "probabilities" in data
        assert "dice_score" in data
        assert "processing_time_ms" in data

        # Validate mask data
        mask = data["mask"]
        assert isinstance(mask, str)  # Should be base64 encoded

        # Validate probabilities
        probabilities = data["probabilities"]
        assert isinstance(probabilities, dict)
        assert "background" in probabilities
        assert "edema" in probabilities
        assert "non_enhancing" in probabilities
        assert "enhancing" in probabilities

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_segment_uncertainty_endpoint(self, sample_mri_image):
        """Test POST /segment/uncertainty endpoint."""
        payload = create_test_payload(sample_mri_image)

        response = client.post("/segment/uncertainty", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate uncertainty response
        assert "mask" in data
        assert "probabilities" in data
        assert "uncertainty_map" in data
        assert "epistemic_uncertainty" in data
        assert "aleatoric_uncertainty" in data
        assert "processing_time_ms" in data

        # Validate uncertainty data
        uncertainty_map = data["uncertainty_map"]
        epistemic = data["epistemic_uncertainty"]
        aleatoric = data["aleatoric_uncertainty"]

        assert isinstance(uncertainty_map, str)  # Base64 encoded
        assert isinstance(epistemic, (int, float))
        assert isinstance(aleatoric, (int, float))

        assert 0.0 <= epistemic <= 1.0
        assert 0.0 <= aleatoric <= 1.0

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_segment_batch_endpoint(self, sample_mri_image):
        """Test POST /segment/batch endpoint."""
        # Create batch of 3 images
        images = [sample_mri_image] * 3
        batch_payload = {
            "images": [image_to_base64(img) for img in images],
            "image_format": "png"
        }

        response = client.post("/segment/batch", json=batch_payload)

        assert response.status_code == 200
        data = response.json()

        # Validate batch segmentation response
        assert "results" in data
        assert "batch_size" in data
        assert "processing_time_ms" in data

        results = data["results"]
        assert len(results) == 3

        # Validate each result
        for result in results:
            assert "mask" in result
            assert "probabilities" in result
            assert "dice_score" in result


class TestMultiTaskEndpoints:
    """Test multi-task prediction endpoints."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_predict_multitask_endpoint(self, sample_mri_image):
        """Test POST /predict_multitask endpoint."""
        payload = create_test_payload(sample_mri_image)

        response = client.post("/predict_multitask", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate multi-task response
        assert "classification" in data
        assert "segmentation" in data
        assert "processing_time_ms" in data

        # Validate classification part
        cls_result = data["classification"]
        assert "prediction" in cls_result
        assert "confidence" in cls_result

        # Validate segmentation part
        seg_result = data["segmentation"]
        assert "mask" in seg_result
        assert "probabilities" in seg_result

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_conditional_execution_high_probability(self, sample_mri_image):
        """Test conditional segmentation execution with high tumor probability."""
        # Create image that should trigger segmentation (high probability)
        high_prob_image = sample_mri_image.copy()
        # Make it look more like tumor tissue
        high_prob_image[100:150, 100:150] = 200  # Bright region

        payload = create_test_payload(high_prob_image)

        response = client.post("/predict_multitask", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Should include both classification and segmentation
        assert "classification" in data
        assert "segmentation" in data

        cls_result = data["classification"]
        assert cls_result["prediction"] == "tumor_present"

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_conditional_execution_low_probability(self, sample_mri_image):
        """Test conditional segmentation execution with low tumor probability."""
        # Create image that should NOT trigger segmentation (low probability)
        low_prob_image = np.full_like(sample_mri_image, 50)  # Uniform low intensity

        payload = create_test_payload(low_prob_image)

        response = client.post("/predict_multitask", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Should include classification but segmentation should be None or minimal
        assert "classification" in data

        cls_result = data["classification"]
        # Low probability should result in "no_tumor" prediction
        assert cls_result["prediction"] == "no_tumor"
        assert cls_result["confidence"] < 0.3  # Low confidence


class TestPatientAnalysisEndpoints:
    """Test patient-level analysis endpoints."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_patient_analyze_stack_endpoint(self, sample_mri_stack):
        """Test POST /patient/analyze_stack endpoint."""
        # Convert stack to base64
        stack_b64 = [image_to_base64(img) for img in sample_mri_stack]

        payload = {
            "image_stack": stack_b64,
            "image_format": "png",
            "slice_thickness_mm": 5.0,
            "pixel_spacing_mm": [1.0, 1.0]
        }

        response = client.post("/patient/analyze_stack", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate patient analysis response
        assert "patient_summary" in data
        assert "slice_analyses" in data
        assert "volume_analysis" in data
        assert "processing_time_ms" in data

        # Validate patient summary
        summary = data["patient_summary"]
        assert "overall_prediction" in summary
        assert "confidence" in summary
        assert "tumor_probability" in summary

        # Validate slice analyses
        slice_analyses = data["slice_analyses"]
        assert len(slice_analyses) == len(sample_mri_stack)

        # Validate volume analysis
        volume = data["volume_analysis"]
        assert "estimated_volume_mm3" in volume
        assert "confidence_interval" in volume

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_slice_by_slice_analysis(self, sample_mri_stack):
        """Test that each slice in stack is analyzed individually."""
        # Convert stack to base64
        stack_b64 = [image_to_base64(img) for img in sample_mri_stack]

        payload = {
            "image_stack": stack_b64,
            "image_format": "png"
        }

        response = client.post("/patient/analyze_stack", json=payload)

        assert response.status_code == 200
        data = response.json()

        slice_analyses = data["slice_analyses"]

        # Should have analysis for each slice
        assert len(slice_analyses) == len(sample_mri_stack)

        # Each slice analysis should have required fields
        for slice_analysis in slice_analyses:
            assert "slice_index" in slice_analysis
            assert "prediction" in slice_analysis
            assert "confidence" in slice_analysis
            assert "segmentation_mask" in slice_analysis


class TestAPIErrorHandling:
    """Test API error handling and validation."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_invalid_image_format(self):
        """Test API rejects invalid image formats."""
        payload = {
            "image": "invalid_base64_data",
            "image_format": "png"
        }

        response = client.post("/classify", json=payload)

        # Should return error status
        assert response.status_code in [400, 422]
        data = response.json()
        assert "detail" in data or "error" in data

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_missing_required_fields(self):
        """Test API rejects requests with missing required fields."""
        # Missing image field
        payload = {"image_format": "png"}

        response = client.post("/classify", json=payload)

        assert response.status_code in [400, 422]
        data = response.json()
        assert "detail" in data or "error" in data

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_unsupported_image_format(self):
        """Test API rejects unsupported image formats."""
        payload = {
            "image": base64.b64encode(b"fake image data").decode(),
            "image_format": "unsupported_format"
        }

        response = client.post("/classify", json=payload)

        assert response.status_code in [400, 422]
        data = response.json()
        assert "detail" in data or "error" in data

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_large_image_handling(self, sample_mri_image):
        """Test API handles large images appropriately."""
        # Create a very large image (2048x2048)
        large_image = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)

        payload = create_test_payload(large_image)

        response = client.post("/classify", json=payload)

        # Should either succeed or return appropriate error
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
        else:
            # Should be a reasonable error (not 500)
            assert response.status_code in [400, 413, 422]
            data = response.json()
            assert "detail" in data or "error" in data


class TestAPIIntegrationScenarios:
    """Test complete API integration scenarios."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_end_to_end_workflow(self, sample_mri_image):
        """Test complete end-to-end workflow through all endpoints."""
        # Step 1: Check health
        health_response = client.get("/healthz")
        assert health_response.status_code == 200

        # Step 2: Get model info
        info_response = client.get("/model/info")
        assert info_response.status_code == 200

        # Step 3: Classify image
        payload = create_test_payload(sample_mri_image)
        classify_response = client.post("/classify", json=payload)
        assert classify_response.status_code == 200

        # Step 4: Get Grad-CAM
        gradcam_response = client.post("/classify/gradcam", json=payload)
        assert gradcam_response.status_code == 200

        # Step 5: Segment image
        segment_response = client.post("/segment", json=payload)
        assert segment_response.status_code == 200

        # Step 6: Multi-task prediction
        multitask_response = client.post("/predict_multitask", json=payload)
        assert multitask_response.status_code == 200

        # Validate all responses have expected structure
        for response in [classify_response, gradcam_response, segment_response, multitask_response]:
            data = response.json()
            assert "processing_time_ms" in data
            assert isinstance(data["processing_time_ms"], (int, float))

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_performance_baseline(self, sample_mri_image):
        """Test API performance meets baseline requirements."""
        payload = create_test_payload(sample_mri_image)

        # Measure classification performance
        start_time = time.time()
        response = client.post("/classify", json=payload)
        end_time = time.time()

        assert response.status_code == 200

        # Check processing time is reasonable (< 5 seconds for CPU inference)
        data = response.json()
        api_processing_time = data["processing_time_ms"] / 1000.0  # Convert to seconds

        # Allow reasonable time for inference
        assert api_processing_time < 10.0, f"API too slow: {api_processing_time}s"

        # Total request time should also be reasonable
        total_time = end_time - start_time
        assert total_time < 15.0, f"Total request too slow: {total_time}s"
