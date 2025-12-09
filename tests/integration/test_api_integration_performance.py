"""
PHASE 2.2: API Integration & Performance Testing - Request/Response Validation

Tests API request/response validation, concurrent load testing, error handling,
and performance characteristics for SliceWise API endpoints.

Validates Pydantic schemas, Base64 handling, JSON responses, CORS, concurrent users,
rate limiting, memory usage, timeouts, and connection pooling.
"""

import sys
import pytest
import numpy as np
import base64
import io
import json
import time
import threading
import concurrent.futures
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import FastAPI components for testing
try:
    from app.backend.main_v2 import app
    from app.backend.schemas import (
        ClassificationRequest, ClassificationResponse,
        SegmentationRequest, SegmentationResponse,
        MultiTaskRequest, MultiTaskResponse,
        PatientAnalysisRequest, PatientAnalysisResponse
    )
    API_AVAILABLE = True
except ImportError:
    # Mock components if not available
    API_AVAILABLE = False
    ClassificationRequest = MagicMock()
    ClassificationResponse = MagicMock()
    SegmentationRequest = MagicMock()
    SegmentationResponse = MagicMock()
    MultiTaskRequest = MagicMock()
    MultiTaskResponse = MagicMock()
    PatientAnalysisRequest = MagicMock()
    PatientAnalysisResponse = MagicMock()

# Import FastAPI test client
from fastapi.testclient import TestClient

if API_AVAILABLE:
    client = TestClient(app)
else:
    client = None


@pytest.fixture
def valid_mri_image():
    """Create a valid MRI image for testing."""
    # Create realistic brain MRI image (256x256)
    image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    # Add brain-like structure
    center = 128
    y, x = np.ogrid[:256, :256]
    mask = ((x - center)**2 + (y - center)**2) < 80**2
    image[mask] = np.random.randint(150, 220, size=np.sum(mask))

    return image


@pytest.fixture
def oversized_image():
    """Create an oversized image for testing."""
    return np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)


@pytest.fixture
def corrupted_image_data():
    """Create corrupted image data for testing."""
    return b"corrupted image data that is not a valid image"


def image_to_base64(image_array, format='PNG'):
    """Convert numpy array to base64 string."""
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


class TestPydanticSchemaValidation:
    """Test Pydantic schema validation for all request/response models."""

    def test_classification_request_schema(self, valid_mri_image):
        """Test ClassificationRequest schema validation."""
        if not API_AVAILABLE:
            pytest.skip("API not available for testing")

        # Valid request
        valid_payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        request = ClassificationRequest(**valid_payload)
        assert request.image == valid_payload["image"]
        assert request.image_format == "png"

    def test_classification_response_schema(self):
        """Test ClassificationResponse schema validation."""
        if not API_AVAILABLE:
            pytest.skip("API not available for testing")

        # Valid response
        response_data = {
            "prediction": "tumor_present",
            "confidence": 0.85,
            "probabilities": {"tumor_present": 0.85, "no_tumor": 0.15},
            "processing_time_ms": 250.5
        }

        response = ClassificationResponse(**response_data)
        assert response.prediction == "tumor_present"
        assert response.confidence == 0.85
        assert response.processing_time_ms == 250.5

    def test_segmentation_request_schema(self, valid_mri_image):
        """Test SegmentationRequest schema validation."""
        if not API_AVAILABLE:
            pytest.skip("API not available for testing")

        valid_payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png",
            "threshold": 0.5
        }

        request = SegmentationRequest(**valid_payload)
        assert request.image == valid_payload["image"]
        assert request.threshold == 0.5

    def test_patient_analysis_request_schema(self, valid_mri_image):
        """Test PatientAnalysisRequest schema validation."""
        if not API_AVAILABLE:
            pytest.skip("API not available for testing")

        # Create stack of images
        image_stack = [image_to_base64(valid_mri_image)] * 5

        valid_payload = {
            "image_stack": image_stack,
            "image_format": "png",
            "slice_thickness_mm": 5.0,
            "pixel_spacing_mm": [1.0, 1.0]
        }

        request = PatientAnalysisRequest(**valid_payload)
        assert len(request.image_stack) == 5
        assert request.slice_thickness_mm == 5.0


class TestBase64ImageHandling:
    """Test Base64 image encoding/decoding functionality."""

    def test_png_base64_encoding_decoding(self, valid_mri_image):
        """Test PNG Base64 encoding and decoding roundtrip."""
        # Encode to base64
        b64_string = image_to_base64(valid_mri_image, 'PNG')

        # Decode back
        decoded_bytes = base64.b64decode(b64_string)
        decoded_image = Image.open(io.BytesIO(decoded_bytes))
        decoded_array = np.array(decoded_image)

        # Should be identical (within PNG compression limits)
        assert decoded_array.shape[:2] == valid_mri_image.shape[:2]
        # Allow some loss due to PNG compression
        diff = np.abs(decoded_array.astype(float) - valid_mri_image.astype(float))
        assert np.mean(diff) < 5.0  # Reasonable compression difference

    def test_jpeg_base64_encoding_decoding(self, valid_mri_image):
        """Test JPEG Base64 encoding and decoding."""
        # Encode to base64
        b64_string = image_to_base64(valid_mri_image, 'JPEG')

        # Decode back
        decoded_bytes = base64.b64decode(b64_string)
        decoded_image = Image.open(io.BytesIO(decoded_bytes))
        decoded_array = np.array(decoded_image)

        # Should maintain basic structure
        assert decoded_array.shape[:2] == valid_mri_image.shape[:2]
        # JPEG compression is lossy, so we just check it's a valid image

    def test_corrupted_base64_handling(self, corrupted_image_data):
        """Test handling of corrupted Base64 data."""
        # This should be tested at the API level
        if API_AVAILABLE:
            payload = {
                "image": base64.b64encode(corrupted_image_data).decode(),
                "image_format": "png"
            }

            response = client.post("/classify", json=payload)
            # Should return error status
            assert response.status_code in [400, 422, 500]

    def test_invalid_base64_format(self):
        """Test handling of invalid Base64 format."""
        if API_AVAILABLE:
            payload = {
                "image": "not_valid_base64!@#$%",
                "image_format": "png"
            }

            response = client.post("/classify", json=payload)
            # Should return error status
            assert response.status_code in [400, 422, 500]


class TestJSONResponseFormat:
    """Test JSON response format consistency."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_classification_response_format(self, valid_mri_image):
        """Test classification endpoint returns consistent JSON format."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        response = client.post("/classify", json=payload)
        assert response.status_code == 200

        data = response.json()

        # Validate all required fields are present
        required_fields = ["prediction", "confidence", "probabilities", "processing_time_ms"]
        for field in required_fields:
            assert field in data

        # Validate data types
        assert isinstance(data["prediction"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["probabilities"], dict)
        assert isinstance(data["processing_time_ms"], (int, float))

        # Validate probabilities sum to ~1.0
        probs = data["probabilities"]
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 0.01

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_segmentation_response_format(self, valid_mri_image):
        """Test segmentation endpoint returns consistent JSON format."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        response = client.post("/segment", json=payload)
        assert response.status_code == 200

        data = response.json()

        # Validate all required fields are present
        required_fields = ["mask", "probabilities", "dice_score", "processing_time_ms"]
        for field in required_fields:
            assert field in data

        # Validate data types
        assert isinstance(data["mask"], str)  # Base64 encoded
        assert isinstance(data["probabilities"], dict)
        assert isinstance(data["dice_score"], (int, float))
        assert isinstance(data["processing_time_ms"], (int, float))

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_error_response_format(self):
        """Test error responses follow consistent JSON format."""
        # Send invalid request
        payload = {"invalid_field": "invalid_value"}

        response = client.post("/classify", json=payload)
        assert response.status_code in [400, 422]

        data = response.json()

        # Should have error details
        assert "detail" in data or "error" in data or "message" in data


class TestCORSConfiguration:
    """Test Cross-Origin Resource Sharing configuration."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_cors_headers_present(self):
        """Test CORS headers are present in responses."""
        response = client.get("/healthz")

        # Check for CORS headers
        cors_headers = [
            "access-control-allow-origin",
            "access-control-allow-methods",
            "access-control-allow-headers"
        ]

        response_headers = {k.lower(): v for k, v in response.headers.items()}

        # At least some CORS headers should be present
        cors_present = any(header in response_headers for header in cors_headers)
        assert cors_present, "CORS headers not found in response"

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_options_request_handling(self):
        """Test OPTIONS requests are handled for CORS preflight."""
        response = client.options("/classify")

        # OPTIONS should be allowed
        assert response.status_code in [200, 204, 404]  # 404 if not explicitly handled

        if response.status_code == 200:
            # Check for CORS headers in OPTIONS response
            response_headers = {k.lower(): v for k, v in response.headers.items()}
            assert "access-control-allow-methods" in response_headers


class TestConcurrentLoadTesting:
    """Test API performance under concurrent load."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_multiple_concurrent_requests(self, valid_mri_image):
        """Test handling multiple concurrent API requests."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        def make_request():
            response = client.post("/classify", json=payload)
            return response.status_code, response.json() if response.status_code == 200 else None

        # Make 5 concurrent requests
        num_requests = 5

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]

        # All requests should succeed
        success_count = sum(1 for status, _ in results if status == 200)
        assert success_count == num_requests, f"Only {success_count}/{num_requests} requests succeeded"

        # Validate all responses have consistent structure
        for status, data in results:
            if status == 200 and data:
                assert "prediction" in data
                assert "confidence" in data

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_request_queuing_under_load(self, valid_mri_image):
        """Test request queuing and rate limiting under load."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        # Make many rapid requests
        num_requests = 10
        responses = []

        start_time = time.time()

        for _ in range(num_requests):
            response = client.post("/classify", json=payload)
            responses.append((response.status_code, time.time() - start_time))

        end_time = time.time()

        # Calculate success rate
        success_responses = [r for r in responses if r[0] == 200]
        success_rate = len(success_responses) / num_requests

        # Should maintain reasonable success rate under load
        assert success_rate >= 0.8, f"Success rate too low: {success_rate}"

        # Total time should be reasonable
        total_time = end_time - start_time
        avg_time_per_request = total_time / num_requests
        assert avg_time_per_request < 5.0, f"Requests too slow: {avg_time_per_request}s average"

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_resource_usage_under_load(self, valid_mri_image):
        """Test resource usage patterns under concurrent load."""
        import psutil
        import os

        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        def make_request():
            response = client.post("/classify", json=payload)
            return response.status_code

        # Make concurrent requests
        num_requests = 8

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]

        # Check memory usage after load
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 500MB)
        assert memory_increase < 500, f"Memory usage too high: +{memory_increase}MB"

        # All requests should succeed
        success_count = sum(1 for status in results if status == 200)
        assert success_count == num_requests


class TestErrorHandlingRecovery:
    """Test comprehensive error handling and recovery."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON requests."""
        # Send invalid JSON
        response = client.post("/classify", data="invalid json {")

        assert response.status_code in [400, 422]
        data = response.json()
        assert "detail" in data or "error" in data

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_invalid_image_data_handling(self):
        """Test handling of invalid image data."""
        # Send text instead of image
        payload = {
            "image": base64.b64encode(b"This is not an image").decode(),
            "image_format": "png"
        }

        response = client.post("/classify", json=payload)

        # Should handle gracefully
        assert response.status_code in [400, 422, 500]
        data = response.json()
        assert "detail" in data or "error" in data or "message" in data

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_oversized_payload_handling(self, oversized_image):
        """Test handling of oversized payloads."""
        payload = {
            "image": image_to_base64(oversized_image),
            "image_format": "png"
        }

        response = client.post("/classify", json=payload)

        # Should either succeed or return appropriate error
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
        else:
            # Should be client error, not server error
            assert response.status_code in [400, 413, 422]
            data = response.json()
            assert "detail" in data or "error" in data

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_timeout_handling(self, valid_mri_image):
        """Test request timeout handling."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        # Make request with short timeout (this is more of a client-side test)
        # In a real scenario, you might configure server timeouts

        start_time = time.time()
        response = client.post("/segment/uncertainty", json=payload)  # Uncertainty is slower
        end_time = time.time()

        # Request should complete or timeout gracefully
        if response.status_code == 200:
            # If successful, should complete within reasonable time
            processing_time = end_time - start_time
            assert processing_time < 30.0, f"Request too slow: {processing_time}s"
        else:
            # If failed, should be due to timeout or other expected error
            assert response.status_code in [408, 500, 503]  # Timeout or server errors

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_service_recovery_after_errors(self, valid_mri_image):
        """Test service recovers and continues working after errors."""
        # First, send some invalid requests
        invalid_payloads = [
            {"invalid": "payload"},
            {"image": "not_base64", "image_format": "png"},
            {"image": base64.b64encode(b"").decode(), "image_format": "png"}
        ]

        for invalid_payload in invalid_payloads:
            response = client.post("/classify", json=invalid_payload)
            # Errors should be handled gracefully
            assert response.status_code in [400, 422, 500]

        # Now send valid request - service should still work
        valid_payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        response = client.post("/classify", json=valid_payload)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "confidence" in data


class TestConnectionPooling:
    """Test efficient connection reuse and pooling."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_connection_reuse_efficiency(self, valid_mri_image):
        """Test that connections are reused efficiently."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        # Make multiple sequential requests
        num_requests = 10
        response_times = []

        for _ in range(num_requests):
            start_time = time.time()
            response = client.post("/classify", json=payload)
            end_time = time.time()

            assert response.status_code == 200
            response_times.append(end_time - start_time)

        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times)

        # First request might be slower (connection establishment)
        # Subsequent requests should be faster due to connection reuse
        first_request_time = response_times[0]
        later_requests_avg = sum(response_times[1:]) / len(response_times[1:])

        # Later requests should be at least 10% faster (connection reuse benefit)
        improvement_ratio = first_request_time / later_requests_avg
        assert improvement_ratio > 1.05, f"Connection reuse not effective: {improvement_ratio:.2f}x improvement"

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_persistent_connection_handling(self, valid_mri_image):
        """Test persistent connection handling across requests."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        # Test with different endpoints to ensure connection persistence
        endpoints = ["/healthz", "/classify", "/healthz", "/classify", "/healthz"]

        for endpoint in endpoints:
            if endpoint == "/healthz":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json=payload)

            assert response.status_code in [200, 422]  # 422 is OK for invalid requests

        # All requests should have succeeded with persistent connections
        # If connections weren't persistent, we might see timeouts or connection errors


class TestAPIIntegrationScenarios:
    """Test complete API integration scenarios."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_cross_endpoint_consistency(self, valid_mri_image):
        """Test consistency across different endpoints for same input."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        # Get results from different endpoints
        classify_response = client.post("/classify", json=payload)
        multitask_response = client.post("/predict_multitask", json=payload)

        assert classify_response.status_code == 200
        assert multitask_response.status_code == 200

        classify_data = classify_response.json()
        multitask_data = multitask_response.json()

        # Classification results should be consistent
        cls_from_classify = classify_data["prediction"]
        cls_from_multitask = multitask_data["classification"]["prediction"]

        # Should be the same prediction
        assert cls_from_classify == cls_from_multitask

        # Confidence should be similar (within 5%)
        conf_from_classify = classify_data["confidence"]
        conf_from_multitask = multitask_data["classification"]["confidence"]

        confidence_diff = abs(conf_from_classify - conf_from_multitask)
        assert confidence_diff < 0.05, f"Confidence mismatch: {confidence_diff}"

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_batch_vs_individual_consistency(self, valid_mri_image):
        """Test batch processing gives same results as individual requests."""
        # Single request
        single_payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        single_response = client.post("/classify", json=single_payload)
        assert single_response.status_code == 200
        single_data = single_response.json()

        # Batch request with same image
        batch_payload = {
            "images": [image_to_base64(valid_mri_image)],
            "image_format": "png"
        }

        batch_response = client.post("/classify/batch", json=batch_payload)
        assert batch_response.status_code == 200
        batch_data = batch_response.json()

        # Results should be identical
        assert len(batch_data["results"]) == 1
        batch_result = batch_data["results"][0]

        assert single_data["prediction"] == batch_result["prediction"]
        assert abs(single_data["confidence"] - batch_result["confidence"]) < 0.01

        # Probabilities should match
        for class_name in single_data["probabilities"]:
            prob_diff = abs(
                single_data["probabilities"][class_name] -
                batch_result["probabilities"][class_name]
            )
            assert prob_diff < 0.01, f"Probability mismatch for {class_name}: {prob_diff}"

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available for testing")
    def test_api_response_stability(self, valid_mri_image):
        """Test API responses are stable across multiple identical requests."""
        payload = {
            "image": image_to_base64(valid_mri_image),
            "image_format": "png"
        }

        # Make multiple identical requests
        num_requests = 5
        responses = []

        for _ in range(num_requests):
            response = client.post("/classify", json=payload)
            assert response.status_code == 200
            responses.append(response.json())

        # All responses should be identical (deterministic)
        first_response = responses[0]

        for response in responses[1:]:
            assert response["prediction"] == first_response["prediction"]
            assert abs(response["confidence"] - first_response["confidence"]) < 0.01

            # Probabilities should match exactly
            for class_name in first_response["probabilities"]:
                assert abs(
                    response["probabilities"][class_name] -
                    first_response["probabilities"][class_name]
                ) < 0.001
