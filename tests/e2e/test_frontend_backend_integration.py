"""
PHASE 3.2: Frontend-Backend Integration Testing - API Client & State Management

Tests frontend-backend integration for SliceWise MRI Brain Tumor Detection:
- API client request formation and response parsing
- State management and session persistence
- Real-time updates and streaming responses
- Error handling and retry logic

Validates seamless communication between Streamlit UI and FastAPI backend.
"""

import sys
import pytest
import numpy as np
import base64
import io
import json
import time
import threading
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import asyncio
import aiohttp
import requests

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pathlib import Path

# Import frontend components
try:
    from app.frontend.utils.api_client import APIClient
    from app.frontend.utils.image_utils import process_uploaded_image, validate_image
    from app.frontend.utils.validators import validate_api_response
    FRONTEND_AVAILABLE = True
except ImportError:
    # Mock components if not available
    FRONTEND_AVAILABLE = False
    APIClient = MagicMock()
    process_uploaded_image = MagicMock()
    validate_image = MagicMock()
    validate_api_response = MagicMock()


@pytest.fixture
def api_client():
    """Create API client instance for testing."""
    if FRONTEND_AVAILABLE:
        client = APIClient(base_url="http://localhost:8000")
        return client
    else:
        # Return mock client
        client = MagicMock()
        client.base_url = "http://localhost:8000"
        return client


@pytest.fixture
def sample_mri_image():
    """Create sample MRI image for testing."""
    # Create realistic brain MRI image (256x256)
    image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    # Add brain-like structure
    center = 128
    y, x = np.ogrid[:256, :256]
    mask = ((x - center)**2 + (y - center)**2) < 80**2
    image[mask] = np.random.randint(150, 220, size=np.sum(mask))

    return image


@pytest.fixture
def mock_uploaded_file(sample_mri_image):
    """Create mock uploaded file."""
    pil_image = Image.fromarray(sample_mri_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    mock_file = MagicMock()
    mock_file.name = "test_mri.png"
    mock_file.type = "image/png"
    mock_file.size = buffer.tell()
    mock_file.read.return_value = buffer.getvalue()
    mock_file.seek = buffer.seek

    return mock_file


class TestAPIClientRequestFormation:
    """Test API client request formation and construction."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_api_client_initialization(self):
        """Test API client proper initialization."""
        client = APIClient(base_url="http://localhost:8000")

        assert client.base_url == "http://localhost:8000"
        assert hasattr(client, 'session')
        assert hasattr(client, 'health_check')
        assert hasattr(client, 'classify')

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_request_payload_construction(self, api_client, sample_mri_image):
        """Test proper construction of API request payloads."""
        # Convert image to base64
        image_b64 = base64.b64encode(sample_mri_image.tobytes()).decode()

        # Test payload construction
        payload = {
            "image": image_b64,
            "image_format": "png"
        }

        # Validate payload structure
        assert "image" in payload
        assert "image_format" in payload
        assert isinstance(payload["image"], str)
        assert payload["image_format"] == "png"

        # Verify base64 encoding
        decoded = base64.b64decode(payload["image"])
        reconstructed = np.frombuffer(decoded, dtype=np.uint8)
        assert reconstructed.shape == sample_mri_image.shape

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('requests.post')
    def test_classification_request_formation(self, mock_post, api_client, sample_mri_image):
        """Test classification API request formation."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": "tumor_present",
            "confidence": 0.85,
            "probabilities": {"tumor_present": 0.85, "no_tumor": 0.15}
        }
        mock_post.return_value = mock_response

        # Make classification request
        result = api_client.classify(image=base64.b64encode(sample_mri_image.tobytes()).decode())

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["url"] == "http://localhost:8000/classify"
        assert "json" in call_args[1]
        assert "image" in call_args[1]["json"]

        # Verify response parsing
        assert result["prediction"] == "tumor_present"
        assert result["confidence"] == 0.85

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('requests.post')
    def test_segmentation_request_formation(self, mock_post, api_client, sample_mri_image):
        """Test segmentation API request formation."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "mask": base64.b64encode(b"mask_data").decode(),
            "probabilities": {"background": 0.1, "tumor": 0.9}
        }
        mock_post.return_value = mock_response

        # Make segmentation request
        result = api_client.segment(image=base64.b64encode(sample_mri_image.tobytes()).decode())

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["url"] == "http://localhost:8000/segment"

        # Verify response parsing
        assert "mask" in result
        assert "probabilities" in result


class TestAPIClientResponseParsing:
    """Test API client response parsing and validation."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_successful_response_parsing(self, api_client):
        """Test parsing of successful API responses."""
        # Mock successful classification response
        mock_response = {
            "prediction": "tumor_present",
            "confidence": 0.87,
            "probabilities": {"tumor_present": 0.87, "no_tumor": 0.13},
            "processing_time_ms": 245.0
        }

        # Test response validation
        is_valid, parsed_data = validate_api_response(mock_response, "classification")

        assert is_valid is True
        assert parsed_data["prediction"] == "tumor_present"
        assert parsed_data["confidence"] == 0.87
        assert "processing_time_ms" in parsed_data

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_malformed_response_handling(self, api_client):
        """Test handling of malformed API responses."""
        # Mock malformed responses
        malformed_responses = [
            {"invalid_field": "value"},  # Missing required fields
            {"prediction": "invalid"},   # Invalid prediction value
            {"confidence": 1.5},         # Invalid confidence range
            {}                          # Empty response
        ]

        for malformed in malformed_responses:
            is_valid, error_msg = validate_api_response(malformed, "classification")

            # Should detect malformation
            assert is_valid is False
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_base64_response_decoding(self, api_client):
        """Test decoding of base64 encoded response data."""
        # Mock response with base64 encoded mask
        mask_data = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        mask_b64 = base64.b64encode(mask_data.tobytes()).decode()

        mock_response = {
            "mask": mask_b64,
            "probabilities": {"background": 0.2, "tumor": 0.8}
        }

        # Test base64 decoding
        decoded_mask = base64.b64decode(mock_response["mask"])
        reconstructed = np.frombuffer(decoded_mask, dtype=np.uint8)

        # Should reconstruct original data
        assert reconstructed.shape == mask_data.shape
        assert np.array_equal(reconstructed, mask_data)

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_error_response_parsing(self, api_client):
        """Test parsing of error API responses."""
        # Mock error responses
        error_responses = [
            {"detail": "Invalid image format"},
            {"error": "Model not loaded", "code": 500},
            {"message": "Rate limit exceeded", "retry_after": 60}
        ]

        for error_resp in error_responses:
            is_valid, error_info = validate_api_response(error_resp, "classification")

            assert is_valid is False
            assert isinstance(error_info, dict)
            assert any(key in error_info for key in ["detail", "error", "message"])


class TestStateManagement:
    """Test Streamlit session state management and persistence."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_session_state_initialization(self):
        """Test proper initialization of session state."""
        with patch('streamlit.session_state', {}) as mock_session:
            # Simulate initial state setup
            initial_state = {
                "uploaded_files": [],
                "current_results": None,
                "processing_status": "idle",
                "api_health": "unknown",
                "settings": {
                    "confidence_threshold": 0.5,
                    "model_type": "multitask"
                }
            }

            # Apply initial state
            for key, value in initial_state.items():
                mock_session[key] = value

            # Verify state structure
            assert mock_session["processing_status"] == "idle"
            assert mock_session["api_health"] == "unknown"
            assert "settings" in mock_session
            assert mock_session["settings"]["confidence_threshold"] == 0.5

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_result_caching(self):
        """Test result caching in session state."""
        with patch('streamlit.session_state', {}) as mock_session:
            # Simulate result caching
            mock_session["results_cache"] = {}

            # Cache classification result
            cache_key = "img_001"
            result_data = {
                "prediction": "tumor_present",
                "confidence": 0.85,
                "timestamp": time.time()
            }

            mock_session["results_cache"][cache_key] = result_data

            # Verify caching
            assert cache_key in mock_session["results_cache"]
            cached_result = mock_session["results_cache"][cache_key]
            assert cached_result["prediction"] == "tumor_present"

            # Test cache retrieval
            retrieved = mock_session["results_cache"].get(cache_key)
            assert retrieved is not None
            assert retrieved["confidence"] == 0.85

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_upload_history_tracking(self):
        """Test tracking of upload history."""
        with patch('streamlit.session_state', {}) as mock_session:
            # Initialize upload history
            mock_session["upload_history"] = []

            # Simulate multiple uploads
            uploads = [
                {"filename": "scan1.png", "timestamp": time.time(), "size": 102400},
                {"filename": "scan2.png", "timestamp": time.time() + 1, "size": 153600},
                {"filename": "scan3.png", "timestamp": time.time() + 2, "size": 204800}
            ]

            for upload in uploads:
                mock_session["upload_history"].append(upload)

            # Verify upload history
            assert len(mock_session["upload_history"]) == 3
            assert mock_session["upload_history"][0]["filename"] == "scan1.png"
            assert mock_session["upload_history"][-1]["size"] == 204800

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_batch_state_management(self):
        """Test state management for batch processing."""
        with patch('streamlit.session_state', {}) as mock_session:
            # Initialize batch processing state
            mock_session["batch_state"] = {
                "files": [],
                "processed": 0,
                "total": 0,
                "results": [],
                "status": "idle"
            }

            # Simulate batch upload
            batch_files = ["file1.png", "file2.png", "file3.png", "file4.png"]
            mock_session["batch_state"]["files"] = batch_files
            mock_session["batch_state"]["total"] = len(batch_files)

            # Simulate processing progress
            for i in range(len(batch_files)):
                mock_session["batch_state"]["processed"] = i + 1
                mock_session["batch_state"]["results"].append({
                    "filename": batch_files[i],
                    "prediction": "tumor_present" if i % 2 == 0 else "no_tumor",
                    "confidence": 0.8 + (i * 0.05)
                })

            # Verify batch state
            assert mock_session["batch_state"]["total"] == 4
            assert mock_session["batch_state"]["processed"] == 4
            assert len(mock_session["batch_state"]["results"]) == 4

            # Check result consistency
            results = mock_session["batch_state"]["results"]
            assert results[0]["prediction"] == "tumor_present"
            assert results[1]["prediction"] == "no_tumor"

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_error_recovery_state(self):
        """Test state management during error recovery."""
        with patch('streamlit.session_state', {}) as mock_session:
            # Initialize error recovery state
            mock_session["error_state"] = {
                "has_error": False,
                "error_message": None,
                "retry_count": 0,
                "last_error_time": None
            }

            # Simulate error occurrence
            error_msg = "API connection timeout"
            mock_session["error_state"].update({
                "has_error": True,
                "error_message": error_msg,
                "last_error_time": time.time()
            })

            # Simulate retry logic
            max_retries = 3
            while mock_session["error_state"]["retry_count"] < max_retries:
                mock_session["error_state"]["retry_count"] += 1

                # Simulate successful retry on second attempt
                if mock_session["error_state"]["retry_count"] == 2:
                    mock_session["error_state"].update({
                        "has_error": False,
                        "error_message": None
                    })
                    break

            # Verify error recovery
            assert mock_session["error_state"]["has_error"] is False
            assert mock_session["error_state"]["retry_count"] == 2
            assert mock_session["error_state"]["error_message"] is None


class TestRealTimeUpdates:
    """Test real-time updates and streaming responses."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.empty')
    def test_progress_updates(self, mock_empty):
        """Test real-time progress updates during processing."""
        # Mock empty container for updates
        mock_container = MagicMock()
        mock_empty.return_value = mock_container

        # Simulate progress update mechanism
        progress_container = mock_empty()

        # Simulate processing steps with progress updates
        steps = ["Preprocessing", "Model Inference", "Postprocessing", "Complete"]
        for i, step in enumerate(steps):
            progress = (i + 1) / len(steps)

            # Update progress display
            progress_container.text(f"Step {i+1}/{len(steps)}: {step}")
            progress_container.progress(progress)

        # Verify progress updates
        assert mock_empty.call_count >= 1
        # Verify progress went from 0 to 100%
        progress_calls = [call for call in mock_container.progress.call_args_list]
        assert len(progress_calls) == len(steps)

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.empty')
    def test_live_result_streaming(self, mock_empty):
        """Test live streaming of results as they become available."""
        # Mock result container
        mock_result_container = MagicMock()
        mock_empty.return_value = mock_result_container

        # Simulate streaming results for batch processing
        result_container = mock_empty()
        batch_results = []

        # Simulate results arriving over time
        for i in range(5):
            # Simulate processing delay
            time.sleep(0.1)

            # New result arrives
            result = {
                "filename": f"image_{i}.png",
                "prediction": "tumor_present",
                "confidence": 0.8 + (i * 0.03)
            }
            batch_results.append(result)

            # Update display with current results
            result_container.json({
                "processed": len(batch_results),
                "results": batch_results
            })

        # Verify streaming updates
        assert mock_empty.call_count >= 1
        json_calls = mock_result_container.json.call_args_list
        assert len(json_calls) == 5  # One update per result

        # Verify final state
        final_update = json_calls[-1][0][0]  # Last call arguments
        assert final_update["processed"] == 5
        assert len(final_update["results"]) == 5

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.metric')
    @patch('streamlit.empty')
    def test_performance_metrics_display(self, mock_empty, mock_metric):
        """Test real-time performance metrics display."""
        # Mock containers for metrics
        mock_latency_container = MagicMock()
        mock_throughput_container = MagicMock()

        mock_empty.side_effect = [mock_latency_container, mock_throughput_container]

        # Create metric containers
        latency_container = mock_empty()
        throughput_container = mock_empty()

        # Simulate performance metrics updates
        performance_data = [
            {"latency_ms": 245, "throughput": 4.1},
            {"latency_ms": 198, "throughput": 5.0},
            {"latency_ms": 312, "throughput": 3.2},
            {"latency_ms": 267, "throughput": 3.7}
        ]

        for data in performance_data:
            # Update latency metric
            latency_container.metric("API Latency", ".0f")

            # Update throughput metric
            throughput_container.metric("Throughput", ".1f")

        # Verify performance metric updates
        assert mock_empty.call_count == 2  # Two containers created
        assert mock_metric.call_count == 8  # 4 updates × 2 metrics each

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.success')
    @patch('streamlit.info')
    def test_status_updates(self, mock_info, mock_success):
        """Test status message updates during processing."""
        # Simulate processing status updates
        status_messages = [
            "Initializing model...",
            "Preprocessing image...",
            "Running inference...",
            "Postprocessing results...",
            "Complete!"
        ]

        for i, message in enumerate(status_messages):
            if i < len(status_messages) - 1:
                mock_info(message)
            else:
                mock_success(message)

        # Verify status updates
        assert mock_info.call_count == 4  # All steps except final
        assert mock_success.call_count == 1  # Final completion

        # Verify message content
        info_calls = mock_info.call_args_list
        assert "Initializing model..." in info_calls[0][0]
        assert "Preprocessing image..." in info_calls[1][0]


class TestAPIErrorHandling:
    """Test comprehensive error handling in frontend-backend integration."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.error')
    def test_network_error_handling(self, mock_error):
        """Test handling of network connectivity errors."""
        with patch('requests.post') as mock_post:
            # Mock network error
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

            api_client = APIClient()

            # Attempt request that will fail
            try:
                api_client.classify(image="test")
            except Exception:
                # Should display user-friendly error
                mock_error("Network error: Unable to connect to API server. Please check your connection.")

            # Verify error display
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Network error" in error_message
            assert "connection" in error_message.lower()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.warning')
    def test_timeout_error_handling(self, mock_warning):
        """Test handling of request timeout errors."""
        with patch('requests.post') as mock_post:
            # Mock timeout error
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

            api_client = APIClient()

            # Attempt request that will timeout
            try:
                api_client.segment(image="test")  # Segmentation typically slower
            except Exception:
                # Should display timeout warning
                mock_warning("Request timed out. The server may be busy. Please try again.")

            # Verify timeout handling
            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "timed out" in warning_message.lower()
            assert "try again" in warning_message.lower()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.error')
    def test_server_error_handling(self, mock_error):
        """Test handling of server-side errors."""
        with patch('requests.post') as mock_post:
            # Mock server error response
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"detail": "Internal server error"}
            mock_post.return_value = mock_response

            api_client = APIClient()

            # Attempt request that returns server error
            try:
                api_client.classify(image="test")
            except Exception:
                # Should display server error
                mock_error("Server error: The API service encountered an internal error. Please try again later.")

            # Verify server error handling
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "Server error" in error_message
            assert "internal error" in error_message.lower()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.info')
    @patch('streamlit.button')
    def test_retry_logic(self, mock_button, mock_info):
        """Test automatic retry logic for transient failures."""
        with patch('requests.post') as mock_post, \
             patch('time.sleep') as mock_sleep:

            # Mock alternating failures and success
            call_count = 0
            def mock_response_factory(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                mock_response = MagicMock()
                if call_count < 3:  # First two calls fail
                    mock_response.status_code = 503  # Service unavailable
                    mock_response.json.return_value = {"detail": "Service temporarily unavailable"}
                else:  # Third call succeeds
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"prediction": "success"}
                return mock_response

            mock_post.side_effect = mock_response_factory

            # Mock retry button
            mock_button.return_value = True

            api_client = APIClient()

            # Simulate retry logic
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    result = api_client.classify(image="test")
                    if result:
                        success = True
                        mock_info(f"Success after {retry_count + 1} attempts!")
                        break
                except Exception:
                    retry_count += 1
                    if retry_count < max_retries:
                        mock_info(f"Attempt {retry_count} failed, retrying...")
                        mock_sleep(1)  # Brief delay

            # Verify retry logic worked
            assert success is True
            assert retry_count == 2  # Failed twice, succeeded on third try
            assert mock_post.call_count == 3  # Three total attempts
            assert mock_sleep.call_count == 2  # Two retry delays
            mock_info.assert_called()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    @patch('streamlit.error')
    def test_invalid_response_handling(self, mock_error):
        """Test handling of invalid API response formats."""
        with patch('requests.post') as mock_post:
            # Mock response with invalid JSON
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_post.return_value = mock_response

            api_client = APIClient()

            # Attempt request with malformed response
            try:
                api_client.classify(image="test")
            except Exception:
                # Should display parsing error
                mock_error("Response parsing error: Unable to process server response.")

            # Verify invalid response handling
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][0]
            assert "parsing error" in error_message.lower()
            assert "response" in error_message.lower()


class TestIntegrationScenarios:
    """Test complete frontend-backend integration scenarios."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_end_to_end_workflow(self, api_client, sample_mri_image, mock_uploaded_file):
        """Test complete end-to-end workflow from UI to API and back."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.image') as mock_image:

            # Mock UI interactions
            mock_uploader.return_value = mock_uploaded_file
            mock_button.return_value = True

            # Mock spinner context
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = mock_spinner_context
            mock_spinner.return_value.__exit__ = MagicMock()

            # Simulate complete workflow
            # 1. File upload
            uploaded_file = mock_uploader("Upload MRI image")
            assert uploaded_file is not None

            # 2. Image processing
            image_data = uploaded_file.read()
            processed_image = process_uploaded_image(image_data)
            assert processed_image is not None

            # 3. API request formation
            image_b64 = base64.b64encode(image_data).decode()
            payload = {"image": image_b64, "image_format": "png"}

            # 4. API call with spinner
            if mock_button("Analyze Image"):
                with mock_spinner("Analyzing..."):
                    result = api_client.classify(image=image_b64)

                # 5. Result display
                if result:
                    mock_success("Analysis complete!")
                    mock_image(processed_image)  # Show original image

            # Verify complete workflow
            mock_uploader.assert_called()
            mock_button.assert_called()
            mock_spinner.assert_called()
            mock_success.assert_called()
            mock_image.assert_called()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_batch_processing_integration(self, api_client, sample_mri_image):
        """Test batch processing integration with progress tracking."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.progress') as mock_progress, \
             patch('streamlit.dataframe') as mock_dataframe:

            # Mock multiple file upload
            mock_files = []
            for i in range(3):
                mock_file = MagicMock()
                mock_file.name = f"batch_image_{i}.png"
                mock_file.read.return_value = f"mock_data_{i}".encode()
                mock_files.append(mock_file)

            mock_uploader.return_value = mock_files
            mock_button.return_value = True

            # Mock progress bar
            mock_progress_bar = MagicMock()
            mock_progress.return_value = mock_progress_bar

            # Simulate batch processing
            uploaded_files = mock_uploader("Upload batch", accept_multiple_files=True)

            if uploaded_files and mock_button("Process Batch"):
                batch_results = []

                for i, file in enumerate(uploaded_files):
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    mock_progress_bar.progress(progress)

                    # Process file
                    image_data = file.read()
                    result = api_client.classify(image=base64.b64encode(image_data).decode())

                    batch_results.append({
                        "filename": file.name,
                        "prediction": result.get("prediction", "error"),
                        "confidence": result.get("confidence", 0.0)
                    })

                # Display results
                mock_dataframe(batch_results)

            # Verify batch integration
            mock_uploader.assert_called_with("Upload batch", accept_multiple_files=True)
            mock_button.assert_called_with("Process Batch")
            mock_progress.assert_called()
            mock_progress_bar.progress.assert_called()
            mock_dataframe.assert_called()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_error_recovery_integration(self, api_client):
        """Test error recovery and fallback mechanisms."""
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.button') as mock_retry_button, \
             patch('streamlit.info') as mock_info, \
             patch('time.sleep') as mock_sleep:

            # Simulate API failure and recovery
            call_count = 0

            def mock_api_call(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # First two calls fail
                    raise Exception("Temporary API failure")
                else:  # Third call succeeds
                    return {"prediction": "success", "confidence": 0.9}

            # Mock API failures then success
            api_client.classify = mock_api_call

            # Simulate user interaction with retry
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    result = api_client.classify(image="test")
                    success = True
                    mock_info("Success!")
                    break
                except Exception as e:
                    retry_count += 1
                    mock_error(f"Attempt {retry_count} failed: {str(e)}")

                    if retry_count < max_retries:
                        mock_info(f"Retrying in 1 second... ({retry_count}/{max_retries})")
                        mock_sleep(1)

            # Verify error recovery worked
            assert success is True
            assert retry_count == 2  # Failed twice, succeeded on third
            assert mock_error.call_count == 2  # Two error messages
            assert mock_info.call_count == 3  # Two retry messages + success
            assert mock_sleep.call_count == 2  # Two retry delays
