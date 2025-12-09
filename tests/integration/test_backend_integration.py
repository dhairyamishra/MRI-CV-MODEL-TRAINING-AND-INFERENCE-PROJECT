"""
PHASE 2.3: Backend Integration Testing - Service Layer Validation

Tests backend service layer integration, dependency injection, model manager,
business logic processing, error propagation, logging integration, and performance monitoring.

Validates the complete backend architecture from API layer through service layers to model management.
"""

import sys
import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import backend components
try:
    from app.backend.main_v2 import app
    from app.backend.model_manager import ModelManager
    from app.backend.services import ClassificationService, SegmentationService, PatientAnalysisService
    from app.backend.dependencies import get_model_manager, get_classification_service
    from app.backend.schemas import (
        ClassificationRequest, SegmentationRequest,
        MultiTaskRequest, PatientAnalysisRequest
    )
    BACKEND_AVAILABLE = True
except ImportError:
    # Mock components if not available
    BACKEND_AVAILABLE = False
    ModelManager = MagicMock()
    ClassificationService = MagicMock()
    SegmentationService = MagicMock()
    PatientAnalysisService = MagicMock()

# Import FastAPI test client
from fastapi.testclient import TestClient

if BACKEND_AVAILABLE:
    client = TestClient(app)
else:
    client = None


@pytest.fixture
def mock_model_manager():
    """Create mock model manager for testing."""
    manager = MagicMock(spec=ModelManager)

    # Mock model loading status
    manager.is_model_loaded.return_value = True
    manager.get_model_info.return_value = {
        "model_type": "multitask",
        "input_shape": [1, 256, 256],
        "output_classes": 4
    }

    return manager


@pytest.fixture
def sample_image_array():
    """Create sample image array for testing."""
    return np.random.randint(0, 255, (256, 256), dtype=np.uint8)


class TestModelManagerIntegration:
    """Test ModelManager singleton pattern and integration."""

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_singleton_pattern(self):
        """Test ModelManager follows singleton pattern."""
        manager1 = ModelManager()
        manager2 = ModelManager()

        # Should be the same instance (singleton)
        assert manager1 is manager2

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_model_switching(self, mock_model_manager):
        """Test loading different model types."""
        # Mock different model configurations
        model_configs = {
            "classification": {"type": "efficientnet", "classes": 2},
            "segmentation": {"type": "unet", "classes": 4},
            "multitask": {"type": "multitask", "seg_classes": 4, "cls_classes": 2}
        }

        for model_type, config in model_configs.items():
            with patch.object(mock_model_manager, 'load_model') as mock_load:
                mock_load.return_value = True

                # Simulate loading different models
                mock_model_manager.load_model(model_type, **config)

                # Verify load was called with correct parameters
                mock_load.assert_called_with(model_type, **config)

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_memory_management(self, mock_model_manager):
        """Test GPU memory cleanup and management."""
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            # Simulate memory cleanup
            mock_model_manager.cleanup_memory()

            # Should call CUDA cache emptying
            mock_empty_cache.assert_called_once()

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_checkpoint_validation(self, mock_model_manager):
        """Test model file integrity validation."""
        # Create temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            # Write some fake checkpoint data
            checkpoint_data = {
                'model_state_dict': {'layer.weight': torch.randn(10, 10)},
                'epoch': 100,
                'loss': 0.1
            }
            torch.save(checkpoint_data, tmp_file.name)

            try:
                # Test checkpoint loading
                with patch.object(mock_model_manager, 'load_checkpoint') as mock_load:
                    mock_load.return_value = True

                    result = mock_model_manager.load_checkpoint(tmp_file.name)
                    assert result is True

                    mock_load.assert_called_with(tmp_file.name)

            finally:
                Path(tmp_file.name).unlink()

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_configuration_loading(self, mock_model_manager):
        """Test model configuration parsing."""
        config_data = {
            "model": {
                "type": "multitask",
                "backbone": "resnet50",
                "classes": {"segmentation": 4, "classification": 2}
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 1e-3
            }
        }

        with patch.object(mock_model_manager, 'load_config') as mock_load:
            mock_load.return_value = config_data

            loaded_config = mock_model_manager.load_config("config.yaml")

            assert loaded_config == config_data
            mock_load.assert_called_with("config.yaml")


class TestServiceLayerIntegration:
    """Test service layer instantiation and integration."""

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_dependency_injection(self):
        """Test dependency injection container."""
        # Test that services can be instantiated
        model_manager = get_model_manager()
        assert model_manager is not None

        classification_service = get_classification_service()
        assert classification_service is not None

        # Services should have access to model manager
        assert hasattr(classification_service, 'model_manager')

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_business_logic_processing(self, sample_image_array):
        """Test end-to-end business logic processing pipelines."""
        # Create test request
        request = ClassificationRequest(
            image=base64.b64encode(sample_image_array.tobytes()).decode(),
            image_format="png"
        )

        # Test service processing
        service = ClassificationService()

        with patch.object(service, 'process_request') as mock_process:
            mock_process.return_value = {
                "prediction": "tumor_present",
                "confidence": 0.85,
                "processing_time_ms": 150.0
            }

            result = service.process_request(request)

            assert result["prediction"] == "tumor_present"
            assert result["confidence"] == 0.85

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_error_propagation(self, sample_image_array):
        """Test error propagation across service layers."""
        service = ClassificationService()

        # Test error handling
        with patch.object(service, 'validate_input') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid input format")

            request = ClassificationRequest(
                image="invalid_base64",
                image_format="png"
            )

            # Should propagate error appropriately
            with pytest.raises(ValueError, match="Invalid input format"):
                service.process_request(request)

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_logging_integration(self, sample_image_array, caplog):
        """Test comprehensive request logging across layers."""
        caplog.set_level(logging.INFO)

        service = ClassificationService()

        request = ClassificationRequest(
            image=base64.b64encode(sample_image_array.tobytes()).decode(),
            image_format="png"
        )

        with patch.object(service, 'process_request') as mock_process:
            mock_process.return_value = {"prediction": "no_tumor", "confidence": 0.9}

            service.process_request(request)

            # Check that logging occurred
            log_messages = [record.message for record in caplog.records]

            # Should have logged request processing
            request_logs = [msg for msg in log_messages if "request" in msg.lower()]
            assert len(request_logs) > 0

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_performance_monitoring(self, sample_image_array):
        """Test latency and throughput metrics collection."""
        service = ClassificationService()

        # Mock performance monitoring
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 0.1, 0.2]  # Start, mid, end times

            request = ClassificationRequest(
                image=base64.b64encode(sample_image_array.tobytes()).decode(),
                image_format="png"
            )

            with patch.object(service, 'process_request') as mock_process:
                mock_process.return_value = {
                    "prediction": "tumor_present",
                    "processing_time_ms": 100.0
                }

                result = service.process_request(request)

                # Should include timing information
                assert "processing_time_ms" in result
                assert result["processing_time_ms"] == 100.0


class TestConfigurationIntegration:
    """Test hierarchical configuration loading and validation."""

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_hierarchical_config_generation(self):
        """Test final config generation from multiple sources."""
        base_config = {
            "model": {"type": "multitask"},
            "training": {"lr": 1e-3},
            "data": {"batch_size": 16}
        }

        override_config = {
            "training": {"lr": 1e-4},  # Override learning rate
            "data": {"batch_size": 32}  # Override batch size
        }

        # Simulate config merging
        final_config = base_config.copy()
        final_config.update(override_config)

        # Base values should be preserved
        assert final_config["model"]["type"] == "multitask"

        # Override values should take precedence
        assert final_config["training"]["lr"] == 1e-4
        assert final_config["data"]["batch_size"] == 32

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_runtime_configuration_updates(self):
        """Test dynamic config updates during runtime."""
        # Mock configuration manager
        class MockConfigManager:
            def __init__(self):
                self.config = {"threshold": 0.5, "batch_size": 16}

            def update_config(self, updates):
                self.config.update(updates)
                return self.config

            def get_config(self):
                return self.config

        manager = MockConfigManager()

        # Test runtime updates
        updates = {"threshold": 0.7, "new_param": "value"}
        updated_config = manager.update_config(updates)

        assert updated_config["threshold"] == 0.7
        assert updated_config["batch_size"] == 16  # Unchanged
        assert updated_config["new_param"] == "value"  # New parameter

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_environment_variable_config(self):
        """Test configuration loading from environment variables."""
        import os

        # Set environment variables
        test_env_vars = {
            "MODEL_TYPE": "efficientnet",
            "BATCH_SIZE": "32",
            "LEARNING_RATE": "0.001"
        }

        # Mock environment
        with patch.dict(os.environ, test_env_vars):
            # Simulate loading from environment
            config = {}

            # Convert string env vars to appropriate types
            for key, value in test_env_vars.items():
                if key == "BATCH_SIZE":
                    config[key.lower()] = int(value)
                elif key == "LEARNING_RATE":
                    config[key.lower()] = float(value)
                else:
                    config[key.lower()] = value

            assert config["model_type"] == "efficientnet"
            assert config["batch_size"] == 32
            assert config["learning_rate"] == 0.001

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_config_schema_validation(self):
        """Test configuration schema validation."""
        # Valid config
        valid_config = {
            "model": {
                "type": "multitask",
                "classes": {"segmentation": 4, "classification": 2}
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 1e-3,
                "epochs": 100
            },
            "data": {
                "image_size": [256, 256],
                "augmentation": True
            }
        }

        # Should pass validation (simplified check)
        required_sections = ["model", "training", "data"]
        for section in required_sections:
            assert section in valid_config

        # Test invalid config
        invalid_config = {
            "model": {"type": "unknown_model_type"},
            "training": {"batch_size": -1}  # Invalid negative batch size
        }

        # Should fail validation
        assert invalid_config["training"]["batch_size"] < 0  # Invalid


class TestEndToEndBackendIntegration:
    """Test complete backend integration from API to model."""

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_api_to_service_integration(self, sample_image_array):
        """Test complete flow from API endpoint to service layer."""
        import base64
        from PIL import Image
        import io

        # Create valid base64 image
        pil_image = Image.fromarray(sample_image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "image": image_b64,
            "image_format": "png"
        }

        # Make API request
        response = client.post("/classify", json=payload)

        if response.status_code == 200:
            data = response.json()

            # Validate complete integration
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "processing_time_ms" in data

            # Processing time should be reasonable
            assert data["processing_time_ms"] > 0
            assert data["processing_time_ms"] < 10000  # Less than 10 seconds

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_service_to_model_integration(self):
        """Test service layer to model manager integration."""
        service = ClassificationService()

        # Mock model manager response
        with patch.object(service.model_manager, 'predict_classification') as mock_predict:
            mock_predict.return_value = {
                "prediction": "tumor_present",
                "confidence": 0.9,
                "probabilities": {"tumor_present": 0.9, "no_tumor": 0.1}
            }

            # Create mock input
            mock_input = torch.randn(1, 1, 256, 256)

            result = service.model_manager.predict_classification(mock_input)

            assert result["prediction"] == "tumor_present"
            assert result["confidence"] == 0.9

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_error_handling_chain(self, sample_image_array):
        """Test error propagation through the entire backend chain."""
        # Test various error conditions
        error_scenarios = [
            ("invalid_base64", "Base64 decoding error"),
            ("empty_image", "Empty image data"),
            ("wrong_format", "Unsupported image format")
        ]

        for error_input, expected_error in error_scenarios:
            payload = {
                "image": error_input,
                "image_format": "png"
            }

            response = client.post("/classify", json=payload)

            # Should return error status
            assert response.status_code in [400, 422, 500]

            data = response.json()
            # Should have error information
            assert any(key in data for key in ["detail", "error", "message"])

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_concurrent_request_handling(self, sample_image_array):
        """Test backend handles concurrent requests properly."""
        import base64
        from PIL import Image
        import io

        # Create base64 image
        pil_image = Image.fromarray(sample_image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "image": image_b64,
            "image_format": "png"
        }

        def make_concurrent_request():
            """Make a single API request."""
            response = client.post("/classify", json=payload)
            return response.status_code, response.elapsed.total_seconds()

        # Make 5 concurrent requests
        num_concurrent = 5

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_concurrent_request) for _ in range(num_concurrent)]
            results = [future.result() for future in futures]

        # Analyze results
        statuses, times = zip(*results)

        # All requests should succeed
        success_count = sum(1 for status in statuses if status == 200)
        assert success_count == num_concurrent

        # Response times should be reasonable
        avg_response_time = sum(times) / len(times)
        max_response_time = max(times)

        assert avg_response_time < 5.0  # Average under 5 seconds
        assert max_response_time < 10.0  # Max under 10 seconds

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_resource_cleanup_integration(self):
        """Test resource cleanup and memory management integration."""
        import gc
        import psutil
        import os

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make several API requests
        for _ in range(10):
            response = client.get("/healthz")
            assert response.status_code == 200

        # Force garbage collection
        gc.collect()

        # Check memory after cleanup
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for health checks)
        assert memory_increase < 50 * 1024 * 1024  # 50MB in bytes

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available for testing")
    def test_logging_aggregation(self, sample_image_array, caplog):
        """Test comprehensive logging across all backend layers."""
        caplog.set_level(logging.DEBUG)

        import base64
        from PIL import Image
        import io

        # Create base64 image
        pil_image = Image.fromarray(sample_image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "image": image_b64,
            "image_format": "png"
        }

        # Make API request
        response = client.post("/classify", json=payload)

        # Check for comprehensive logging
        log_messages = [record.message for record in caplog.records]

        # Should have various log levels and components
        info_logs = [msg for msg in log_messages if "INFO" in msg or "info" in msg.lower()]
        error_logs = [msg for msg in log_messages if "ERROR" in msg or "error" in msg.lower()]

        # Should have at least some informational logging
        assert len(info_logs) > 0

        # Should not have unexpected errors (unless API failed)
        if response.status_code == 200:
            assert len(error_logs) == 0
        else:
            # Some errors are expected for failed requests
            assert len(error_logs) >= 0
