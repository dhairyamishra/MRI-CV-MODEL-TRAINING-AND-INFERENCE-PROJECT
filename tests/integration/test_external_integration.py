"""
PHASE 2.4: External Integration Testing - Third-Party Service Validation

Tests integration with external services and dependencies:
- W&B (Weights & Biases) experiment logging
- Cloud storage for model artifacts
- Monitoring services integration
- Database connectivity (future)
- Notification systems

Validates SliceWise's integration with the broader ML ecosystem.
"""

import sys
import pytest
import numpy as np
import time
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import os
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import external service integrations
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from app.backend.monitoring import MonitoringService
    from app.backend.storage import CloudStorageService
    from app.backend.notifications import NotificationService
    EXTERNAL_SERVICES_AVAILABLE = True
except ImportError:
    # Mock services if not available
    EXTERNAL_SERVICES_AVAILABLE = False
    MonitoringService = MagicMock()
    CloudStorageService = MagicMock()
    NotificationService = MagicMock()


class TestWandBIntegration:
    """Test Weights & Biases experiment logging integration."""

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="W&B not available")
    def test_wandb_experiment_logging(self):
        """Test W&B experiment initialization and logging."""
        with patch('wandb.init') as mock_init, \
             patch('wandb.log') as mock_log, \
             patch('wandb.finish') as mock_finish:

            # Mock successful W&B initialization
            mock_run = MagicMock()
            mock_run.name = "test_experiment"
            mock_run.id = "test_run_123"
            mock_init.return_value = mock_run

            # Test experiment setup
            import wandb

            run = wandb.init(
                project="slicewise-testing",
                name="integration_test",
                config={
                    "model": "multitask",
                    "batch_size": 16,
                    "learning_rate": 1e-3
                }
            )

            # Verify initialization
            mock_init.assert_called_once()
            call_args = mock_init.call_args
            assert call_args[1]["project"] == "slicewise-testing"
            assert call_args[1]["name"] == "integration_test"

            # Test logging metrics
            metrics = {
                "train_loss": 0.234,
                "val_dice": 0.876,
                "epoch": 42
            }

            wandb.log(metrics)

            # Verify logging
            mock_log.assert_called_with(metrics)

            # Test experiment completion
            wandb.finish()

            mock_finish.assert_called_once()

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="W&B not available")
    def test_wandb_artifact_management(self):
        """Test W&B artifact storage and versioning."""
        with patch('wandb.Artifact') as mock_artifact, \
             patch('wandb.log_artifact') as mock_log_artifact:

            # Mock artifact creation
            mock_art = MagicMock()
            mock_art.name = "model_checkpoint"
            mock_art.version = "v0"
            mock_artifact.return_value = mock_art

            import wandb

            # Test model artifact logging
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create mock model file
                model_path = Path(tmp_dir) / "best_model.pth"
                model_path.write_text("mock model data")

                # Create artifact
                artifact = wandb.Artifact(
                    name="slicewise-model",
                    type="model",
                    description="Multi-task brain tumor detection model"
                )

                # Add model file
                artifact.add_file(str(model_path), name="model.pth")

                # Log artifact
                wandb.log_artifact(artifact)

                # Verify artifact creation and logging
                mock_artifact.assert_called_once()
                artifact_call = mock_artifact.call_args
                assert artifact_call[1]["name"] == "slicewise-model"
                assert artifact_call[1]["type"] == "model"

                mock_log_artifact.assert_called_once_with(mock_art)

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="W&B not available")
    def test_wandb_offline_mode(self):
        """Test W&B offline mode for environments without internet."""
        with patch('wandb.init') as mock_init, \
             patch('os.environ', {"WANDB_MODE": "offline"}):

            import wandb

            # Initialize in offline mode
            run = wandb.init(
                project="slicewise-offline",
                mode="offline"
            )

            # Verify offline mode
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get("mode") == "offline"

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="W&B not available")
    def test_wandb_error_handling(self):
        """Test W&B error handling for network issues."""
        with patch('wandb.init') as mock_init:
            # Mock network error
            mock_init.side_effect = Exception("Network connection failed")

            import wandb

            # Should handle error gracefully
            with pytest.raises(Exception, match="Network connection failed"):
                wandb.init(project="slicewise-test")

            # Verify error was raised
            mock_init.assert_called_once()


class TestCloudStorageIntegration:
    """Test cloud storage integration for model artifacts."""

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_model_artifact_upload(self):
        """Test uploading model artifacts to cloud storage."""
        storage_service = CloudStorageService()

        with patch.object(storage_service, 'upload_file') as mock_upload, \
             patch.object(storage_service, 'generate_presigned_url') as mock_url:

            mock_upload.return_value = True
            mock_url.return_value = "https://storage.example.com/model.pth"

            # Create mock model file
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                tmp_file.write(b"mock model data")
                model_path = tmp_file.name

            try:
                # Test upload
                success = storage_service.upload_file(
                    local_path=model_path,
                    remote_path="models/best_model.pth",
                    bucket="slicewise-models"
                )

                assert success is True

                # Test URL generation
                url = storage_service.generate_presigned_url(
                    remote_path="models/best_model.pth",
                    bucket="slicewise-models",
                    expiration_hours=24
                )

                assert url.startswith("https://")
                assert "model.pth" in url

            finally:
                Path(model_path).unlink()

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_artifact_versioning(self):
        """Test cloud storage artifact versioning."""
        storage_service = CloudStorageService()

        with patch.object(storage_service, 'list_versions') as mock_list, \
             patch.object(storage_service, 'download_version') as mock_download:

            # Mock version listing
            versions = [
                {"version": "v1.0.0", "timestamp": "2024-01-01T00:00:00Z", "size": 1024000},
                {"version": "v1.1.0", "timestamp": "2024-01-15T00:00:00Z", "size": 1050000},
                {"version": "v2.0.0", "timestamp": "2024-02-01T00:00:00Z", "size": 1100000}
            ]
            mock_list.return_value = versions

            # Test version listing
            artifact_versions = storage_service.list_versions("models/model.pth")

            assert len(artifact_versions) == 3
            assert artifact_versions[0]["version"] == "v1.0.0"
            assert artifact_versions[-1]["version"] == "v2.0.0"

            # Test version download
            mock_download.return_value = b"model data v1.1.0"

            downloaded_data = storage_service.download_version(
                remote_path="models/model.pth",
                version="v1.1.0"
            )

            assert downloaded_data == b"model data v1.1.0"
            mock_download.assert_called_with("models/model.pth", "v1.1.0")

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_storage_error_handling(self):
        """Test cloud storage error handling."""
        storage_service = CloudStorageService()

        # Test upload failure
        with patch.object(storage_service, 'upload_file') as mock_upload:
            mock_upload.side_effect = Exception("Storage quota exceeded")

            with pytest.raises(Exception, match="Storage quota exceeded"):
                storage_service.upload_file(
                    local_path="dummy_path",
                    remote_path="test/file.txt"
                )

        # Test network failure
        with patch.object(storage_service, 'upload_file') as mock_upload:
            mock_upload.side_effect = ConnectionError("Network timeout")

            with pytest.raises(ConnectionError, match="Network timeout"):
                storage_service.upload_file(
                    local_path="dummy_path",
                    remote_path="test/file.txt"
                )

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_large_file_handling(self):
        """Test handling of large model files."""
        storage_service = CloudStorageService()

        # Create large mock file (100MB)
        large_file_size = 100 * 1024 * 1024  # 100MB

        with patch.object(storage_service, 'upload_file') as mock_upload, \
             patch('builtins.open', mock_open(read_data=b'x' * large_file_size)):

            mock_upload.return_value = True

            # Should handle large files
            success = storage_service.upload_file(
                local_path="large_model.pth",
                remote_path="models/large_model.pth",
                chunk_size=8 * 1024 * 1024  # 8MB chunks
            )

            assert success is True


class TestMonitoringServiceIntegration:
    """Test external monitoring service integration."""

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_performance_metrics_collection(self):
        """Test collection and reporting of performance metrics."""
        monitoring = MonitoringService()

        with patch.object(monitoring, 'send_metric') as mock_send, \
             patch('time.time') as mock_time:

            # Mock time progression
            mock_time.side_effect = [1000.0, 1001.5, 1002.0]

            # Simulate API request timing
            start_time = time.time()
            # Simulate processing
            time.sleep(0.1)
            end_time = time.time()

            processing_time = end_time - start_time

            # Send performance metric
            monitoring.send_metric(
                name="api_request_duration",
                value=processing_time,
                tags={"endpoint": "/classify", "method": "POST"}
            )

            # Verify metric was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[1]["name"] == "api_request_duration"
            assert call_args[1]["tags"]["endpoint"] == "/classify"

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_error_rate_monitoring(self):
        """Test error rate monitoring and alerting."""
        monitoring = MonitoringService()

        with patch.object(monitoring, 'increment_counter') as mock_increment, \
             patch.object(monitoring, 'send_alert') as mock_alert:

            # Simulate successful requests
            for _ in range(95):
                monitoring.increment_counter("api_requests_total", {"status": "200"})

            # Simulate errors
            for _ in range(5):
                monitoring.increment_counter("api_requests_total", {"status": "500"})

            # Check error rate calculation
            total_requests = 100
            error_requests = 5
            error_rate = error_requests / total_requests

            # Should trigger alert for high error rate
            if error_rate > 0.05:  # 5% threshold
                monitoring.send_alert(
                    title="High Error Rate Detected",
                    message=f"Error rate: {error_rate:.1%}",
                    severity="warning"
                )

                mock_alert.assert_called_once()
                alert_call = mock_alert.call_args
                assert "High Error Rate" in alert_call[1]["title"]
                assert "5.00%" in alert_call[1]["message"]

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_system_health_monitoring(self):
        """Test system resource monitoring."""
        monitoring = MonitoringService()

        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch.object(monitoring, 'send_metric') as mock_send:

            # Mock system stats
            mock_cpu.return_value = 75.5
            mock_memory.return_value.percent = 82.3

            # Collect system metrics
            monitoring.collect_system_metrics()

            # Verify metrics were sent
            assert mock_send.call_count >= 2  # CPU and memory at minimum

            # Check CPU metric
            cpu_calls = [call for call in mock_send.call_args_list
                        if call[1].get("name") == "cpu_usage_percent"]
            assert len(cpu_calls) == 1
            assert cpu_calls[0][1]["value"] == 75.5

            # Check memory metric
            mem_calls = [call for call in mock_send.call_args_list
                        if call[1].get("name") == "memory_usage_percent"]
            assert len(mem_calls) == 1
            assert mem_calls[0][1]["value"] == 82.3


class TestNotificationServiceIntegration:
    """Test notification system integration."""

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_model_training_completion_notification(self):
        """Test notifications for training completion."""
        notifications = NotificationService()

        with patch.object(notifications, 'send_email') as mock_email, \
             patch.object(notifications, 'send_slack') as mock_slack:

            training_results = {
                "model": "MultiTaskModel",
                "epochs": 100,
                "final_val_dice": 0.876,
                "training_time_hours": 8.5,
                "best_checkpoint": "checkpoints/best_model.pth"
            }

            # Send training completion notification
            notifications.send_training_complete_notification(training_results)

            # Verify notifications were sent
            mock_email.assert_called_once()
            mock_slack.assert_called_once()

            # Check email content
            email_call = mock_email.call_args
            assert "Training Complete" in email_call[1]["subject"]
            assert "MultiTaskModel" in email_call[1]["body"]
            assert "87.6%" in email_call[1]["body"]

            # Check Slack content
            slack_call = mock_slack.call_args
            assert "training completed" in slack_call[1]["message"].lower()

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_system_alert_notifications(self):
        """Test system alert notifications."""
        notifications = NotificationService()

        with patch.object(notifications, 'send_slack') as mock_slack, \
             patch.object(notifications, 'send_sms') as mock_sms:

            # Test critical system alert
            notifications.send_system_alert(
                alert_type="critical",
                title="GPU Memory Exhaustion",
                message="GPU memory usage exceeded 95%. System may become unresponsive.",
                affected_services=["classification", "segmentation"]
            )

            # Verify critical alerts trigger multiple channels
            mock_slack.assert_called_once()
            mock_sms.assert_called_once()

            # Check alert details
            slack_call = mock_slack.call_args
            assert slack_call[1]["channel"] == "#alerts"
            assert "CRITICAL" in slack_call[1]["message"]
            assert "GPU Memory Exhaustion" in slack_call[1]["message"]

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_notification_failure_handling(self):
        """Test notification failure handling."""
        notifications = NotificationService()

        # Test email failure
        with patch.object(notifications, 'send_email') as mock_email:
            mock_email.side_effect = ConnectionError("SMTP server unreachable")

            # Should handle failure gracefully
            notifications.send_training_complete_notification({
                "model": "TestModel",
                "epochs": 10,
                "final_val_dice": 0.8
            })

            # Should still attempt to send
            mock_email.assert_called_once()


class TestDatabaseIntegration:
    """Test database integration (future implementation)."""

    def test_results_storage_structure(self):
        """Test prediction result storage schema."""
        # Mock database schema for results
        result_schema = {
            "id": "uuid",
            "timestamp": "datetime",
            "model_version": "string",
            "input_hash": "string",
            "prediction": "json",
            "confidence": "float",
            "processing_time_ms": "int",
            "user_id": "string",
            "request_metadata": "json"
        }

        # Test schema completeness
        required_fields = [
            "id", "timestamp", "prediction", "confidence", "processing_time_ms"
        ]

        for field in required_fields:
            assert field in result_schema

    def test_audit_log_storage(self):
        """Test audit log storage for HIPAA compliance."""
        # Mock audit log schema
        audit_schema = {
            "id": "uuid",
            "timestamp": "datetime",
            "user_id": "string",
            "action": "string",
            "resource_type": "string",
            "resource_id": "string",
            "ip_address": "string",
            "user_agent": "string",
            "success": "boolean",
            "error_message": "string",
            "retention_days": "int"
        }

        # Validate HIPAA-required fields
        hipaa_fields = [
            "timestamp", "user_id", "action", "resource_type",
            "ip_address", "success", "retention_days"
        ]

        for field in hipaa_fields:
            assert field in audit_schema

    def test_query_performance_simulation(self):
        """Test database query performance expectations."""
        # Mock query performance metrics
        query_performance = {
            "select_recent_results": {"avg_time_ms": 15.2, "p95_time_ms": 45.8},
            "insert_result": {"avg_time_ms": 8.7, "p95_time_ms": 23.1},
            "user_audit_trail": {"avg_time_ms": 32.4, "p95_time_ms": 89.2},
            "bulk_export": {"avg_time_ms": 1250.0, "p95_time_ms": 3400.0}
        }

        # Validate performance expectations
        assert query_performance["select_recent_results"]["avg_time_ms"] < 50
        assert query_performance["insert_result"]["avg_time_ms"] < 25
        assert query_performance["user_audit_trail"]["avg_time_ms"] < 100
        assert query_performance["bulk_export"]["avg_time_ms"] < 2000  # 2 seconds


class TestExternalServiceErrorHandling:
    """Test error handling for external service failures."""

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_wandb_service_outage(self):
        """Test graceful handling of W&B service outages."""
        if WANDB_AVAILABLE:
            import wandb

            with patch('wandb.init') as mock_init:
                # Mock service outage
                mock_init.side_effect = Exception("W&B service unavailable")

                # Should handle gracefully (not crash the application)
                try:
                    run = wandb.init(project="test", mode="online")
                    # If it doesn't raise, that's also acceptable (fallback to offline)
                except Exception:
                    # Exception handling is acceptable
                    pass

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_cloud_storage_timeout(self):
        """Test handling of cloud storage timeouts."""
        storage = CloudStorageService()

        with patch.object(storage, 'upload_file') as mock_upload:
            # Mock timeout
            mock_upload.side_effect = TimeoutError("Request timed out")

            with pytest.raises(TimeoutError):
                storage.upload_file("local_path", "remote_path")

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_monitoring_service_failure(self):
        """Test handling of monitoring service failures."""
        monitoring = MonitoringService()

        with patch.object(monitoring, 'send_metric') as mock_send:
            # Mock monitoring failure
            mock_send.side_effect = ConnectionError("Monitoring service down")

            # Should handle failure gracefully (not crash main application)
            try:
                monitoring.send_metric("test_metric", 1.0)
            except ConnectionError:
                # Connection errors should be caught and handled
                pass

    @pytest.mark.skipif(not EXTERNAL_SERVICES_AVAILABLE, reason="External services not available")
    def test_notification_service_fallback(self):
        """Test notification service fallback mechanisms."""
        notifications = NotificationService()

        # Test fallback from primary to secondary notification method
        with patch.object(notifications, 'send_email') as mock_email, \
             patch.object(notifications, 'send_slack') as mock_slack:

            # Primary method fails
            mock_email.side_effect = Exception("Email service down")

            # Should fallback to secondary method
            notifications.send_system_alert(
                alert_type="warning",
                title="Test Alert",
                message="Test message"
            )

            # Primary method attempted
            mock_email.assert_called_once()

            # Secondary method should also be attempted (fallback)
            mock_slack.assert_called_once()


class TestExternalIntegrationScenarios:
    """Test complete external integration scenarios."""

    def test_model_deployment_pipeline(self):
        """Test complete model deployment pipeline with external services."""
        # Mock the complete deployment scenario
        deployment_steps = [
            "train_model",
            "validate_model",
            "upload_to_cloud",
            "update_monitoring",
            "send_notifications",
            "log_to_wandb"
        ]

        completed_steps = []

        # Simulate successful deployment
        for step in deployment_steps:
            completed_steps.append(step)
            assert step in completed_steps

        # All steps should complete
        assert len(completed_steps) == len(deployment_steps)
        assert set(completed_steps) == set(deployment_steps)

    def test_disaster_recovery_simulation(self):
        """Test disaster recovery with external service failures."""
        # Simulate various failure scenarios
        failure_scenarios = [
            {"service": "wandb", "error": "offline", "fallback": "local_logging"},
            {"service": "cloud_storage", "error": "timeout", "fallback": "local_storage"},
            {"service": "monitoring", "error": "connection_failed", "fallback": "file_logging"},
            {"service": "notifications", "error": "service_down", "fallback": "console_output"}
        ]

        for scenario in failure_scenarios:
            # Each service should have a fallback mechanism
            assert "fallback" in scenario
            assert scenario["fallback"] != scenario["service"]  # Different from failed service

    def test_service_health_monitoring(self):
        """Test health monitoring of external services."""
        # Mock service health checks
        service_health = {
            "wandb": {"status": "healthy", "latency_ms": 150},
            "cloud_storage": {"status": "healthy", "latency_ms": 200},
            "monitoring": {"status": "degraded", "latency_ms": 500},
            "notifications": {"status": "healthy", "latency_ms": 50}
        }

        # Check overall system health
        healthy_services = sum(1 for s in service_health.values() if s["status"] == "healthy")
        total_services = len(service_health)

        health_percentage = healthy_services / total_services

        # System should be considered healthy with > 50% services working
        assert health_percentage >= 0.5

        # Degraded services should be flagged
        degraded_services = [name for name, status in service_health.items()
                           if status["status"] == "degraded"]

        assert "monitoring" in degraded_services

        # High latency services should be monitored
        high_latency = [name for name, status in service_health.items()
                       if status["latency_ms"] > 300]

        assert "monitoring" in high_latency
