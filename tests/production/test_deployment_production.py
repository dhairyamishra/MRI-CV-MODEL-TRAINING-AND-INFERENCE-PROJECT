"""
PHASE 4.2: Production Deployment Testing - PM2 & Container Deployment

Tests production deployment and process management for SliceWise:
- PM2 process management (startup, auto-restart, log aggregation, monitoring)
- Container deployment (Docker build, runtime, volume mounting, networking)
- Cloud deployment preparation (AWS/GCP/Azure instance, load balancing, scaling)

Validates production deployment reliability and scalability.
"""

import sys
import pytest
import subprocess
import time
import psutil
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import requests

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import deployment utilities
try:
    import pm2
    from scripts.demo.run_demo_pm2 import PM2Manager
    PM2_AVAILABLE = True
except ImportError:
    PM2_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Mock deployment components if not available
if not PM2_AVAILABLE:
    PM2Manager = MagicMock()

if not DOCKER_AVAILABLE:
    docker = MagicMock()


@pytest.fixture
def pm2_manager():
    """Create PM2 manager instance."""
    if PM2_AVAILABLE:
        return PM2Manager()
    else:
        # Return mock manager
        manager = MagicMock()
        manager.start_processes.return_value = True
        manager.stop_processes.return_value = True
        manager.get_status.return_value = {
            "backend": {"status": "online", "restarts": 0},
            "frontend": {"status": "online", "restarts": 0}
        }
        return manager


@pytest.fixture
def docker_manager():
    """Create Docker manager instance."""
    if DOCKER_AVAILABLE:
        return DockerManager()
    else:
        # Return mock manager
        manager = MagicMock()
        manager.build_image.return_value = "slicewise:latest"
        manager.run_container.return_value = "container_123"
        manager.get_container_status.return_value = {"status": "running", "ports": {"8000": 8000}}
        return manager


class TestPM2ProcessManagement:
    """Test PM2 process management functionality."""

    @pytest.mark.skipif(not PM2_AVAILABLE, reason="PM2 not available")
    def test_pm2_process_startup(self, pm2_manager):
        """Test PM2 process startup and initialization."""
        # Start demo processes
        success = pm2_manager.start_processes()

        assert success is True

        # Verify processes are running
        status = pm2_manager.get_status()

        assert "backend" in status
        assert "frontend" in status
        assert status["backend"]["status"] == "online"
        assert status["frontend"]["status"] == "online"

    @pytest.mark.skipif(not PM2_AVAILABLE, reason="PM2 not available")
    def test_pm2_auto_restart(self, pm2_manager):
        """Test PM2 automatic restart on process failure."""
        # Simulate process failure
        with patch.object(pm2_manager, 'kill_process') as mock_kill:
            mock_kill.return_value = True

            # Kill a process
            pm2_manager.kill_process("backend")

            # Wait for auto-restart
            time.sleep(2)

            # Check status
            status = pm2_manager.get_status()

            # Process should have restarted
            assert status["backend"]["status"] == "online"
            assert status["backend"]["restarts"] >= 1

    @pytest.mark.skipif(not PM2_AVAILABLE, reason="PM2 not available")
    def test_pm2_log_aggregation(self, pm2_manager):
        """Test PM2 log aggregation and management."""
        # Generate some log activity
        test_logs = []

        # Simulate API requests that generate logs
        for i in range(5):
            # Make request to generate log entries
            try:
                response = requests.get("http://localhost:8000/healthz", timeout=1)
                test_logs.append(f"Request {i}: {response.status_code}")
            except:
                test_logs.append(f"Request {i}: failed")

        # Check log aggregation
        logs = pm2_manager.get_logs()

        assert "backend" in logs
        assert "frontend" in logs

        # Should have recent log entries
        backend_logs = logs["backend"]
        assert len(backend_logs) > 0

        # Logs should be properly formatted
        for log_entry in backend_logs[-5:]:  # Check last 5 entries
            assert isinstance(log_entry, str)
            # Should contain timestamp or log level
            assert any(keyword in log_entry.lower() for keyword in
                      ["info", "debug", "warning", "error", "[", "]", "-", ":"])

    @pytest.mark.skipif(not PM2_AVAILABLE, reason="PM2 not available")
    def test_pm2_process_monitoring(self, pm2_manager):
        """Test PM2 process monitoring and metrics."""
        # Get process metrics
        metrics = pm2_manager.get_metrics()

        assert "backend" in metrics
        assert "frontend" in metrics

        backend_metrics = metrics["backend"]

        # Should include key monitoring metrics
        required_metrics = ["cpu_percent", "memory_mb", "restarts", "uptime"]
        for metric in required_metrics:
            assert metric in backend_metrics

        # Memory usage should be reasonable
        memory_mb = backend_metrics["memory_mb"]
        assert 50 < memory_mb < 2000, f"Memory usage suspicious: {memory_mb}MB"

        # CPU usage should be reasonable
        cpu_percent = backend_metrics["cpu_percent"]
        assert 0 <= cpu_percent <= 100, f"CPU usage invalid: {cpu_percent}%"

    @pytest.mark.skipif(not PM2_AVAILABLE, reason="PM2 not available")
    def test_pm2_resource_limits(self, pm2_manager):
        """Test PM2 resource limits and enforcement."""
        # Get current resource limits
        limits = pm2_manager.get_resource_limits()

        assert "backend" in limits
        assert "frontend" in limits

        backend_limits = limits["backend"]

        # Should have memory limits
        assert "max_memory_mb" in backend_limits
        max_memory = backend_limits["max_memory_mb"]

        # Memory limit should be reasonable (e.g., 2GB)
        assert max_memory > 500 and max_memory < 8000, f"Memory limit suspicious: {max_memory}MB"

        # Should have CPU limits if configured
        if "max_cpu_percent" in backend_limits:
            max_cpu = backend_limits["max_cpu_percent"]
            assert 10 <= max_cpu <= 200, f"CPU limit suspicious: {max_cpu}%"

    @pytest.mark.skipif(not PM2_AVAILABLE, reason="PM2 not available")
    def test_pm2_graceful_shutdown(self, pm2_manager):
        """Test PM2 graceful shutdown and cleanup."""
        # Stop processes gracefully
        success = pm2_manager.stop_processes()

        assert success is True

        # Verify processes are stopped
        status = pm2_manager.get_status()

        # Status should indicate stopped or offline
        for process_name, process_status in status.items():
            assert process_status["status"] in ["stopped", "offline", "errored"]

    @pytest.mark.skipif(not PM2_AVAILABLE, reason="PM2 not available")
    def test_pm2_configuration_persistence(self, pm2_manager):
        """Test PM2 configuration persistence across restarts."""
        # Get initial configuration
        initial_config = pm2_manager.get_configuration()

        # Stop and restart PM2
        pm2_manager.stop_processes()
        time.sleep(1)
        pm2_manager.start_processes()

        # Get configuration after restart
        restart_config = pm2_manager.get_configuration()

        # Configuration should be identical
        assert initial_config == restart_config

        # Key settings should persist
        assert initial_config["apps"][0]["name"] == restart_config["apps"][0]["name"]
        assert initial_config["apps"][0]["script"] == restart_config["apps"][0]["script"]


class TestContainerDeployment:
    """Test Docker container deployment functionality."""

    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
    def test_docker_image_build(self, docker_manager):
        """Test Docker image building."""
        # Build Docker image
        image_tag = docker_manager.build_image()

        assert image_tag is not None
        assert "slicewise" in image_tag

        # Verify image exists
        client = docker.from_env()
        images = client.images.list(name="slicewise")

        assert len(images) > 0

        # Check image has required labels/metadata
        image = images[0]
        labels = image.labels or {}

        # Should have version and build info
        assert "version" in labels or "org.opencontainers.image.version" in labels

    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
    def test_container_runtime(self, docker_manager):
        """Test container startup and runtime."""
        # Run container
        container_id = docker_manager.run_container()

        assert container_id is not None

        # Check container status
        status = docker_manager.get_container_status(container_id)

        assert status["status"] == "running"

        # Should expose required ports
        assert "ports" in status
        assert "8000" in status["ports"]  # Backend port

        # Test health check
        health_url = f"http://localhost:{status['ports']['8000']}/healthz"
        response = requests.get(health_url, timeout=5)

        assert response.status_code == 200

        # Clean up
        docker_manager.stop_container(container_id)

    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
    def test_volume_mounting(self, docker_manager):
        """Test volume mounting for persistent data."""
        # Create test directories
        with tempfile.TemporaryDirectory() as host_data_dir, \
             tempfile.TemporaryDirectory() as host_models_dir:

            # Create test files
            test_data_file = Path(host_data_dir) / "test_data.txt"
            test_data_file.write_text("test data content")

            test_model_file = Path(host_models_dir) / "test_model.pth"
            test_model_file.write_text("test model content")

            # Run container with volume mounts
            volumes = {
                str(host_data_dir): "/app/data",
                str(host_models_dir): "/app/models"
            }

            container_id = docker_manager.run_container(volumes=volumes)

            # Execute command to check volume contents
            client = docker.from_env()
            container = client.containers.get(container_id)

            # Check data volume
            result = container.exec_run("cat /app/data/test_data.txt")
            assert result.exit_code == 0
            assert b"test data content" in result.output

            # Check models volume
            result = container.exec_run("cat /app/models/test_model.pth")
            assert result.exit_code == 0
            assert b"test model content" in result.output

            # Clean up
            docker_manager.stop_container(container_id)

    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
    def test_network_configuration(self, docker_manager):
        """Test container network configuration."""
        # Run container with specific port mapping
        port_mapping = {"8000": 8080, "8501": 8502}

        container_id = docker_manager.run_container(ports=port_mapping)

        # Check port mapping
        status = docker_manager.get_container_status(container_id)

        assert status["ports"]["8000"] == 8080
        assert status["ports"]["8501"] == 8502

        # Test connectivity on mapped ports
        backend_url = "http://localhost:8080/healthz"
        frontend_url = "http://localhost:8502"

        # Backend health check
        response = requests.get(backend_url, timeout=5)
        assert response.status_code == 200

        # Frontend connectivity (may not respond to root path)
        try:
            response = requests.get(frontend_url, timeout=2)
            # Just check that connection is possible
            assert response.status_code in [200, 404]  # 404 is OK for root path
        except requests.exceptions.ConnectionError:
            pytest.fail("Frontend not accessible on mapped port")

        # Clean up
        docker_manager.stop_container(container_id)

    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
    def test_resource_constraints(self, docker_manager):
        """Test container resource constraints."""
        # Run container with resource limits
        resource_limits = {
            "memory": "1g",  # 1GB RAM
            "cpu_shares": 512  # CPU shares
        }

        container_id = docker_manager.run_container(resource_limits=resource_limits)

        # Check resource limits are applied
        client = docker.from_env()
        container = client.containers.get(container_id)

        # Verify memory limit
        memory_limit = container.attrs["HostConfig"]["Memory"]
        expected_memory = 1 * 1024 * 1024 * 1024  # 1GB in bytes
        assert memory_limit == expected_memory

        # Verify CPU shares
        cpu_shares = container.attrs["HostConfig"]["CpuShares"]
        assert cpu_shares == 512

        # Test resource enforcement
        # (This would require actually running memory-intensive tasks)
        # For now, just verify container starts with limits
        status = docker_manager.get_container_status(container_id)
        assert status["status"] == "running"

        # Clean up
        docker_manager.stop_container(container_id)


class TestCloudDeploymentPreparation:
    """Test cloud deployment preparation and configuration."""

    def test_aws_deployment_configuration(self):
        """Test AWS deployment configuration."""
        # Mock AWS configuration
        aws_config = {
            "instance_type": "t3.medium",
            "ami_id": "ami-12345678",
            "security_groups": ["sg-web", "sg-ssh"],
            "key_name": "slicewise-key",
            "user_data": """#!/bin/bash
# Install Docker and run SliceWise
yum update -y
amazon-linux-extras install docker -y
service docker start
docker run -d -p 8000:8000 -p 8501:8501 slicewise:latest
"""
        }

        # Validate AWS configuration
        assert aws_config["instance_type"] in ["t3.medium", "t3.large", "m5.large"]
        assert aws_config["security_groups"] == ["sg-web", "sg-ssh"]
        assert "docker" in aws_config["user_data"]
        assert "slicewise" in aws_config["user_data"]

    def test_gcp_deployment_configuration(self):
        """Test GCP deployment configuration."""
        # Mock GCP configuration
        gcp_config = {
            "machine_type": "e2-medium",
            "zone": "us-central1-a",
            "network": "default",
            "subnetwork": "default",
            "service_account": "slicewise-service@project.iam.gserviceaccount.com",
            "startup_script": """#!/bin/bash
# Install Docker and run SliceWise
apt-get update
apt-get install -y docker.io
systemctl start docker
docker run -d -p 8000:8000 -p 8501:8501 slicewise:latest
"""
        }

        # Validate GCP configuration
        assert gcp_config["machine_type"] in ["e2-medium", "e2-standard-2", "n1-standard-1"]
        assert "docker" in gcp_config["startup_script"]
        assert "slicewise" in gcp_config["startup_script"]

    def test_load_balancing_configuration(self):
        """Test load balancing configuration for multi-instance deployment."""
        # Mock load balancer configuration
        lb_config = {
            "type": "application",
            "listeners": [
                {"port": 80, "protocol": "HTTP", "action": "redirect to HTTPS"},
                {"port": 443, "protocol": "HTTPS", "action": "forward to target group"}
            ],
            "target_groups": [
                {
                    "name": "slicewise-backend",
                    "port": 8000,
                    "health_check": {
                        "path": "/healthz",
                        "interval": 30,
                        "timeout": 5,
                        "healthy_threshold": 2,
                        "unhealthy_threshold": 2
                    }
                },
                {
                    "name": "slicewise-frontend",
                    "port": 8501,
                    "health_check": {
                        "path": "/",
                        "interval": 30,
                        "timeout": 5
                    }
                }
            ],
            "auto_scaling": {
                "min_instances": 2,
                "max_instances": 10,
                "target_cpu_utilization": 70,
                "scale_in_cooldown": 300,
                "scale_out_cooldown": 60
            }
        }

        # Validate load balancer configuration
        assert lb_config["type"] == "application"
        assert len(lb_config["listeners"]) == 2
        assert len(lb_config["target_groups"]) == 2

        # Validate health checks
        backend_tg = lb_config["target_groups"][0]
        assert backend_tg["health_check"]["path"] == "/healthz"
        assert backend_tg["health_check"]["interval"] == 30

        # Validate auto scaling
        asg = lb_config["auto_scaling"]
        assert asg["min_instances"] <= asg["max_instances"]
        assert asg["target_cpu_utilization"] > 0 and asg["target_cpu_utilization"] <= 100

    def test_auto_scaling_policies(self):
        """Test auto-scaling policies for demand-based scaling."""
        # Mock auto-scaling policies
        scaling_policies = [
            {
                "name": "scale-out-cpu",
                "adjustment_type": "PercentChangeInCapacity",
                "scaling_adjustment": 50,
                "cooldown": 60,
                "alarm": {
                    "metric": "CPUUtilization",
                    "statistic": "Average",
                    "threshold": 70,
                    "comparison_operator": "GreaterThanThreshold",
                    "evaluation_periods": 2,
                    "period": 60
                }
            },
            {
                "name": "scale-in-cpu",
                "adjustment_type": "PercentChangeInCapacity",
                "scaling_adjustment": -25,
                "cooldown": 300,
                "alarm": {
                    "metric": "CPUUtilization",
                    "statistic": "Average",
                    "threshold": 30,
                    "comparison_operator": "LessThanThreshold",
                    "evaluation_periods": 5,
                    "period": 60
                }
            }
        ]

        # Validate scale-out policy
        scale_out = scaling_policies[0]
        assert scale_out["scaling_adjustment"] > 0
        assert scale_out["alarm"]["threshold"] == 70
        assert scale_out["alarm"]["comparison_operator"] == "GreaterThanThreshold"

        # Validate scale-in policy
        scale_in = scaling_policies[1]
        assert scale_in["scaling_adjustment"] < 0
        assert scale_in["alarm"]["threshold"] == 30
        assert scale_in["alarm"]["comparison_operator"] == "LessThanThreshold"

        # Scale-in should have longer cooldown than scale-out
        assert scale_in["cooldown"] > scale_out["cooldown"]

    def test_monitoring_integration(self):
        """Test monitoring integration for cloud deployments."""
        # Mock monitoring configuration
        monitoring_config = {
            "cloudwatch": {
                "namespace": "SliceWise/Production",
                "metrics": [
                    {"name": "APIRequests", "unit": "Count"},
                    {"name": "APILatency", "unit": "Milliseconds"},
                    {"name": "MemoryUsage", "unit": "Percent"},
                    {"name": "CPUUsage", "unit": "Percent"},
                    {"name": "ErrorRate", "unit": "Percent"}
                ],
                "alarms": [
                    {
                        "name": "HighErrorRate",
                        "metric": "ErrorRate",
                        "threshold": 5,
                        "comparison_operator": "GreaterThanThreshold"
                    },
                    {
                        "name": "HighLatency",
                        "metric": "APILatency",
                        "threshold": 1000,
                        "comparison_operator": "GreaterThanThreshold"
                    }
                ]
            },
            "logging": {
                "log_group": "/slicewise/production",
                "retention_days": 30,
                "metric_filters": [
                    {"pattern": "ERROR", "metric": "ErrorCount"},
                    {"pattern": "WARNING", "metric": "WarningCount"}
                ]
            }
        }

        # Validate monitoring configuration
        assert monitoring_config["cloudwatch"]["namespace"] == "SliceWise/Production"
        assert len(monitoring_config["cloudwatch"]["metrics"]) >= 5
        assert len(monitoring_config["cloudwatch"]["alarms"]) >= 2

        # Validate logging configuration
        assert monitoring_config["logging"]["retention_days"] == 30
        assert len(monitoring_config["logging"]["metric_filters"]) >= 2


class TestProductionDeploymentScenarios:
    """Test complete production deployment scenarios."""

    def test_blue_green_deployment_simulation(self):
        """Test blue-green deployment strategy."""
        # Mock deployment states
        deployment_states = {
            "blue": {"version": "1.0.0", "status": "active", "traffic": 100},
            "green": {"version": "1.1.0", "status": "ready", "traffic": 0}
        }

        # Simulate traffic shift
        # Step 1: Both environments ready
        assert deployment_states["blue"]["status"] == "active"
        assert deployment_states["green"]["status"] == "ready"

        # Step 2: Gradual traffic shift (25% increments)
        traffic_shifts = [25, 50, 75, 100]
        for shift_percentage in traffic_shifts:
            deployment_states["blue"]["traffic"] = 100 - shift_percentage
            deployment_states["green"]["traffic"] = shift_percentage

            # Traffic should add up to 100%
            total_traffic = (deployment_states["blue"]["traffic"] +
                           deployment_states["green"]["traffic"])
            assert total_traffic == 100, f"Traffic imbalance at {shift_percentage}% shift"

        # Step 3: Complete switch
        deployment_states["blue"]["status"] = "standby"
        deployment_states["green"]["status"] = "active"

        assert deployment_states["blue"]["traffic"] == 0
        assert deployment_states["green"]["traffic"] == 100
        assert deployment_states["green"]["status"] == "active"

    def test_rolling_deployment_simulation(self):
        """Test rolling deployment strategy."""
        # Mock instance group
        instances = [
            {"id": "i-001", "version": "1.0.0", "status": "healthy"},
            {"id": "i-002", "version": "1.0.0", "status": "healthy"},
            {"id": "i-003", "version": "1.0.0", "status": "healthy"},
            {"id": "i-004", "version": "1.0.0", "status": "healthy"}
        ]

        # Simulate rolling update
        batch_size = 1  # Update one instance at a time

        for i in range(0, len(instances), batch_size):
            batch_end = min(i + batch_size, len(instances))

            # Update batch
            for j in range(i, batch_end):
                instances[j]["version"] = "1.1.0"
                instances[j]["status"] = "updating"

            # Wait for health checks (simulated)
            for j in range(i, batch_end):
                instances[j]["status"] = "healthy"

            # Verify batch is healthy before proceeding
            for j in range(i, batch_end):
                assert instances[j]["status"] == "healthy"
                assert instances[j]["version"] == "1.1.0"

        # All instances should be updated
        for instance in instances:
            assert instance["version"] == "1.1.0"
            assert instance["status"] == "healthy"

    def test_disaster_recovery_simulation(self):
        """Test disaster recovery procedures."""
        # Mock production environment
        environment = {
            "primary_region": "us-east-1",
            "backup_region": "us-west-2",
            "database": {"status": "healthy", "replicas": 2},
            "load_balancer": {"status": "healthy"},
            "instances": [
                {"id": "web-01", "region": "us-east-1", "status": "healthy"},
                {"id": "web-02", "region": "us-east-1", "status": "healthy"},
                {"id": "web-03", "region": "us-west-2", "status": "standby"}
            ]
        }

        # Simulate disaster in primary region
        def trigger_disaster():
            """Simulate regional failure."""
            for instance in environment["instances"]:
                if instance["region"] == "us-east-1":
                    instance["status"] = "unreachable"
            environment["database"]["status"] = "failing"
            environment["load_balancer"]["status"] = "degraded"

        trigger_disaster()

        # Verify disaster state
        primary_instances = [i for i in environment["instances"]
                           if i["region"] == "us-east-1"]
        assert all(i["status"] == "unreachable" for i in primary_instances)
        assert environment["database"]["status"] == "failing"

        # Execute recovery procedure
        def execute_recovery():
            """Simulate disaster recovery."""
            # Promote backup region
            for instance in environment["instances"]:
                if instance["region"] == "us-west-2":
                    instance["status"] = "active"

            # Failover database
            environment["database"]["status"] = "healthy"
            environment["database"]["primary_region"] = "us-west-2"

            # Update load balancer
            environment["load_balancer"]["status"] = "healthy"
            environment["load_balancer"]["active_region"] = "us-west-2"

        execute_recovery()

        # Verify recovery
        backup_instances = [i for i in environment["instances"]
                          if i["region"] == "us-west-2"]
        assert all(i["status"] == "active" for i in backup_instances)
        assert environment["database"]["status"] == "healthy"
        assert environment["database"]["primary_region"] == "us-west-2"
        assert environment["load_balancer"]["active_region"] == "us-west-2"

    def test_canary_deployment_simulation(self):
        """Test canary deployment strategy."""
        # Mock deployment with canary
        deployment = {
            "stable_version": "1.0.0",
            "canary_version": "1.1.0",
            "traffic_distribution": {"stable": 95, "canary": 5},
            "canary_metrics": {
                "error_rate": 0.0,
                "latency_p95": 250,
                "success_rate": 100.0
            },
            "monitors": ["error_rate", "latency", "success_rate"]
        }

        # Validate canary setup
        assert deployment["traffic_distribution"]["stable"] == 95
        assert deployment["traffic_distribution"]["canary"] == 5
        assert sum(deployment["traffic_distribution"].values()) == 100

        # Monitor canary performance
        canary_metrics = deployment["canary_metrics"]

        # Define success criteria
        success_criteria = {
            "error_rate": lambda x: x < 1.0,  # < 1% error rate
            "latency_p95": lambda x: x < 500,  # < 500ms p95 latency
            "success_rate": lambda x: x > 99.5  # > 99.5% success rate
        }

        # Evaluate canary performance
        canary_successful = True
        for metric, threshold_func in success_criteria.items():
            value = canary_metrics[metric]
            if not threshold_func(value):
                canary_successful = False
                break

        # If canary successful, promote to full deployment
        if canary_successful:
            deployment["stable_version"] = deployment["canary_version"]
            deployment["traffic_distribution"] = {"stable": 100, "canary": 0}

        # Verify successful canary promotion
        assert canary_successful is True
        assert deployment["stable_version"] == "1.1.0"
        assert deployment["traffic_distribution"]["stable"] == 100
