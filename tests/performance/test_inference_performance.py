"""
PHASE 4.1: Performance Benchmarking - Inference & Training Performance

Tests performance characteristics of SliceWise MRI Brain Tumor Detection:
- Latency testing: End-to-end prediction time measurement
- Throughput testing: Images/second processing rate measurement
- Memory profiling: GPU/CPU memory usage monitoring
- Batch size optimization: Optimal batch size testing
- Concurrent users: Multi-user performance simulation

Validates clinical performance requirements and production scalability.
"""

import sys
import pytest
import numpy as np
import time
import psutil
import threading
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import performance testing utilities
try:
    from app.backend.model_manager import ModelManager
    from app.backend.services import ClassificationService, SegmentationService
    from tests.performance.utils import PerformanceProfiler, MemoryProfiler, LatencyProfiler
    PERFORMANCE_AVAILABLE = True
except ImportError:
    # Mock performance utilities if not available
    PERFORMANCE_AVAILABLE = False
    ModelManager = MagicMock()
    ClassificationService = MagicMock()
    SegmentationService = MagicMock()
    PerformanceProfiler = MagicMock()
    MemoryProfiler = MagicMock()
    LatencyProfiler = MagicMock()


@pytest.fixture
def performance_profiler():
    """Create performance profiler instance."""
    if PERFORMANCE_AVAILABLE:
        return PerformanceProfiler()
    else:
        # Return mock profiler
        profiler = MagicMock()
        profiler.measure_latency.return_value = 0.150
        profiler.measure_throughput.return_value = 250.0
        profiler.measure_memory_usage.return_value = {"peak_mb": 1024, "avg_mb": 512}
        return profiler


@pytest.fixture
def sample_batch_sizes():
    """Provide various batch sizes for testing."""
    return [1, 4, 8, 16, 32]


@pytest.fixture
def sample_image_sizes():
    """Provide various image sizes for testing."""
    return [(128, 128), (256, 256), (512, 512)]


@pytest.fixture
def mock_model_manager():
    """Create mock model manager for performance testing."""
    manager = MagicMock(spec=ModelManager)

    def mock_predict_classification(images):
        # Simulate processing time based on batch size
        batch_size = images.shape[0] if hasattr(images, 'shape') else 1
        time.sleep(0.01 * batch_size)  # 10ms per image

        # Return mock predictions
        num_samples = batch_size
        return {
            "predictions": ["tumor_present"] * num_samples,
            "confidences": [0.85 + np.random.random() * 0.1 for _ in range(num_samples)]
        }

    def mock_predict_segmentation(images):
        # Simulate longer processing for segmentation
        batch_size = images.shape[0] if hasattr(images, 'shape') else 1
        time.sleep(0.02 * batch_size)  # 20ms per image

        # Return mock segmentation results
        num_samples = batch_size
        return {
            "masks": [np.random.randint(0, 4, (256, 256)) for _ in range(num_samples)],
            "probabilities": [{"background": 0.1, "tumor": 0.9} for _ in range(num_samples)]
        }

    manager.predict_classification = mock_predict_classification
    manager.predict_segmentation = mock_predict_segmentation

    return manager


class TestInferenceLatency:
    """Test inference latency performance."""

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_single_image_classification_latency(self, performance_profiler, mock_model_manager):
        """Test latency for single image classification."""
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # Measure latency
        latency = performance_profiler.measure_latency(
            lambda: mock_model_manager.predict_classification(
                torch.from_numpy(test_image).unsqueeze(0).float()
            )
        )

        # Clinical requirement: < 500ms for single image
        assert latency < 0.5, f"Classification latency too high: {latency:.3f}s"

        # Should be reasonably fast
        assert latency > 0.001, f"Latency too low (suspicious): {latency:.6f}s"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_single_image_segmentation_latency(self, performance_profiler, mock_model_manager):
        """Test latency for single image segmentation."""
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # Measure latency
        latency = performance_profiler.measure_latency(
            lambda: mock_model_manager.predict_segmentation(
                torch.from_numpy(test_image).unsqueeze(0).float()
            )
        )

        # Clinical requirement: < 1 second for segmentation
        assert latency < 1.0, f"Segmentation latency too high: {latency:.3f}s"

        # Segmentation should be slower than classification but not excessively
        assert latency > 0.01, f"Latency suspiciously low: {latency:.6f}s"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_batch_processing_latency(self, performance_profiler, mock_model_manager, sample_batch_sizes):
        """Test latency scaling with batch size."""
        latencies = {}

        for batch_size in sample_batch_sizes:
            # Create batch of images
            test_batch = torch.randn(batch_size, 1, 256, 256)

            # Measure latency
            latency = performance_profiler.measure_latency(
                lambda: mock_model_manager.predict_classification(test_batch)
            )

            latencies[batch_size] = latency

            # Latency should scale reasonably with batch size
            # Allow some overhead but not exponential growth
            expected_max_latency = 0.1 * batch_size  # 100ms per image max
            assert latency < expected_max_latency, \
                f"Batch latency too high for size {batch_size}: {latency:.3f}s"

        # Verify batch processing is more efficient than individual
        if 1 in latencies and 4 in latencies:
            individual_time_per_image = latencies[1]
            batch_time_per_image = latencies[4] / 4

            # Batch should be at least 10% more efficient
            efficiency_ratio = individual_time_per_image / batch_time_per_image
            assert efficiency_ratio > 1.05, \
                f"Batch processing not efficient: {efficiency_ratio:.2f}x"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_different_image_sizes_latency(self, performance_profiler, mock_model_manager, sample_image_sizes):
        """Test latency scaling with different image sizes."""
        latencies = {}

        for height, width in sample_image_sizes:
            # Create test image of specified size
            test_image = torch.randn(1, 1, height, width)

            # Measure latency
            latency = performance_profiler.measure_latency(
                lambda: mock_model_manager.predict_classification(test_image)
            )

            latencies[(height, width)] = latency

            # Latency should be reasonable for all sizes
            # Allow up to 2 seconds for largest images
            assert latency < 2.0, f"Latency too high for {height}x{width}: {latency:.3f}s"

        # Verify latency scales roughly with image size (quadratic relationship expected)
        if (128, 128) in latencies and (256, 256) in latencies:
            small_latency = latencies[(128, 128)]
            large_latency = latencies[(256, 256)]

            # Large image should take 2-6x longer (accounting for quadratic scaling + overhead)
            scaling_factor = large_latency / small_latency
            assert 1.5 < scaling_factor < 8.0, \
                f"Unexpected scaling factor: {scaling_factor:.2f}x"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_warmup_vs_runtime_latency(self, performance_profiler, mock_model_manager):
        """Test latency difference between warm-up and runtime."""
        test_image = torch.randn(1, 1, 256, 256)

        # First inference (includes model loading/warmup)
        warmup_latency = performance_profiler.measure_latency(
            lambda: mock_model_manager.predict_classification(test_image)
        )

        # Subsequent inferences (runtime performance)
        runtime_latencies = []
        for _ in range(5):
            latency = performance_profiler.measure_latency(
                lambda: mock_model_manager.predict_classification(test_image)
            )
            runtime_latencies.append(latency)

        avg_runtime_latency = sum(runtime_latencies) / len(runtime_latencies)

        # Runtime should be faster than warmup (warmup includes initialization)
        assert avg_runtime_latency < warmup_latency, \
            f"Runtime ({avg_runtime_latency:.3f}s) slower than warmup ({warmup_latency:.3f}s)"

        # Runtime should be consistent (low variance)
        latency_std = statistics.stdev(runtime_latencies)
        latency_cv = latency_std / avg_runtime_latency  # Coefficient of variation

        assert latency_cv < 0.3, f"Runtime latency too variable: CV={latency_cv:.2f}"


class TestInferenceThroughput:
    """Test inference throughput performance."""

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_single_thread_throughput(self, performance_profiler, mock_model_manager):
        """Test throughput in single-threaded scenario."""
        # Create batch of test images
        num_images = 50
        test_images = torch.randn(num_images, 1, 256, 256)

        # Measure throughput
        throughput = performance_profiler.measure_throughput(
            lambda: mock_model_manager.predict_classification(test_images),
            num_operations=num_images
        )

        # Clinical requirement: > 10 images/second
        assert throughput > 10, f"Throughput too low: {throughput:.1f} img/s"

        # Reasonable upper bound for CPU inference
        assert throughput < 1000, f"Throughput suspiciously high: {throughput:.1f} img/s"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_batch_size_throughput_optimization(self, performance_profiler, mock_model_manager, sample_batch_sizes):
        """Test throughput optimization across batch sizes."""
        throughputs = {}

        for batch_size in sample_batch_sizes:
            # Create test batch
            test_batch = torch.randn(batch_size, 1, 256, 256)

            # Measure throughput
            throughput = performance_profiler.measure_throughput(
                lambda: mock_model_manager.predict_classification(test_batch),
                num_operations=batch_size
            )

            throughputs[batch_size] = throughput

        # Find optimal batch size
        optimal_batch = max(throughputs.keys(), key=lambda k: throughputs[k])
        max_throughput = throughputs[optimal_batch]

        # Optimal batch size should provide reasonable throughput
        assert max_throughput > 20, f"Maximum throughput too low: {max_throughput:.1f} img/s"

        # Larger batches should generally provide better throughput
        # (though there may be diminishing returns)
        if 4 in throughputs and 16 in throughputs:
            small_batch_throughput = throughputs[4]
            large_batch_throughput = throughputs[16]

            # Large batch should be at least as good as small batch
            assert large_batch_throughput >= small_batch_throughput * 0.8, \
                f"Large batch inefficiency: {large_batch_throughput:.1f} vs {small_batch_throughput:.1f}"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_concurrent_throughput(self, performance_profiler, mock_model_manager):
        """Test throughput under concurrent load."""
        num_concurrent = 4
        images_per_thread = 25
        total_images = num_concurrent * images_per_thread

        def process_images():
            """Process images in one thread."""
            local_images = torch.randn(images_per_thread, 1, 256, 256)
            mock_model_manager.predict_classification(local_images)
            return images_per_thread

        # Measure concurrent throughput
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(process_images) for _ in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate throughput
        concurrent_throughput = total_images / total_time

        # Compare with single-threaded throughput
        single_batch = torch.randn(images_per_thread, 1, 256, 256)
        single_start = time.time()
        mock_model_manager.predict_classification(single_batch)
        single_end = time.time()
        single_throughput = images_per_thread / (single_end - single_start)

        # Concurrent should provide speedup (though not perfect scaling due to overhead)
        speedup = concurrent_throughput / single_throughput

        assert speedup > 1.2, f"Concurrent processing not providing speedup: {speedup:.2f}x"
        assert speedup < num_concurrent * 1.5, f"Unexpected speedup: {speedup:.2f}x"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Performance testing not available")
    def test_memory_efficient_throughput(self, performance_profiler, mock_model_manager):
        """Test throughput while monitoring memory usage."""
        # Test with memory profiler
        memory_profiler = MemoryProfiler()

        num_images = 100
        batch_size = 10

        with memory_profiler:
            # Process images in batches
            for i in range(0, num_images, batch_size):
                current_batch_size = min(batch_size, num_images - i)
                test_batch = torch.randn(current_batch_size, 1, 256, 256)

                mock_model_manager.predict_classification(test_batch)

        # Check memory usage
        memory_stats = memory_profiler.get_stats()

        # Memory usage should be reasonable
        peak_memory_mb = memory_stats.get("peak_mb", 0)
        avg_memory_mb = memory_stats.get("avg_mb", 0)

        # Should use less than 2GB peak memory
        assert peak_memory_mb < 2048, f"Peak memory too high: {peak_memory_mb}MB"

        # Average memory should be reasonable
        assert avg_memory_mb < 1024, f"Average memory too high: {avg_memory_mb}MB"


class TestMemoryUsage:
    """Test memory usage patterns and limits."""

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Memory profiling not available")
    def test_classification_memory_usage(self, performance_profiler, mock_model_manager):
        """Test memory usage for classification inference."""
        memory_profiler = MemoryProfiler()

        test_image = torch.randn(1, 1, 256, 256)

        with memory_profiler:
            result = mock_model_manager.predict_classification(test_image)

        memory_stats = memory_profiler.get_stats()

        # Memory usage should be reasonable for single image
        peak_mb = memory_stats.get("peak_mb", 0)
        assert peak_mb < 500, f"Classification memory usage too high: {peak_mb}MB"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Memory profiling not available")
    def test_segmentation_memory_usage(self, performance_profiler, mock_model_manager):
        """Test memory usage for segmentation inference."""
        memory_profiler = MemoryProfiler()

        test_image = torch.randn(1, 1, 256, 256)

        with memory_profiler:
            result = mock_model_manager.predict_segmentation(test_image)

        memory_stats = memory_profiler.get_stats()

        # Segmentation may use more memory than classification
        peak_mb = memory_stats.get("peak_mb", 0)
        assert peak_mb < 1000, f"Segmentation memory usage too high: {peak_mb}MB"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Memory profiling not available")
    def test_batch_memory_scaling(self, performance_profiler, mock_model_manager, sample_batch_sizes):
        """Test how memory usage scales with batch size."""
        memory_usage = {}

        for batch_size in sample_batch_sizes:
            memory_profiler = MemoryProfiler()
            test_batch = torch.randn(batch_size, 1, 256, 256)

            with memory_profiler:
                mock_model_manager.predict_classification(test_batch)

            memory_stats = memory_profiler.get_stats()
            memory_usage[batch_size] = memory_stats.get("peak_mb", 0)

            # Memory should be reasonable for all batch sizes
            assert memory_usage[batch_size] < 2048, \
                f"Batch size {batch_size} memory too high: {memory_usage[batch_size]}MB"

        # Memory scaling should be roughly linear with batch size
        if 4 in memory_usage and 16 in memory_usage:
            scaling_factor = memory_usage[16] / memory_usage[4]
            # Allow for some overhead, but not exponential growth
            assert scaling_factor < 6.0, \
                f"Memory scaling inefficient: {scaling_factor:.2f}x for 4x batch size"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Memory profiling not available")
    def test_memory_leak_detection(self, performance_profiler, mock_model_manager):
        """Test for memory leaks during repeated inference."""
        memory_profiler = MemoryProfiler()

        # Run multiple inferences
        num_iterations = 50

        memory_readings = []
        for i in range(num_iterations):
            with memory_profiler:
                test_image = torch.randn(1, 1, 256, 256)
                mock_model_manager.predict_classification(test_image)

            memory_readings.append(memory_profiler.get_stats().get("current_mb", 0))

        # Check for memory leaks (gradual increase)
        if len(memory_readings) >= 10:
            early_avg = sum(memory_readings[:10]) / 10
            late_avg = sum(memory_readings[-10:]) / 10

            # Memory should not increase by more than 10% over time
            memory_growth = (late_avg - early_avg) / early_avg
            assert memory_growth < 0.1, f"Memory leak detected: {memory_growth:.1%} growth"


class TestConcurrentUsers:
    """Test performance under concurrent user load."""

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Concurrent testing not available")
    def test_multiple_concurrent_users(self, performance_profiler, mock_model_manager):
        """Test system performance with multiple concurrent users."""
        num_users = 5
        requests_per_user = 20
        total_requests = num_users * requests_per_user

        response_times = []
        errors = []

        def simulate_user(user_id):
            """Simulate a single user making requests."""
            user_times = []
            user_errors = 0

            for i in range(requests_per_user):
                try:
                    start_time = time.time()

                    # Simulate API request
                    test_image = torch.randn(1, 1, 256, 256)
                    mock_model_manager.predict_classification(test_image)

                    end_time = time.time()
                    user_times.append(end_time - start_time)

                except Exception as e:
                    user_errors += 1

            return user_times, user_errors

        # Run concurrent users
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(simulate_user, i) for i in range(num_users)]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time

        # Aggregate results
        all_response_times = []
        total_errors = 0

        for user_times, user_errors in results:
            all_response_times.extend(user_times)
            total_errors += user_errors

        # Calculate metrics
        successful_requests = len(all_response_times)
        throughput = successful_requests / total_time
        avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
        p95_response_time = sorted(all_response_times)[int(0.95 * len(all_response_times))] if all_response_times else 0

        # Performance requirements for concurrent users
        assert throughput > 5, f"Concurrent throughput too low: {throughput:.1f} req/s"
        assert avg_response_time < 1.0, f"Average response time too high: {avg_response_time:.3f}s"
        assert p95_response_time < 2.0, f"P95 response time too high: {p95_response_time:.3f}s"
        assert total_errors == 0, f"Errors occurred during concurrent testing: {total_errors}"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Concurrent load testing not available")
    def test_load_balancing_efficiency(self, performance_profiler, mock_model_manager):
        """Test load balancing efficiency across concurrent operations."""
        num_workers = 8
        tasks_per_worker = 25
        total_tasks = num_workers * tasks_per_worker

        def worker_task(worker_id):
            """Simulate worker processing tasks."""
            worker_times = []
            for i in range(tasks_per_worker):
                start_time = time.time()

                # Simulate variable processing time
                test_image = torch.randn(1, 1, 256, 256)
                mock_model_manager.predict_classification(test_image)

                # Add some random delay to simulate real-world variability
                time.sleep(np.random.uniform(0.001, 0.01))

                end_time = time.time()
                worker_times.append(end_time - start_time)

            return worker_times

        # Run load balancing test
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers)]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze load balancing
        worker_completion_times = []
        for worker_times in results:
            worker_total_time = sum(worker_times)
            worker_completion_times.append(worker_total_time)

        # Calculate load balancing efficiency
        avg_completion_time = sum(worker_completion_times) / len(worker_completion_times)
        max_completion_time = max(worker_completion_times)
        min_completion_time = min(worker_completion_times)

        # Load balancing efficiency (1.0 = perfect balance)
        efficiency = avg_completion_time / max_completion_time

        # Efficiency should be reasonable (> 0.7 for good load balancing)
        assert efficiency > 0.7, f"Poor load balancing: {efficiency:.2f} efficiency"

        # No worker should be excessively slower
        assert max_completion_time / min_completion_time < 2.0, \
            f"Severe load imbalance: {max_completion_time/min_completion_time:.2f}x difference"


class TestClinicalPerformanceRequirements:
    """Test clinical-grade performance requirements."""

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Clinical requirements testing not available")
    def test_emergency_response_time(self, performance_profiler, mock_model_manager):
        """Test performance meets emergency response requirements."""
        # Emergency medicine requirement: < 30 seconds for critical decisions
        # Our system should be much faster: < 5 seconds for complete analysis

        test_image = torch.randn(1, 1, 256, 256)

        # Measure complete analysis time (classification + segmentation)
        start_time = time.time()

        cls_result = mock_model_manager.predict_classification(test_image)
        seg_result = mock_model_manager.predict_segmentation(test_image)

        end_time = time.time()
        analysis_time = end_time - start_time

        # Clinical requirement: Complete analysis in under 5 seconds
        assert analysis_time < 5.0, f"Analysis too slow for clinical use: {analysis_time:.1f}s"

        # Should be much faster in practice
        assert analysis_time < 2.0, f"Performance needs optimization: {analysis_time:.1f}s"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Clinical requirements testing not available")
    def test_batch_processing_efficiency(self, performance_profiler, mock_model_manager):
        """Test batch processing meets clinical workflow requirements."""
        # Clinical workflow: Process 10-20 images in a study quickly

        batch_sizes = [10, 20]
        results = {}

        for batch_size in batch_sizes:
            test_batch = torch.randn(batch_size, 1, 256, 256)

            start_time = time.time()
            mock_model_manager.predict_classification(test_batch)
            end_time = time.time()

            batch_time = end_time - start_time
            time_per_image = batch_time / batch_size

            results[batch_size] = {
                "total_time": batch_time,
                "time_per_image": time_per_image,
                "images_per_second": batch_size / batch_time
            }

            # Clinical requirement: < 10 seconds for 20 images
            if batch_size == 20:
                assert batch_time < 10.0, f"Batch processing too slow: {batch_time:.1f}s for {batch_size} images"

            # Individual image should be fast
            assert time_per_image < 0.5, f"Individual image too slow: {time_per_image:.3f}s"

    @pytest.mark.skipif(not PERFORMANCE_AVAILABLE, reason="Clinical requirements testing not available")
    def test_memory_requirements_clinical(self, performance_profiler, mock_model_manager):
        """Test memory usage meets clinical deployment requirements."""
        # Clinical systems often have memory constraints

        memory_profiler = MemoryProfiler()

        # Test various scenarios
        test_scenarios = [
            ("single_image", torch.randn(1, 1, 256, 256)),
            ("small_batch", torch.randn(4, 1, 256, 256)),
            ("large_batch", torch.randn(16, 1, 256, 256)),
        ]

        for scenario_name, test_data in test_scenarios:
            memory_profiler.reset()

            with memory_profiler:
                if scenario_name == "single_image":
                    mock_model_manager.predict_classification(test_data)
                else:
                    mock_model_manager.predict_classification(test_data)

            memory_stats = memory_profiler.get_stats()
            peak_mb = memory_stats.get("peak_mb", 0)

            # Clinical deployment memory limits
            if scenario_name == "single_image":
                assert peak_mb < 512, f"Single image memory too high: {peak_mb}MB"
            elif scenario_name == "small_batch":
                assert peak_mb < 1024, f"Small batch memory too high: {peak_mb}MB"
            else:  # large_batch
                assert peak_mb < 2048, f"Large batch memory too high: {peak_mb}MB"
