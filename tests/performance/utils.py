"""
Performance Testing Utilities for SliceWise.

This module provides profiling utilities for measuring latency, throughput,
and memory usage during inference testing.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, Any, Callable, List, Optional
from contextlib import contextmanager
import gc


class PerformanceProfiler:
    """
    Comprehensive performance profiler for inference testing.
    
    Measures latency, throughput, and provides timing utilities.
    """
    
    def __init__(self):
        """Initialize performance profiler."""
        self.measurements = []
        
    def measure_latency(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> float:
        """
        Measure latency of a function call.
        
        Args:
            func: Function to measure
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Latency in seconds
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> latency = profiler.measure_latency(model.predict, image)
            >>> print(f"Latency: {latency*1000:.2f}ms")
        """
        # Warm up if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        latency = end_time - start_time
        
        self.measurements.append({
            'type': 'latency',
            'value': latency,
            'timestamp': time.time()
        })
        
        return latency
    
    def measure_throughput(
        self,
        func: Callable,
        num_samples: int = None,
        num_operations: int = None,
        *args,
        **kwargs
    ) -> float:
        """
        Measure throughput (samples/second).
        
        Args:
            func: Function to measure
            num_samples: Number of samples processed (preferred)
            num_operations: Alias for num_samples (for backward compatibility)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Throughput in samples/second
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> throughput = profiler.measure_throughput(model.predict_batch, num_samples=100)
            >>> print(f"Throughput: {throughput:.1f} images/sec")
        """
        # Support both num_samples and num_operations
        count = num_samples or num_operations
        if count is None:
            raise ValueError("Either num_samples or num_operations must be provided")
        
        latency = self.measure_latency(func, *args, **kwargs)
        throughput = count / latency if latency > 0 else 0
        
        self.measurements.append({
            'type': 'throughput',
            'value': throughput,
            'num_samples': count,
            'timestamp': time.time()
        })
        
        return throughput
    
    def measure_memory_usage(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """
        Measure memory usage during function execution.
        
        Args:
            func: Function to measure
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Dictionary with memory statistics (peak_mb, avg_mb, delta_mb)
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> mem_stats = profiler.measure_memory_usage(model.predict, image)
            >>> print(f"Peak memory: {mem_stats['peak_mb']:.1f}MB")
        """
        # Clear cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Get initial memory
        process = psutil.Process()
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            initial_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Run function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_mem = process.memory_info().rss / 1024 / 1024  # MB
        peak_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_stats = {
            'peak_mb': peak_mem,
            'avg_mb': (initial_mem + final_mem) / 2,
            'delta_mb': final_mem - initial_mem,
            'initial_mb': initial_mem,
            'final_mb': final_mem
        }
        
        if torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            gpu_current = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_stats.update({
                'gpu_peak_mb': gpu_peak,
                'gpu_current_mb': gpu_current,
                'gpu_delta_mb': gpu_current - initial_gpu_mem
            })
        
        self.measurements.append({
            'type': 'memory',
            'value': memory_stats,
            'timestamp': time.time()
        })
        
        return memory_stats
    
    @contextmanager
    def profile_context(self, name: str = "operation"):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name of the operation being profiled
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> with profiler.profile_context("inference"):
            >>>     result = model.predict(image)
        """
        start_time = time.perf_counter()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        self.measurements.append({
            'type': 'context',
            'name': name,
            'value': elapsed,
            'timestamp': time.time()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all measurements.
        
        Returns:
            Dictionary with aggregated statistics
        """
        if not self.measurements:
            return {}
        
        latencies = [m['value'] for m in self.measurements if m['type'] == 'latency']
        throughputs = [m['value'] for m in self.measurements if m['type'] == 'throughput']
        
        stats = {}
        
        if latencies:
            stats['latency'] = {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
            }
        
        if throughputs:
            stats['throughput'] = {
                'mean': np.mean(throughputs),
                'std': np.std(throughputs),
                'min': np.min(throughputs),
                'max': np.max(throughputs),
            }
        
        return stats
    
    def reset(self):
        """Reset all measurements."""
        self.measurements = []


class MemoryProfiler:
    """
    Memory profiler for tracking CPU and GPU memory usage.
    
    Monitors memory consumption during inference and detects leaks.
    """
    
    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots = []
        self.process = psutil.Process()
        
    def take_snapshot(self, label: str = None) -> Dict[str, float]:
        """
        Take a memory snapshot.
        
        Args:
            label: Optional label for the snapshot
            
        Returns:
            Dictionary with memory usage statistics
        """
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'cpu_mb': self.process.memory_info().rss / 1024 / 1024,
            'cpu_percent': self.process.memory_percent()
        }
        
        if torch.cuda.is_available():
            snapshot.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            })
        
        self.snapshots.append(snapshot)
        return snapshot
    
    @contextmanager
    def profile(self, label: str = "operation"):
        """
        Context manager for profiling memory usage.
        
        Args:
            label: Label for the profiled operation
            
        Example:
            >>> profiler = MemoryProfiler()
            >>> with profiler.profile("inference"):
            >>>     result = model.predict(image)
            >>> stats = profiler.get_statistics()
        """
        # Clear cache before profiling
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Take initial snapshot
        self.take_snapshot(f"{label}_start")
        
        yield
        
        # Take final snapshot
        self.take_snapshot(f"{label}_end")
    
    def detect_leak(self, threshold_mb: float = 100.0) -> bool:
        """
        Detect potential memory leaks.
        
        Args:
            threshold_mb: Memory increase threshold in MB
            
        Returns:
            True if potential leak detected
        """
        if len(self.snapshots) < 2:
            return False
        
        initial = self.snapshots[0]['cpu_mb']
        final = self.snapshots[-1]['cpu_mb']
        increase = final - initial
        
        return increase > threshold_mb
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with aggregated memory statistics
        """
        if not self.snapshots:
            return {'peak_mb': 0, 'current_mb': 0, 'delta_mb': 0}
        
        cpu_usage = [s['cpu_mb'] for s in self.snapshots]
        
        stats = {
            'peak_mb': np.max(cpu_usage),
            'current_mb': cpu_usage[-1] if cpu_usage else 0,
            'delta_mb': cpu_usage[-1] - cpu_usage[0] if len(cpu_usage) > 0 else 0,
            'cpu': {
                'mean_mb': np.mean(cpu_usage),
                'peak_mb': np.max(cpu_usage),
                'min_mb': np.min(cpu_usage),
                'delta_mb': cpu_usage[-1] - cpu_usage[0]
            }
        }
        
        if torch.cuda.is_available() and 'gpu_allocated_mb' in self.snapshots[0]:
            gpu_usage = [s['gpu_allocated_mb'] for s in self.snapshots]
            stats['gpu'] = {
                'mean_mb': np.mean(gpu_usage),
                'peak_mb': np.max(gpu_usage),
                'min_mb': np.min(gpu_usage),
                'delta_mb': gpu_usage[-1] - gpu_usage[0]
            }
        
        return stats
    
    def reset(self):
        """Reset all snapshots."""
        self.snapshots = []
    
    def __enter__(self):
        """Enter context manager - take initial snapshot."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        self.take_snapshot("context_start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - take final snapshot."""
        self.take_snapshot("context_end")
        return False  # Don't suppress exceptions
    
    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_statistics() for backward compatibility."""
        return self.get_statistics()


class LatencyProfiler:
    """
    Specialized profiler for latency measurements.
    
    Provides detailed latency analysis with percentiles and distributions.
    """
    
    def __init__(self):
        """Initialize latency profiler."""
        self.latencies = []
        
    def measure(self, func: Callable, *args, **kwargs) -> float:
        """
        Measure latency of a single function call.
        
        Args:
            func: Function to measure
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Latency in seconds
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        latency = end - start
        
        self.latencies.append(latency)
        return latency
    
    def measure_multiple(
        self,
        func: Callable,
        num_iterations: int,
        *args,
        **kwargs
    ) -> List[float]:
        """
        Measure latency over multiple iterations.
        
        Args:
            func: Function to measure
            num_iterations: Number of iterations
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            List of latencies
        """
        latencies = []
        for _ in range(num_iterations):
            latency = self.measure(func, *args, **kwargs)
            latencies.append(latency)
        
        return latencies
    
    def get_percentiles(self, percentiles: List[float] = None) -> Dict[str, float]:
        """
        Get latency percentiles.
        
        Args:
            percentiles: List of percentiles to compute (default: [50, 90, 95, 99])
            
        Returns:
            Dictionary mapping percentile to latency value
        """
        if not self.latencies:
            return {}
        
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        
        return {
            f'p{int(p)}': np.percentile(self.latencies, p)
            for p in percentiles
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive latency statistics.
        
        Returns:
            Dictionary with mean, std, min, max, and percentiles
        """
        if not self.latencies:
            return {}
        
        stats = {
            'mean': np.mean(self.latencies),
            'std': np.std(self.latencies),
            'min': np.min(self.latencies),
            'max': np.max(self.latencies),
            'count': len(self.latencies)
        }
        
        stats.update(self.get_percentiles())
        
        return stats
    
    def reset(self):
        """Reset all latency measurements."""
        self.latencies = []


# Export all profiler classes
__all__ = [
    'PerformanceProfiler',
    'MemoryProfiler',
    'LatencyProfiler',
]
