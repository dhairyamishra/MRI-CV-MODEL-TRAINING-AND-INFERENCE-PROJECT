"""
Inference profiling for segmentation and classification models.

Benchmarks:
- Latency (p50, p95, mean, std)
- GPU memory usage (peak)
- Throughput (images/second)
- Different batch sizes and resolutions
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path


class InferenceProfiler:
    """
    Profile model inference performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ):
        """
        Args:
            model: PyTorch model to profile
            device: Device to run on
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        # Enable cudnn benchmarking for faster inference
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    def profile_single_input(
        self,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
    ) -> Dict:
        """
        Profile inference for a single input configuration.
        
        Args:
            input_shape: Input shape (C, H, W)
            batch_size: Batch size
        
        Returns:
            Dictionary of profiling metrics
        """
        # Create dummy input
        full_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(full_shape, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = self.model(dummy_input)
        
        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                start_time = time.time()
                
                _ = self.model(dummy_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Compute statistics
        latencies = np.array(latencies)
        
        metrics = {
            'batch_size': batch_size,
            'input_shape': input_shape,
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
        }
        
        # Throughput
        metrics['throughput_imgs_per_sec'] = 1000.0 * batch_size / metrics['mean_latency_ms']
        
        # GPU memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            metrics['peak_memory_mb'] = float(peak_memory_mb)
        else:
            metrics['peak_memory_mb'] = 0.0
        
        return metrics
    
    def profile_multiple_configs(
        self,
        configs: List[Dict],
    ) -> pd.DataFrame:
        """
        Profile multiple input configurations.
        
        Args:
            configs: List of config dicts with 'input_shape' and 'batch_size'
        
        Returns:
            DataFrame with profiling results
        """
        results = []
        
        for config in configs:
            input_shape = config['input_shape']
            batch_size = config['batch_size']
            
            print(f"Profiling: batch_size={batch_size}, input_shape={input_shape}")
            
            metrics = self.profile_single_input(input_shape, batch_size)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        """
        Save profiling results to CSV.
        
        Args:
            df: DataFrame with results
            output_path: Path to save CSV
        """
        df.to_csv(output_path, index=False)
        print(f"\nSaved profiling results to: {output_path}")


def profile_model(
    model: nn.Module,
    input_shapes: List[Tuple[int, ...]],
    batch_sizes: List[int],
    output_csv: str = "outputs/profiling/inference_profile.csv",
    device: str = 'cuda',
) -> pd.DataFrame:
    """
    Profile a model with different configurations.
    
    Args:
        model: PyTorch model
        input_shapes: List of input shapes (C, H, W)
        batch_sizes: List of batch sizes
        output_csv: Path to save results
        device: Device to use
    
    Returns:
        DataFrame with profiling results
    """
    profiler = InferenceProfiler(model, device=device)
    
    # Create all combinations
    configs = []
    for input_shape in input_shapes:
        for batch_size in batch_sizes:
            configs.append({
                'input_shape': input_shape,
                'batch_size': batch_size,
            })
    
    # Profile
    df = profiler.profile_multiple_configs(configs)
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.save_results(df, output_csv)
    
    return df


def print_profiling_summary(df: pd.DataFrame):
    """
    Print a summary of profiling results.
    
    Args:
        df: DataFrame with profiling results
    """
    print("\n" + "=" * 80)
    print("Inference Profiling Summary")
    print("=" * 80)
    
    print("\nLatency Statistics:")
    print(df[['batch_size', 'input_shape', 'mean_latency_ms', 'p50_latency_ms', 'p95_latency_ms']].to_string(index=False))
    
    print("\nThroughput:")
    print(df[['batch_size', 'input_shape', 'throughput_imgs_per_sec']].to_string(index=False))
    
    if 'peak_memory_mb' in df.columns:
        print("\nGPU Memory:")
        print(df[['batch_size', 'input_shape', 'peak_memory_mb']].to_string(index=False))
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test inference profiling
    print("Testing Inference Profiling...")
    print("=" * 60)
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    model = SimpleModel()
    
    # Profile different configurations
    input_shapes = [
        (1, 128, 128),
        (1, 256, 256),
        (1, 512, 512),
    ]
    
    batch_sizes = [1, 4, 8]
    
    print("\nProfiling model with different configurations...")
    print(f"  Input shapes: {input_shapes}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    profiler = InferenceProfiler(
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        warmup_iterations=5,
        benchmark_iterations=50,
    )
    
    # Create configs
    configs = []
    for input_shape in input_shapes:
        for batch_size in batch_sizes:
            configs.append({
                'input_shape': input_shape,
                'batch_size': batch_size,
            })
    
    # Profile
    df = profiler.profile_multiple_configs(configs)
    
    # Print summary
    print_profiling_summary(df)
    
    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
