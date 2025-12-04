"""
Smoke tests to verify basic functionality.
"""

import pytest
import torch
import numpy as np


def test_pytorch_import():
    """Test that PyTorch is installed and working."""
    assert torch.__version__ is not None
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0


def test_cuda_availability():
    """Test CUDA availability (will pass on CPU-only systems)."""
    # Just check that the function works, don't require CUDA
    cuda_available = torch.cuda.is_available()
    assert isinstance(cuda_available, bool)


def test_numpy_import():
    """Test that NumPy is installed and working."""
    assert np.__version__ is not None
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6


def test_basic_conv2d():
    """Test basic Conv2d operation."""
    conv = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
    x = torch.randn(1, 1, 32, 32)
    y = conv(x)
    assert y.shape == (1, 16, 32, 32)


def test_basic_unet_forward():
    """Test a minimal U-Net-like forward pass."""
    
    class MiniUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Conv2d(1, 16, 3, padding=1)
            self.dec = torch.nn.Conv2d(16, 1, 3, padding=1)
        
        def forward(self, x):
            x = self.enc(x)
            x = torch.relu(x)
            x = self.dec(x)
            return x
    
    model = MiniUNet()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_src_package_exists():
    """Test that src package can be imported."""
    try:
        import src
        assert hasattr(src, '__version__')
    except ImportError:
        pytest.skip("src package not installed in editable mode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
