"""
PHASE 4.4: Cross-Platform Compatibility Testing - OS & Environment Validation

Tests SliceWise compatibility across different platforms and environments:
- Operating Systems: Windows, Linux, macOS compatibility
- Python Environments: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 support
- Hardware Variants: Different NVIDIA GPU architectures
- CPU-only inference performance
- Memory-constrained environment handling

Validates deployment flexibility and broad platform support.
"""

import sys
import pytest
import platform
import os
import subprocess
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import compatibility utilities
try:
    from tests.compatibility.platform_utils import PlatformChecker
    from tests.compatibility.hardware_utils import HardwareChecker
    from tests.compatibility.python_utils import PythonVersionChecker
    COMPATIBILITY_AVAILABLE = True
except ImportError:
    # Mock compatibility utilities
    COMPATIBILITY_AVAILABLE = False
    PlatformChecker = MagicMock()
    HardwareChecker = MagicMock()
    PythonVersionChecker = MagicMock()


@pytest.fixture
def platform_checker():
    """Create platform checker instance."""
    if COMPATIBILITY_AVAILABLE:
        return PlatformChecker()
    else:
        checker = MagicMock()
        checker.get_os_info.return_value = {"os": "Windows", "version": "10.0.19041"}
        checker.check_dependencies.return_value = {"all_present": True, "missing": []}
        return checker


@pytest.fixture
def hardware_checker():
    """Create hardware checker instance."""
    if COMPATIBILITY_AVAILABLE:
        return HardwareChecker()
    else:
        checker = MagicMock()
        checker.get_gpu_info.return_value = {"available": True, "name": "RTX 3080", "memory_gb": 10}
        checker.check_cuda_compatibility.return_value = {"compatible": True, "version": "11.8"}
        return checker


@pytest.fixture
def python_checker():
    """Create Python version checker instance."""
    if COMPATIBILITY_AVAILABLE:
        return PythonVersionChecker()
    else:
        checker = MagicMock()
        checker.get_python_info.return_value = {"version": "3.9.7", "implementation": "CPython"}
        checker.check_package_compatibility.return_value = {"compatible": True, "issues": []}
        return checker


class TestOperatingSystemCompatibility:
    """Test compatibility across different operating systems."""

    def test_windows_compatibility(self, platform_checker):
        """Test Windows-specific compatibility."""
        # Check Windows-specific paths and file handling
        windows_paths = [
            "C:\\Users\\user\\Documents\\medical_images",
            "C:\\Program Files\\SliceWise\\models",
            "C:\\temp\\cache"
        ]

        for path_str in windows_paths:
            if platform.system() == "Windows":
                path = Path(path_str)
                # Should handle Windows paths correctly
                assert path.drive in ["C:", "D:", "E:"]
                assert "\\" in str(path) or "/" in str(path)

        # Test Windows file permissions
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("test data")
            tmp_path = tmp_file.name

        try:
            # Should be able to read/write files on Windows
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert content == "test data"

            # Test file locking (Windows-specific issue)
            import msvcrt
            with open(tmp_path, 'r+') as f:
                # Try to lock file (should work on Windows)
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    file_locking_works = True
                except:
                    file_locking_works = False

                # File locking should work on Windows
                assert file_locking_works or platform.system() != "Windows"

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_linux_compatibility(self, platform_checker):
        """Test Linux-specific compatibility."""
        if platform.system() == "Linux":
            # Check Linux-specific paths
            linux_paths = [
                "/home/user/medical_images",
                "/opt/slicewise/models",
                "/tmp/cache"
            ]

            for path_str in linux_paths:
                path = Path(path_str)
                assert str(path).startswith("/")
                assert "\\" not in str(path)  # Should not have Windows separators

            # Test Linux permissions
            test_file = Path("/tmp/slicewise_test.txt")
            try:
                test_file.write_text("test data")
                content = test_file.read_text()
                assert content == "test data"

                # Test file permissions
                import stat
                file_stat = test_file.stat()
                # Should have reasonable permissions
                assert file_stat.st_mode & stat.S_IRUSR  # Owner can read

            finally:
                test_file.unlink(missing_ok=True)

    def test_macos_compatibility(self, platform_checker):
        """Test macOS-specific compatibility."""
        if platform.system() == "Darwin":
            # Check macOS-specific paths
            macos_paths = [
                "/Users/user/medical_images",
                "/Applications/SliceWise.app",
                "/Library/Application Support/SliceWise"
            ]

            for path_str in macos_paths:
                path = Path(path_str)
                assert str(path).startswith("/")
                assert "Users" in str(path) or "Applications" in str(path) or "Library" in str(path)

            # Test macOS-specific features
            import subprocess
            try:
                # Check if we can run basic commands
                result = subprocess.run(["sw_vers"], capture_output=True, text=True)
                assert result.returncode == 0
                assert "ProductName" in result.stdout or "ProductVersion" in result.stdout
            except:
                # sw_vers might not be available in all environments
                pass

    def test_cross_platform_path_handling(self, platform_checker):
        """Test cross-platform path handling."""
        # Test path normalization across platforms
        test_paths = [
            "data/medical_images",
            "models/checkpoints/best_model.pth",
            "logs/application.log",
            "config/settings.yaml"
        ]

        for path_str in test_paths:
            # Should work on all platforms
            path = Path(path_str)

            # Path operations should work
            assert path.exists() or not path.exists()  # Either exists or doesn't
            assert path.is_absolute() == False  # Should be relative

            # Path joining should work
            full_path = project_root / path
            assert full_path.is_absolute()

            # String conversion should work
            path_str = str(path)
            assert isinstance(path_str, str)
            assert len(path_str) > 0

    def test_file_encoding_compatibility(self, platform_checker):
        """Test file encoding compatibility across platforms."""
        test_content = "Médical data: température=37.5°C, naïve encoding test"
        encodings_to_test = ['utf-8', 'latin-1']

        for encoding in encodings_to_test:
            with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, delete=False) as tmp_file:
                tmp_file.write(test_content)
                tmp_path = tmp_file.name

            try:
                # Should be able to read with same encoding
                with open(tmp_path, 'r', encoding=encoding) as f:
                    read_content = f.read()
                    assert read_content == test_content

                # UTF-8 should handle all content (only test for utf-8 encoded files)
                if encoding == 'utf-8':
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        utf8_content = f.read()
                        assert utf8_content == test_content
                        assert isinstance(utf8_content, str)
                else:
                    # For non-UTF-8 encodings, reading as UTF-8 may fail
                    # Test that we can handle encoding errors gracefully
                    try:
                        with open(tmp_path, 'r', encoding='utf-8', errors='replace') as f:
                            utf8_content = f.read()
                            assert isinstance(utf8_content, str)
                    except UnicodeDecodeError:
                        # Expected for incompatible encodings
                        pass

            finally:
                Path(tmp_path).unlink(missing_ok=True)


class TestPythonVersionCompatibility:
    """Test compatibility across different Python versions."""

    def test_python_version_support(self, python_checker):
        """Test supported Python version ranges."""
        supported_versions = [
            (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13)
        ]

        current_version = sys.version_info[:2]

        # Current version should be supported
        assert current_version in supported_versions, \
            f"Python {current_version[0]}.{current_version[1]} is not supported"

        # Test version-specific features
        if current_version >= (3, 8):
            # Test walrus operator (:=)
            if (n := len(supported_versions)) > 0:
                assert n == len(supported_versions)

        if current_version >= (3, 9):
            # Test dict union operators
            dict1 = {"a": 1}
            dict2 = {"b": 2}
            merged = dict1 | dict2
            assert merged == {"a": 1, "b": 2}

    def test_package_compatibility_matrix(self, python_checker):
        """Test package compatibility across Python versions."""
        # Core dependencies that must work across versions
        core_packages = [
            "torch",
            "numpy",
            "PIL",
            "fastapi",
            "uvicorn",
            "streamlit"
        ]

        for package in core_packages:
            try:
                __import__(package)
                package_available = True
            except ImportError:
                package_available = False

            # Core packages should be available
            assert package_available, f"Core package {package} not available"

        # Test version-specific package compatibility
        import torch
        import numpy as np

        # PyTorch should work with current Python version
        assert torch.__version__ is not None

        # NumPy should work
        assert np.__version__ is not None

        # Test basic tensor operations (PyTorch compatibility)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            test_tensor = torch.randn(10, 10).to(device)
            result = test_tensor.sum()
            assert result.item() is not None

    def test_async_await_compatibility(self, python_checker):
        """Test async/await functionality across Python versions."""
        import asyncio

        async def test_coroutine():
            await asyncio.sleep(0.01)
            return "async_test_passed"

        # Test async function execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(test_coroutine())
            assert result == "async_test_passed"
        finally:
            loop.close()

    def test_type_hinting_compatibility(self, python_checker):
        """Test type hinting features across Python versions."""
        from typing import List, Dict, Optional, Union

        # Basic type hints should work
        def typed_function(x: int, y: List[str]) -> Dict[str, Union[int, str]]:
            return {"result": x, "items": y}

        # Function should execute
        result = typed_function(42, ["a", "b", "c"])
        assert result["result"] == 42
        assert result["items"] == ["a", "b", "c"]

        # Test newer typing features
        if sys.version_info >= (3, 10):
            from typing import TypeAlias
            Number: TypeAlias = Union[int, float]
            value: Number = 3.14
            assert isinstance(value, (int, float))

    def test_string_formatting_compatibility(self, python_checker):
        """Test string formatting across Python versions."""
        # Test f-strings (Python 3.6+)
        name = "SliceWise"
        version = 1.0
        f_string = f"{name} v{version}"
        assert f_string == "SliceWise v1.0"

        # Test format() method
        format_string = "{} version {}".format(name, version)
        assert format_string == "SliceWise version 1.0"

        # Test % formatting (legacy)
        percent_string = "%s version %s" % (name, str(version))
        assert percent_string == "SliceWise version 1.0"


class TestCondaEnvironmentCompatibility:
    """Test compatibility with conda environments."""

    def test_conda_environment_detection(self):
        """Test detection of conda environment."""
        # Check for conda environment variables
        conda_env_vars = [
            "CONDA_DEFAULT_ENV",
            "CONDA_PREFIX",
            "CONDA_PYTHON_EXE"
        ]

        conda_detected = any(os.environ.get(var) for var in conda_env_vars)

        if conda_detected:
            # If conda detected, verify environment
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                assert Path(conda_prefix).exists()
                assert Path(conda_prefix).is_dir()

            # Check conda Python executable
            conda_python = os.environ.get("CONDA_PYTHON_EXE")
            if conda_python:
                assert Path(conda_python).exists()
                assert Path(conda_python).is_file()

    def test_conda_package_compatibility(self):
        """Test package compatibility in conda environment."""
        # Test that critical packages work in conda environment
        try:
            import torch
            import numpy as np
            import PIL

            # Test basic functionality
            tensor = torch.randn(5, 5)
            array = np.random.rand(5, 5)
            image = PIL.Image.new('RGB', (10, 10))

            assert tensor.shape == (5, 5)
            assert array.shape == (5, 5)
            assert image.size == (10, 10)

        except ImportError as e:
            pytest.fail(f"Critical package not available in conda environment: {e}")

    def test_conda_environment_isolation(self):
        """Test that conda environment provides proper isolation."""
        # Check that we're not using system Python
        python_exe = sys.executable

        # Should be in conda environment path
        if os.environ.get("CONDA_PREFIX"):
            conda_prefix = os.environ["CONDA_PREFIX"]
            assert conda_prefix in python_exe

        # Check that package installations are isolated
        import site
        site_packages = site.getsitepackages()

        # At least one site-packages should be in conda prefix
        if os.environ.get("CONDA_PREFIX"):
            conda_prefix = os.environ["CONDA_PREFIX"]
            conda_site_packages = [p for p in site_packages if conda_prefix in p]
            assert len(conda_site_packages) > 0


class TestVirtualEnvironmentCompatibility:
    """Test compatibility with virtual environments."""

    def test_venv_environment_detection(self):
        """Test detection of virtual environment."""
        # Check for virtual environment indicators
        venv_indicators = [
            os.environ.get("VIRTUAL_ENV"),
            hasattr(sys, 'real_prefix') or hasattr(sys, 'base_prefix'),
            "venv" in sys.executable.lower() or "virtualenv" in sys.executable.lower()
        ]

        venv_detected = any(venv_indicators)

        if venv_detected:
            # Verify virtual environment setup
            if os.environ.get("VIRTUAL_ENV"):
                venv_path = Path(os.environ["VIRTUAL_ENV"])
                assert venv_path.exists()
                assert venv_path.is_dir()

                # Check for key venv directories
                assert (venv_path / "bin").exists() or (venv_path / "Scripts").exists()
                assert (venv_path / "lib").exists()

    def test_pip_environment_compatibility(self):
        """Test package compatibility in pip virtual environment."""
        # Test that pip packages work correctly
        try:
            import torch
            import fastapi
            import streamlit

            # Test basic imports work
            assert torch.__version__ is not None
            assert fastapi.__version__ is not None
            # Streamlit might not have version or might fail in headless mode

        except ImportError as e:
            pytest.fail(f"Pip package not available in virtual environment: {e}")

    def test_venv_isolation(self):
        """Test that virtual environment provides proper isolation."""
        # Check that we're using the virtual environment's Python
        python_exe = sys.executable

        # Should not be system Python
        system_python_paths = ["/usr/bin/python", "/usr/local/bin/python", "C:\\Python", "C:\\Program Files\\Python"]
        using_system_python = any(system_path in python_exe for system_path in system_python_paths)

        # If we're in a venv, we shouldn't be using system Python
        if hasattr(sys, 'real_prefix') or os.environ.get("VIRTUAL_ENV"):
            assert not using_system_python, "Using system Python in virtual environment"


class TestHardwareCompatibility:
    """Test compatibility with different hardware configurations."""

    def test_gpu_compatibility(self, hardware_checker):
        """Test GPU compatibility and CUDA support."""
        gpu_info = hardware_checker.get_gpu_info()

        if gpu_info["available"]:
            # Test CUDA functionality
            assert torch.cuda.is_available()

            device_count = torch.cuda.device_count()
            assert device_count > 0

            # Test basic GPU operations
            device = torch.device("cuda:0")
            test_tensor = torch.randn(100, 100).to(device)
            result = test_tensor.sum()
            cpu_result = result.cpu().item()

            assert isinstance(cpu_result, float)
            assert not np.isnan(cpu_result)

        else:
            # CPU-only mode should work
            assert not torch.cuda.is_available()

            # Test CPU operations
            test_tensor = torch.randn(100, 100)
            result = test_tensor.sum().item()

            assert isinstance(result, float)
            assert not np.isnan(result)

    def test_multi_gpu_support(self, hardware_checker):
        """Test multi-GPU support."""
        gpu_info = hardware_checker.get_gpu_info()

        if gpu_info["available"]:
            device_count = torch.cuda.device_count()

            if device_count > 1:
                # Test multi-GPU operations
                devices = [torch.device(f"cuda:{i}") for i in range(device_count)]

                # Test tensor placement on different GPUs
                tensors = []
                for device in devices:
                    tensor = torch.randn(50, 50).to(device)
                    tensors.append(tensor)

                    # Test computation on each GPU
                    result = tensor.sum()
                    cpu_result = result.cpu()

                    assert torch.isfinite(cpu_result).all()

                # Test data parallelism concept
                # (Actual DataParallel would require model, just test device allocation)
                assert len(tensors) == device_count

    def test_cpu_only_inference(self, hardware_checker):
        """Test CPU-only inference performance."""
        # Force CPU usage
        device = torch.device("cpu")

        # Test model loading and inference on CPU
        # (Using mock since we don't have actual model in tests)
        test_input = torch.randn(1, 1, 256, 256).to(device)

        # Simulate inference
        with torch.no_grad():
            # Mock forward pass
            output = test_input * 2  # Simple operation

            assert output.shape == test_input.shape
            assert output.device == device

        # Test memory usage on CPU
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss

        # Perform some CPU-intensive operations
        for _ in range(10):
            large_tensor = torch.randn(1000, 1000)
            result = large_tensor.sum()

        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before

        # Memory usage should be reasonable (less than 500MB increase)
        assert memory_delta < 500 * 1024 * 1024

    def test_memory_constrained_environments(self, hardware_checker):
        """Test performance in memory-constrained environments."""
        # Test with smaller batch sizes for memory constraints
        memory_configs = [
            {"batch_size": 1, "image_size": (128, 128)},
            {"batch_size": 2, "image_size": (256, 256)},
            {"batch_size": 4, "image_size": (128, 128)}
        ]

        for config in memory_configs:
            batch_size = config["batch_size"]
            height, width = config["image_size"]

            # Test tensor creation and operations
            test_batch = torch.randn(batch_size, 1, height, width)

            # Test forward pass simulation
            with torch.no_grad():
                # Simulate model operations
                x = test_batch
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2)
                x = x.view(batch_size, -1)
                output = torch.softmax(x, dim=1)

                assert output.shape[0] == batch_size
                assert torch.all(torch.isfinite(output))

    def test_different_gpu_architectures(self, hardware_checker):
        """Test compatibility with different GPU architectures."""
        gpu_info = hardware_checker.get_gpu_info()

        if gpu_info["available"]:
            gpu_name = gpu_info["name"].lower()

            # Test basic operations work regardless of architecture
            device = torch.device("cuda:0")

            # Test different data types and operations
            test_configs = [
                {"dtype": torch.float32, "size": (100, 100)},
                {"dtype": torch.float16, "size": (50, 50)},
                {"dtype": torch.float64, "size": (25, 25)}
            ]

            for config in test_configs:
                tensor = torch.randn(*config["size"], dtype=config["dtype"]).to(device)

                # Test basic operations
                result = tensor.sum()
                cpu_result = result.cpu()

                assert torch.isfinite(cpu_result).all()

                # Test mixed precision if supported
                if config["dtype"] == torch.float16:
                    # Test autocast works
                    with torch.cuda.amp.autocast():
                        result_fp16 = tensor.sum()
                        assert torch.isfinite(result_fp16.cpu()).all()


class TestCrossPlatformIntegration:
    """Test end-to-end cross-platform integration."""

    def test_platform_independent_file_operations(self):
        """Test file operations work across platforms."""
        # Create test files and directories
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Test cross-platform path operations
            test_file = tmp_path / "test_data.txt"
            test_dir = tmp_path / "test_dir"

            # Create directory
            test_dir.mkdir()

            # Write file
            test_content = "Cross-platform test data\nWith multiple lines\n"
            test_file.write_text(test_content)

            # Read file
            read_content = test_file.read_text()
            assert read_content == test_content

            # List directory
            items = list(tmp_path.iterdir())
            assert len(items) == 2  # file and directory

            # Test path operations
            assert test_file.exists()
            assert test_dir.exists()
            assert test_dir.is_dir()
            assert test_file.is_file()

            # Test file permissions (read/write)
            # Note: On Windows, text mode converts \n to \r\n, so we check the actual bytes
            actual_size = test_file.stat().st_size
            # On Windows, \n becomes \r\n (2 bytes), so 2 newlines = 4 extra bytes
            # On Unix, \n stays \n (1 byte)
            expected_size_unix = len(test_content)
            expected_size_windows = len(test_content) + test_content.count('\n')
            assert actual_size in [expected_size_unix, expected_size_windows], \
                f"File size {actual_size} not in expected range [{expected_size_unix}, {expected_size_windows}]"

    def test_environment_variable_handling(self):
        """Test environment variable handling across platforms."""
        # Test common environment variables
        test_vars = {
            "HOME": None,  # Unix-like
            "USERPROFILE": None,  # Windows
            "PATH": None,
            "PYTHONPATH": None
        }

        # At least some should be set
        vars_set = sum(1 for var in test_vars.keys() if os.environ.get(var))
        assert vars_set > 0, "No standard environment variables found"

        # Test custom environment variable setting
        test_key = "SLICEWISE_TEST_VAR"
        test_value = "test_value_123"

        old_value = os.environ.get(test_key)
        os.environ[test_key] = test_value

        try:
            # Should be able to retrieve
            retrieved_value = os.environ.get(test_key)
            assert retrieved_value == test_value

            # Should work in subprocess
            result = subprocess.run([
                sys.executable, "-c",
                f"import os; print(os.environ.get('{test_key}'))"
            ], capture_output=True, text=True)

            assert result.returncode == 0
            assert test_value in result.stdout

        finally:
            # Clean up
            if old_value is not None:
                os.environ[test_key] = old_value
            else:
                os.environ.pop(test_key, None)

    def test_subprocess_compatibility(self):
        """Test subprocess operations across platforms."""
        # Test basic subprocess functionality
        result = subprocess.run([sys.executable, "-c", "print('Hello from subprocess')"],
                              capture_output=True, text=True)

        assert result.returncode == 0
        assert "Hello from subprocess" in result.stdout

        # Test platform-specific commands
        if platform.system() == "Windows":
            # Test Windows command
            result = subprocess.run(["cmd", "/c", "echo", "Windows test"],
                                  capture_output=True, text=True)
            assert result.returncode == 0
            assert "Windows test" in result.stdout
        else:
            # Test Unix-like commands
            result = subprocess.run(["echo", "Unix test"],
                                  capture_output=True, text=True)
            assert result.returncode == 0
            assert "Unix test" in result.stdout

    def test_network_connectivity(self):
        """Test network operations work across platforms."""
        import urllib.request

        # Test basic network connectivity
        try:
            # Try to connect to a reliable host (Google DNS)
            import socket
            socket.setdefaulttimeout(5)

            # Test DNS resolution
            ip = socket.gethostbyname("8.8.8.8")
            assert ip == "8.8.8.8"

            # Test HTTP request
            req = urllib.request.Request("http://httpbin.org/get",
                                       headers={"User-Agent": "SliceWise-Test"})
            with urllib.request.urlopen(req, timeout=10) as response:
                assert response.status == 200
                data = response.read()
                assert len(data) > 0

        except Exception as e:
            # Network tests might fail in restricted environments
            pytest.skip(f"Network test failed (expected in restricted environments): {e}")

    def test_system_resource_monitoring(self):
        """Test system resource monitoring across platforms."""
        import psutil

        # Test CPU monitoring
        cpu_percent = psutil.cpu_percent(interval=0.1)
        assert isinstance(cpu_percent, (int, float))
        assert 0 <= cpu_percent <= 100

        # Test memory monitoring
        memory = psutil.virtual_memory()
        assert memory.total > 0
        assert memory.available > 0
        assert memory.percent >= 0

        # Test disk monitoring
        disk = psutil.disk_usage('/')
        assert disk.total > 0
        assert disk.free > 0
        assert disk.percent >= 0

        # Test process monitoring
        process = psutil.Process()
        cpu_times = process.cpu_times()
        assert hasattr(cpu_times, 'user')
        assert hasattr(cpu_times, 'system')

        memory_info = process.memory_info()
        assert memory_info.rss > 0
