#!/usr/bin/env python3
"""
Verification script for SliceWise project setup.
Tests all dependencies, imports, and basic functionality.
"""

import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(test_name, passed, message=""):
    """Print test status."""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} - {test_name}")
    if message:
        print(f"       {message}")


def test_python_version():
    """Test Python version."""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    passed = version.major == 3 and version.minor >= 10
    print_status(
        "Python version >= 3.10",
        passed,
        f"Found: {version.major}.{version.minor}" if not passed else ""
    )
    return passed


def test_core_imports():
    """Test core library imports."""
    print_header("Core Dependencies")
    
    tests = []
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        tests.append(("PyTorch", True, ""))
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA: Available (Device: {torch.cuda.get_device_name(0)})")
        else:
            print("CUDA: Not available (CPU only)")
        tests.append(("CUDA detection", True, ""))
    except ImportError as e:
        tests.append(("PyTorch", False, str(e)))
    
    # NumPy
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        tests.append(("NumPy", True, ""))
    except ImportError as e:
        tests.append(("NumPy", False, str(e)))
    
    # OpenCV
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
        tests.append(("OpenCV", True, ""))
    except ImportError as e:
        tests.append(("OpenCV", False, str(e)))
    
    # scikit-learn
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
        tests.append(("scikit-learn", True, ""))
    except ImportError as e:
        tests.append(("scikit-learn", False, str(e)))
    
    # tqdm
    try:
        import tqdm
        print(f"tqdm: {tqdm.__version__}")
        tests.append(("tqdm", True, ""))
    except ImportError as e:
        tests.append(("tqdm", False, str(e)))
    
    print()
    for name, passed, msg in tests:
        print_status(name, passed, msg)
    
    return all(passed for _, passed, _ in tests)


def test_project_imports():
    """Test project module imports."""
    print_header("Project Modules")
    
    tests = []
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test src package
    try:
        import src
        print(f"src package: {src.__version__}")
        tests.append(("src package", True, ""))
    except ImportError as e:
        tests.append(("src package", False, str(e)))
    
    # Test data module
    try:
        from src.data import transforms
        tests.append(("src.data.transforms", True, ""))
    except ImportError as e:
        tests.append(("src.data.transforms", False, str(e)))
    
    try:
        from src.data import kaggle_mri_dataset
        tests.append(("src.data.kaggle_mri_dataset", True, ""))
    except ImportError as e:
        tests.append(("src.data.kaggle_mri_dataset", False, str(e)))
    
    print()
    for name, passed, msg in tests:
        print_status(name, passed, msg)
    
    return all(passed for _, passed, _ in tests)


def test_transforms():
    """Test data transforms."""
    print_header("Data Transforms")
    
    tests = []
    
    try:
        import torch
        from src.data.transforms import (
            get_train_transforms,
            get_val_transforms,
            RandomRotation90,
            RandomIntensityShift,
        )
        
        # Create dummy image
        img = torch.rand(1, 256, 256)
        
        # Test train transforms
        train_transform = get_train_transforms()
        augmented = train_transform(img)
        
        passed = augmented.shape == img.shape
        tests.append((
            "Train transforms",
            passed,
            f"Shape: {augmented.shape}" if passed else "Shape mismatch"
        ))
        
        # Test val transforms
        val_transform = get_val_transforms()
        if val_transform is None:
            tests.append(("Val transforms", True, "No augmentation (as expected)"))
        else:
            val_out = val_transform(img)
            tests.append(("Val transforms", val_out.shape == img.shape, ""))
        
        # Test custom transforms
        rot90 = RandomRotation90(p=1.0)
        rotated = rot90(img)
        tests.append(("RandomRotation90", rotated.shape == img.shape, ""))
        
        intensity_shift = RandomIntensityShift(p=1.0)
        shifted = intensity_shift(img)
        tests.append(("RandomIntensityShift", shifted.shape == img.shape, ""))
        
    except Exception as e:
        tests.append(("Transforms", False, str(e)))
    
    print()
    for name, passed, msg in tests:
        print_status(name, passed, msg)
    
    return all(passed for _, passed, _ in tests)


def test_dataset_class():
    """Test PyTorch dataset class (without actual data)."""
    print_header("Dataset Classes")
    
    tests = []
    
    try:
        from src.data.kaggle_mri_dataset import KaggleBrainMRIDataset
        tests.append(("KaggleBrainMRIDataset import", True, ""))
        
        # Test that class can be instantiated (will fail if no data, but that's ok)
        try:
            dataset = KaggleBrainMRIDataset("data/processed/kaggle/train")
            tests.append((
                "Dataset instantiation",
                True,
                f"Found {len(dataset)} samples"
            ))
        except ValueError as e:
            if "No .npz files found" in str(e):
                tests.append((
                    "Dataset instantiation",
                    True,
                    "No data yet (expected)"
                ))
            else:
                tests.append(("Dataset instantiation", False, str(e)))
        
    except Exception as e:
        tests.append(("Dataset classes", False, str(e)))
    
    print()
    for name, passed, msg in tests:
        print_status(name, passed, msg)
    
    return all(passed for _, passed, _ in tests)


def test_file_structure():
    """Test project file structure."""
    print_header("Project Structure")
    
    project_root = Path(__file__).parent.parent
    
    required_files = [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "LICENSE",
        ".gitignore",
        ".pre-commit-config.yaml",
        "README.md",
    ]
    
    required_dirs = [
        "src",
        "src/data",
        "tests",
        "scripts",
        "configs",
        "documentation",
        "jupyter_notebooks",
    ]
    
    tests = []
    
    for file in required_files:
        path = project_root / file
        tests.append((f"File: {file}", path.exists(), ""))
    
    for dir_path in required_dirs:
        path = project_root / dir_path
        tests.append((f"Dir: {dir_path}", path.exists(), ""))
    
    print()
    for name, passed, msg in tests:
        print_status(name, passed, msg)
    
    return all(passed for _, passed, _ in tests)


def test_scripts():
    """Test that scripts exist and are executable."""
    print_header("Scripts")
    
    project_root = Path(__file__).parent.parent
    
    scripts = [
        "scripts/smoke_test.py",
        "scripts/download_kaggle_data.py",
        "src/data/preprocess_kaggle.py",
        "src/data/split_kaggle.py",
    ]
    
    tests = []
    
    for script in scripts:
        path = project_root / script
        exists = path.exists()
        tests.append((script, exists, ""))
    
    print()
    for name, passed, msg in tests:
        print_status(name, passed, msg)
    
    return all(passed for _, passed, _ in tests)


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("  SliceWise Project Verification")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Python Version", test_python_version()))
    results.append(("Core Dependencies", test_core_imports()))
    results.append(("Project Imports", test_project_imports()))
    results.append(("Data Transforms", test_transforms()))
    results.append(("Dataset Classes", test_dataset_class()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Scripts", test_scripts()))
    
    # Summary
    print_header("Summary")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        print_status(name, passed)
    
    print(f"\nTotal: {passed_count}/{total_count} test groups passed")
    
    if passed_count == total_count:
        print("\n✓ All verification tests passed!")
        print("You're ready to proceed with data download and preprocessing.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -e \".[dev]\"")
        print("2. Make sure you're in the project root directory")
        print("3. Check that all files were created correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
