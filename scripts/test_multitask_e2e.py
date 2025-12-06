"""
End-to-End Test for Multi-Task Model Integration

This script tests the complete multi-task inference pipeline:
1. Checkpoint validation
2. MultiTaskPredictor functionality
3. Conditional segmentation logic
4. API endpoint testing
5. Performance benchmarking

Usage:
    python scripts/test_multitask_e2e.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import json
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.multi_task_predictor import create_multi_task_predictor

# Configuration
CHECKPOINT_PATH = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
TEST_IMAGES_DIR = project_root / "data" / "dataset_examples"
API_URL = "http://localhost:8000"

# Test results
test_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "tests": {},
    "summary": {}
}


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test(name, passed, details=""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status} - {name}")
    if details:
        print(f"       {details}")
    return passed


def test_checkpoint_exists():
    """Test 1: Verify checkpoint exists."""
    print_section("Test 1: Checkpoint Validation")
    
    exists = CHECKPOINT_PATH.exists()
    if exists:
        size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
        details = f"Size: {size_mb:.2f} MB"
    else:
        details = f"Not found at {CHECKPOINT_PATH}"
    
    passed = print_test("Checkpoint exists", exists, details)
    test_results["tests"]["checkpoint_exists"] = {
        "passed": passed,
        "path": str(CHECKPOINT_PATH),
        "size_mb": size_mb if exists else None
    }
    
    return passed


def test_predictor_creation():
    """Test 2: Create MultiTaskPredictor."""
    print_section("Test 2: MultiTaskPredictor Creation")
    
    try:
        predictor = create_multi_task_predictor(
            checkpoint_path=str(CHECKPOINT_PATH),
            classification_threshold=0.3,
            segmentation_threshold=0.5
        )
        
        # Get model info
        info = predictor.get_model_info()
        
        details = f"Parameters: {info['parameters']['total']:,}"
        passed = print_test("Predictor created successfully", True, details)
        
        test_results["tests"]["predictor_creation"] = {
            "passed": True,
            "model_info": info
        }
        
        return predictor
    
    except Exception as e:
        passed = print_test("Predictor created successfully", False, str(e))
        test_results["tests"]["predictor_creation"] = {
            "passed": False,
            "error": str(e)
        }
        return None


def test_single_inference(predictor):
    """Test 3: Single image inference."""
    print_section("Test 3: Single Image Inference")
    
    if predictor is None:
        print_test("Single inference", False, "Predictor not available")
        return False
    
    try:
        # Create dummy image
        dummy_image = np.random.rand(256, 256).astype(np.float32)
        
        # Test predict_single
        start_time = time.time()
        result = predictor.predict_single(dummy_image, do_seg=True, do_cls=True)
        inference_time = (time.time() - start_time) * 1000
        
        # Validate result structure
        has_cls = 'classification' in result
        has_seg = 'segmentation' in result
        
        if has_cls:
            tumor_prob = result['classification']['tumor_probability']
            details = f"Tumor prob: {tumor_prob:.3f}, Time: {inference_time:.2f}ms"
        else:
            details = "Missing classification results"
        
        passed = print_test("Single inference", has_cls and has_seg, details)
        
        test_results["tests"]["single_inference"] = {
            "passed": passed,
            "inference_time_ms": inference_time,
            "has_classification": has_cls,
            "has_segmentation": has_seg,
            "tumor_probability": tumor_prob if has_cls else None
        }
        
        return passed
    
    except Exception as e:
        passed = print_test("Single inference", False, str(e))
        test_results["tests"]["single_inference"] = {
            "passed": False,
            "error": str(e)
        }
        return False


def test_conditional_inference(predictor):
    """Test 4: Conditional segmentation logic."""
    print_section("Test 4: Conditional Segmentation Logic")
    
    if predictor is None:
        print_test("Conditional inference", False, "Predictor not available")
        return False
    
    try:
        # Create dummy image
        dummy_image = np.random.rand(256, 256).astype(np.float32)
        
        # Test conditional prediction
        result = predictor.predict_conditional(dummy_image)
        
        has_cls = 'classification' in result
        has_computed = 'segmentation_computed' in result
        has_recommendation = 'recommendation' in result
        
        if has_cls and has_computed:
            tumor_prob = result['classification']['tumor_probability']
            seg_computed = result['segmentation_computed']
            
            # Verify conditional logic
            expected_computed = tumor_prob >= 0.3
            logic_correct = seg_computed == expected_computed
            
            details = f"Tumor prob: {tumor_prob:.3f}, Seg computed: {seg_computed}, Logic: {'✓' if logic_correct else '✗'}"
        else:
            details = "Missing required fields"
            logic_correct = False
        
        passed = print_test("Conditional inference", has_cls and has_computed and logic_correct, details)
        
        test_results["tests"]["conditional_inference"] = {
            "passed": passed,
            "tumor_probability": tumor_prob if has_cls else None,
            "segmentation_computed": seg_computed if has_computed else None,
            "logic_correct": logic_correct
        }
        
        return passed
    
    except Exception as e:
        passed = print_test("Conditional inference", False, str(e))
        test_results["tests"]["conditional_inference"] = {
            "passed": False,
            "error": str(e)
        }
        return False


def test_batch_inference(predictor):
    """Test 5: Batch inference."""
    print_section("Test 5: Batch Inference")
    
    if predictor is None:
        print_test("Batch inference", False, "Predictor not available")
        return False
    
    try:
        # Create batch of dummy images
        batch_size = 4
        dummy_images = [np.random.rand(256, 256).astype(np.float32) for _ in range(batch_size)]
        
        # Test batch prediction
        start_time = time.time()
        results = predictor.predict_batch(dummy_images, do_seg=True, do_cls=True)
        batch_time = (time.time() - start_time) * 1000
        
        # Validate results
        correct_count = len(results) == batch_size
        all_have_cls = all('classification' in r for r in results)
        all_have_seg = all('segmentation' in r for r in results)
        
        avg_time = batch_time / batch_size
        details = f"Batch size: {batch_size}, Avg time: {avg_time:.2f}ms/image"
        
        passed = print_test("Batch inference", correct_count and all_have_cls and all_have_seg, details)
        
        test_results["tests"]["batch_inference"] = {
            "passed": passed,
            "batch_size": batch_size,
            "total_time_ms": batch_time,
            "avg_time_per_image_ms": avg_time,
            "all_have_classification": all_have_cls,
            "all_have_segmentation": all_have_seg
        }
        
        return passed
    
    except Exception as e:
        passed = print_test("Batch inference", False, str(e))
        test_results["tests"]["batch_inference"] = {
            "passed": False,
            "error": str(e)
        }
        return False


def test_gradcam(predictor):
    """Test 6: Grad-CAM visualization."""
    print_section("Test 6: Grad-CAM Visualization")
    
    if predictor is None:
        print_test("Grad-CAM", False, "Predictor not available")
        return False
    
    try:
        # Create dummy image
        dummy_image = np.random.rand(256, 256).astype(np.float32)
        
        # Test Grad-CAM
        start_time = time.time()
        result = predictor.predict_with_gradcam(dummy_image)
        gradcam_time = (time.time() - start_time) * 1000
        
        # Validate result
        has_gradcam = 'gradcam' in result
        has_heatmap = has_gradcam and 'heatmap' in result['gradcam']
        
        if has_heatmap:
            heatmap_shape = result['gradcam']['heatmap'].shape
            details = f"Heatmap shape: {heatmap_shape}, Time: {gradcam_time:.2f}ms"
        else:
            details = "Missing heatmap"
        
        passed = print_test("Grad-CAM generation", has_gradcam and has_heatmap, details)
        
        test_results["tests"]["gradcam"] = {
            "passed": passed,
            "time_ms": gradcam_time,
            "has_heatmap": has_heatmap,
            "heatmap_shape": list(heatmap_shape) if has_heatmap else None
        }
        
        return passed
    
    except Exception as e:
        passed = print_test("Grad-CAM generation", False, str(e))
        test_results["tests"]["gradcam"] = {
            "passed": False,
            "error": str(e)
        }
        return False


def test_api_health():
    """Test 7: API health check."""
    print_section("Test 7: API Health Check")
    
    try:
        response = requests.get(f"{API_URL}/healthz", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            multitask_loaded = data.get('multitask_loaded', False)
            device = data.get('device', 'unknown')
            
            details = f"Multi-task loaded: {multitask_loaded}, Device: {device}"
            passed = print_test("API health check", multitask_loaded, details)
            
            test_results["tests"]["api_health"] = {
                "passed": passed,
                "response": data
            }
        else:
            passed = print_test("API health check", False, f"Status: {response.status_code}")
            test_results["tests"]["api_health"] = {
                "passed": False,
                "status_code": response.status_code
            }
        
        return passed
    
    except requests.exceptions.RequestException as e:
        passed = print_test("API health check", False, "API not running")
        test_results["tests"]["api_health"] = {
            "passed": False,
            "error": "API not running",
            "note": "Start API with: python scripts/run_multitask_demo.py"
        }
        return False


def test_api_predict(predictor):
    """Test 8: API prediction endpoint."""
    print_section("Test 8: API Prediction Endpoint")
    
    try:
        # Create dummy image
        dummy_image = np.random.rand(256, 256).astype(np.uint8)
        pil_image = Image.fromarray(dummy_image)
        
        # Save to bytes
        import io
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Call API
        files = {'file': ('test.png', img_bytes, 'image/png')}
        params = {'include_gradcam': True}
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict_multitask",
            files=files,
            params=params,
            timeout=30
        )
        api_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            has_cls = 'classification' in result
            has_seg_computed = 'segmentation_computed' in result
            has_processing_time = 'processing_time_ms' in result
            
            if has_cls:
                tumor_prob = result['classification']['tumor_probability']
                details = f"Tumor prob: {tumor_prob:.3f}, API time: {api_time:.2f}ms"
            else:
                details = "Missing classification"
            
            passed = print_test("API prediction", has_cls and has_seg_computed, details)
            
            test_results["tests"]["api_predict"] = {
                "passed": passed,
                "api_time_ms": api_time,
                "processing_time_ms": result.get('processing_time_ms'),
                "tumor_probability": tumor_prob if has_cls else None,
                "segmentation_computed": result.get('segmentation_computed')
            }
        else:
            passed = print_test("API prediction", False, f"Status: {response.status_code}")
            test_results["tests"]["api_predict"] = {
                "passed": False,
                "status_code": response.status_code,
                "error": response.text
            }
        
        return passed
    
    except requests.exceptions.RequestException as e:
        passed = print_test("API prediction", False, "API not running")
        test_results["tests"]["api_predict"] = {
            "passed": False,
            "error": "API not running"
        }
        return False
    except Exception as e:
        passed = print_test("API prediction", False, str(e))
        test_results["tests"]["api_predict"] = {
            "passed": False,
            "error": str(e)
        }
        return False


def test_performance_benchmark(predictor):
    """Test 9: Performance benchmarking."""
    print_section("Test 9: Performance Benchmarking")
    
    if predictor is None:
        print_test("Performance benchmark", False, "Predictor not available")
        return False
    
    try:
        # Create test images
        num_iterations = 10
        dummy_image = np.random.rand(256, 256).astype(np.float32)
        
        # Benchmark conditional inference
        times = []
        for _ in range(num_iterations):
            start = time.time()
            result = predictor.predict_conditional(dummy_image)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        details = f"Avg: {avg_time:.2f}ms, Std: {std_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms"
        passed = print_test("Performance benchmark", True, details)
        
        test_results["tests"]["performance_benchmark"] = {
            "passed": True,
            "num_iterations": num_iterations,
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "times_ms": times
        }
        
        return passed
    
    except Exception as e:
        passed = print_test("Performance benchmark", False, str(e))
        test_results["tests"]["performance_benchmark"] = {
            "passed": False,
            "error": str(e)
        }
        return False


def print_summary():
    """Print test summary."""
    print_section("Test Summary")
    
    total_tests = len(test_results["tests"])
    passed_tests = sum(1 for t in test_results["tests"].values() if t.get("passed", False))
    failed_tests = total_tests - passed_tests
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✓ Passed: {passed_tests}")
    print(f"✗ Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "success_rate": (passed_tests/total_tests)*100
    }
    
    # Save results
    output_file = project_root / "multitask_e2e_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return passed_tests == total_tests


def main():
    """Run all tests."""
    print("=" * 80)
    print("  Multi-Task Model End-to-End Testing")
    print("=" * 80)
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"API URL: {API_URL}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Run tests
    predictor = None
    
    # Test 1: Checkpoint
    if not test_checkpoint_exists():
        print("\n⚠️  Checkpoint not found. Please train the model first:")
        print("   python scripts/train_multitask_joint.py")
        return False
    
    # Test 2: Predictor creation
    predictor = test_predictor_creation()
    
    # Test 3-6: Predictor functionality
    if predictor:
        test_single_inference(predictor)
        test_conditional_inference(predictor)
        test_batch_inference(predictor)
        test_gradcam(predictor)
        test_performance_benchmark(predictor)
    
    # Test 7-8: API tests
    test_api_health()
    test_api_predict(predictor)
    
    # Print summary
    all_passed = print_summary()
    
    if all_passed:
        print("\n" + "=" * 80)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\n🎉 Multi-task model integration is working correctly!")
        print("\nNext steps:")
        print("  1. Launch demo: python scripts/run_multitask_demo.py")
        print("  2. Test with real images")
        print("  3. Deploy to production")
    else:
        print("\n" + "=" * 80)
        print("  ⚠️  SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease review the failures above and:")
        print("  1. Check if checkpoint exists")
        print("  2. Verify API is running (for API tests)")
        print("  3. Review error messages")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
