"""
API Testing Script for SliceWise Backend

This script tests all FastAPI endpoints to ensure they work correctly.
Make sure the backend is running before executing this script:
    python scripts/run_backend.py

Tests:
1. Health check endpoint
2. Model info endpoint
3. Single image classification
4. Batch classification
5. Classification with Grad-CAM
"""

import sys
from pathlib import Path
import requests
import numpy as np
from PIL import Image
import io
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class APITester:
    """Test suite for SliceWise API."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'tests': {},
            'overall_status': 'PENDING'
        }
    
    def print_header(self, text):
        """Print formatted header."""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)
    
    def print_test(self, name, status, message=""):
        """Print test result."""
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name}: {message}")
        self.test_results['tests'][name] = {
            'status': 'PASSED' if status else 'FAILED',
            'message': message
        }
        return status
    
    def create_test_image(self, size=(256, 256)):
        """Create a test image."""
        # Create random grayscale image
        img_array = (np.random.rand(*size) * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def test_1_health_check(self):
        """Test 1: Health check endpoint."""
        self.print_header("Test 1: Health Check")
        
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=5)
            
            if response.status_code != 200:
                return self.print_test(
                    "Health Check",
                    False,
                    f"Status code: {response.status_code}"
                )
            
            data = response.json()
            
            # Validate response structure
            assert 'status' in data, "Missing 'status' field"
            assert 'model_loaded' in data, "Missing 'model_loaded' field"
            assert 'device' in data, "Missing 'device' field"
            
            assert data['status'] == 'healthy', "Status is not healthy"
            assert data['model_loaded'] == True, "Model not loaded"
            
            return self.print_test(
                "Health Check",
                True,
                f"Status: {data['status']}, Device: {data['device']}"
            )
        
        except requests.exceptions.ConnectionError:
            return self.print_test(
                "Health Check",
                False,
                "Cannot connect to API. Is the backend running?"
            )
        except Exception as e:
            return self.print_test("Health Check", False, str(e))
    
    def test_2_model_info(self):
        """Test 2: Model info endpoint."""
        self.print_header("Test 2: Model Info")
        
        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=5)
            
            if response.status_code != 200:
                return self.print_test(
                    "Model Info",
                    False,
                    f"Status code: {response.status_code}"
                )
            
            data = response.json()
            
            # Validate response structure
            assert 'model_name' in data, "Missing 'model_name' field"
            assert 'num_classes' in data, "Missing 'num_classes' field"
            assert 'class_names' in data, "Missing 'class_names' field"
            assert 'input_size' in data, "Missing 'input_size' field"
            
            assert data['num_classes'] == 2, "Should have 2 classes"
            assert len(data['class_names']) == 2, "Should have 2 class names"
            
            return self.print_test(
                "Model Info",
                True,
                f"Model: {data['model_name']}, Classes: {data['class_names']}"
            )
        
        except Exception as e:
            return self.print_test("Model Info", False, str(e))
    
    def test_3_classify_single(self):
        """Test 3: Single image classification."""
        self.print_header("Test 3: Single Image Classification")
        
        try:
            # Create test image
            img_bytes = self.create_test_image()
            
            # Send request
            files = {'file': ('test.png', img_bytes, 'image/png')}
            params = {'return_probabilities': True}
            response = requests.post(
                f"{self.base_url}/classify_slice",
                files=files,
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                return self.print_test(
                    "Single Classification",
                    False,
                    f"Status code: {response.status_code}"
                )
            
            data = response.json()
            
            # Validate response structure
            assert 'predicted_class' in data, "Missing 'predicted_class'"
            assert 'predicted_label' in data, "Missing 'predicted_label'"
            assert 'confidence' in data, "Missing 'confidence'"
            assert 'probabilities' in data, "Missing 'probabilities'"
            
            # Validate values
            assert data['predicted_class'] in [0, 1], "Invalid class"
            assert data['predicted_label'] in ['No Tumor', 'Tumor'], "Invalid label"
            assert 0.0 <= data['confidence'] <= 1.0, "Invalid confidence"
            
            # Check probabilities sum to 1
            prob_sum = sum(data['probabilities'].values())
            assert abs(prob_sum - 1.0) < 0.01, "Probabilities don't sum to 1"
            
            return self.print_test(
                "Single Classification",
                True,
                f"Predicted: {data['predicted_label']} ({data['confidence']:.2%})"
            )
        
        except Exception as e:
            return self.print_test("Single Classification", False, str(e))
    
    def test_4_classify_batch(self):
        """Test 4: Batch classification."""
        self.print_header("Test 4: Batch Classification")
        
        try:
            # Create multiple test images
            num_images = 3
            files = []
            for i in range(num_images):
                img_bytes = self.create_test_image()
                files.append(('files', (f'test_{i}.png', img_bytes, 'image/png')))
            
            # Send request
            response = requests.post(
                f"{self.base_url}/classify_batch",
                files=files,
                timeout=15
            )
            
            if response.status_code != 200:
                return self.print_test(
                    "Batch Classification",
                    False,
                    f"Status code: {response.status_code}"
                )
            
            data = response.json()
            
            # Validate response structure
            assert 'num_images' in data, "Missing 'num_images'"
            assert 'predictions' in data, "Missing 'predictions'"
            
            assert data['num_images'] == num_images, "Wrong number of images"
            assert len(data['predictions']) == num_images, "Wrong number of predictions"
            
            # Validate each prediction
            for pred in data['predictions']:
                assert 'predicted_class' in pred
                assert 'predicted_label' in pred
                assert 'confidence' in pred
            
            return self.print_test(
                "Batch Classification",
                True,
                f"Processed {num_images} images successfully"
            )
        
        except Exception as e:
            return self.print_test("Batch Classification", False, str(e))
    
    def test_5_classify_with_gradcam(self):
        """Test 5: Classification with Grad-CAM."""
        self.print_header("Test 5: Classification with Grad-CAM")
        
        try:
            # Create test image
            img_bytes = self.create_test_image()
            
            # Send request
            files = {'file': ('test.png', img_bytes, 'image/png')}
            response = requests.post(
                f"{self.base_url}/classify_with_gradcam",
                files=files,
                timeout=15
            )
            
            if response.status_code != 200:
                return self.print_test(
                    "Grad-CAM Classification",
                    False,
                    f"Status code: {response.status_code}"
                )
            
            data = response.json()
            
            # Validate response structure
            assert 'predicted_class' in data, "Missing 'predicted_class'"
            assert 'predicted_label' in data, "Missing 'predicted_label'"
            assert 'confidence' in data, "Missing 'confidence'"
            assert 'gradcam_overlay' in data, "Missing 'gradcam_overlay'"
            
            # Validate Grad-CAM overlay is base64 encoded
            assert len(data['gradcam_overlay']) > 0, "Empty Grad-CAM overlay"
            
            # Try to decode base64
            import base64
            try:
                overlay_bytes = base64.b64decode(data['gradcam_overlay'])
                assert len(overlay_bytes) > 0, "Invalid base64 encoding"
            except:
                raise AssertionError("Failed to decode Grad-CAM overlay")
            
            return self.print_test(
                "Grad-CAM Classification",
                True,
                f"Predicted: {data['predicted_label']}, Grad-CAM generated"
            )
        
        except Exception as e:
            return self.print_test("Grad-CAM Classification", False, str(e))
    
    def test_6_root_endpoint(self):
        """Test 6: Root endpoint."""
        self.print_header("Test 6: Root Endpoint")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            
            if response.status_code != 200:
                return self.print_test(
                    "Root Endpoint",
                    False,
                    f"Status code: {response.status_code}"
                )
            
            data = response.json()
            
            assert 'message' in data, "Missing 'message' field"
            assert 'version' in data, "Missing 'version' field"
            assert 'endpoints' in data, "Missing 'endpoints' field"
            
            return self.print_test(
                "Root Endpoint",
                True,
                f"Version: {data['version']}, {len(data['endpoints'])} endpoints"
            )
        
        except Exception as e:
            return self.print_test("Root Endpoint", False, str(e))
    
    def test_7_error_handling(self):
        """Test 7: Error handling."""
        self.print_header("Test 7: Error Handling")
        
        try:
            # Test with invalid file
            files = {'file': ('test.txt', b'not an image', 'text/plain')}
            response = requests.post(
                f"{self.base_url}/classify_slice",
                files=files,
                timeout=10
            )
            
            # Should return error status
            if response.status_code == 200:
                return self.print_test(
                    "Error Handling",
                    False,
                    "API accepted invalid file type"
                )
            
            # Should have error message
            data = response.json()
            assert 'detail' in data or 'error' in data, "No error message"
            
            return self.print_test(
                "Error Handling",
                True,
                "API correctly rejects invalid inputs"
            )
        
        except Exception as e:
            return self.print_test("Error Handling", False, str(e))
    
    def run_all_tests(self):
        """Run all API tests."""
        print("\n" + "="*70)
        print("  SLICEWISE API TEST SUITE")
        print("="*70)
        print(f"  Base URL: {self.base_url}")
        print(f"  Timestamp: {self.test_results['timestamp']}")
        print("="*70)
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_1_health_check,
            self.test_2_model_info,
            self.test_3_classify_single,
            self.test_4_classify_batch,
            self.test_5_classify_with_gradcam,
            self.test_6_root_endpoint,
            self.test_7_error_handling,
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                results.append(False)
        
        elapsed_time = time.time() - start_time
        
        # Summary
        self.print_header("TEST SUMMARY")
        
        passed = sum(results)
        total = len(results)
        success_rate = (passed / total) * 100
        
        print(f"\n📊 Results:")
        print(f"   Total Tests: {total}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {total - passed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Time: {elapsed_time:.2f}s")
        
        # Save results
        self.test_results['overall_status'] = 'PASSED' if passed == total else 'FAILED'
        self.test_results['summary'] = {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': success_rate,
            'elapsed_time': elapsed_time
        }
        
        results_file = project_root / "api_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Final status
        print("\n" + "="*70)
        if passed == total:
            print("✅ ALL API TESTS PASSED!")
            print("🎉 API is fully functional and production-ready!")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Review errors above.")
        print("="*70 + "\n")
        
        return passed == total


if __name__ == "__main__":
    print("\n🚀 SliceWise API Testing")
    print("="*70)
    print("\n⚠️  Make sure the backend is running:")
    print("   python scripts/run_backend.py")
    print("\nWaiting 3 seconds before starting tests...")
    time.sleep(3)
    
    tester = APITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
