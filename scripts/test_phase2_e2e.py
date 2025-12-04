"""
End-to-End Test for Phase 2: Classification MVP

This script performs a comprehensive test of the entire Phase 2 pipeline:
1. Data loading and preprocessing
2. Model creation and initialization
3. Training (mini version with few epochs)
4. Evaluation and metrics
5. Grad-CAM generation
6. Inference on new images
7. API endpoint testing (if backend is running)

Run this to validate the complete Phase 2 implementation.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import time
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import create_classifier
from src.data.kaggle_mri_dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.inference.predict import ClassifierPredictor
from src.eval.grad_cam import GradCAM


class Phase2E2ETest:
    """End-to-end test suite for Phase 2."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'tests': {},
            'overall_status': 'PENDING'
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
    
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
    
    def test_1_data_pipeline(self):
        """Test 1: Data loading and preprocessing."""
        self.print_header("Test 1: Data Pipeline")
        
        try:
            # Check if data exists
            data_dir = self.project_root / "data" / "processed" / "kaggle"
            if not data_dir.exists():
                return self.print_test(
                    "Data Pipeline",
                    False,
                    "Data not found. Run preprocessing first."
                )
            
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                batch_size=4,
                num_workers=0,
                train_transform=get_train_transforms(),
                val_transform=get_val_transforms()
            )
            
            # Test loading a batch
            images, labels = next(iter(train_loader))
            
            assert images.shape[1] == 1, "Should be single channel"
            assert images.shape[2] == 256, "Height should be 256"
            assert images.shape[3] == 256, "Width should be 256"
            assert len(labels) > 0, "Should have labels"
            
            return self.print_test(
                "Data Pipeline",
                True,
                f"Loaded {len(train_loader.dataset)} train, "
                f"{len(val_loader.dataset)} val, "
                f"{len(test_loader.dataset)} test samples"
            )
        
        except Exception as e:
            return self.print_test("Data Pipeline", False, str(e))
    
    def test_2_model_creation(self):
        """Test 2: Model creation and architecture."""
        self.print_header("Test 2: Model Creation")
        
        try:
            # Test EfficientNet
            model_eff = create_classifier('efficientnet', pretrained=False)
            model_eff = model_eff.to(self.device)
            
            # Test forward pass
            dummy_input = torch.randn(2, 1, 256, 256).to(self.device)
            output = model_eff(dummy_input)
            
            assert output.shape == (2, 2), "Output should be (batch_size, num_classes)"
            
            # Test ConvNeXt
            model_conv = create_classifier('convnext', pretrained=False)
            
            # Store model for later tests
            self.test_model = model_eff
            
            return self.print_test(
                "Model Creation",
                True,
                f"EfficientNet: {model_eff.get_num_trainable_params():,} params"
            )
        
        except Exception as e:
            return self.print_test("Model Creation", False, str(e))
    
    def test_3_mini_training(self):
        """Test 3: Mini training run (2 epochs)."""
        self.print_header("Test 3: Mini Training Run")
        
        try:
            # Create small dataloaders
            train_loader, val_loader, _ = create_dataloaders(
                batch_size=8,
                num_workers=0,
                train_transform=None,
                val_transform=None
            )
            
            # Use existing model
            model = self.test_model
            model.train()
            
            # Setup training
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train for 2 epochs
            num_epochs = 2
            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for i, (images, labels) in enumerate(train_loader):
                    if i >= 5:  # Only 5 batches per epoch for speed
                        break
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                epoch_loss = running_loss / min(5, len(train_loader))
                epoch_acc = 100 * correct / total
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
            
            # Save checkpoint for later tests
            checkpoint_dir = self.project_root / "checkpoints" / "cls"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': epoch_acc / 100,
                'config': {}
            }
            
            checkpoint_path = checkpoint_dir / "e2e_test_model.pth"
            torch.save(checkpoint, checkpoint_path)
            self.checkpoint_path = str(checkpoint_path)
            
            return self.print_test(
                "Mini Training",
                True,
                f"Trained for {num_epochs} epochs, final acc={epoch_acc:.2f}%"
            )
        
        except Exception as e:
            return self.print_test("Mini Training", False, str(e))
    
    def test_4_evaluation(self):
        """Test 4: Model evaluation."""
        self.print_header("Test 4: Model Evaluation")
        
        try:
            model = self.test_model
            model.eval()
            
            # Get test data
            _, _, test_loader = create_dataloaders(
                batch_size=8,
                num_workers=0
            )
            
            # Evaluate
            correct = 0
            total = 0
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    if i >= 5:  # Only 5 batches for speed
                        break
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            accuracy = 100 * correct / total
            
            # Calculate basic metrics
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(all_labels, all_probs)
                metric_str = f"Acc={accuracy:.2f}%, AUC={auc:.3f}"
            except:
                metric_str = f"Acc={accuracy:.2f}%"
            
            return self.print_test("Model Evaluation", True, metric_str)
        
        except Exception as e:
            return self.print_test("Model Evaluation", False, str(e))
    
    def test_5_gradcam(self):
        """Test 5: Grad-CAM generation."""
        self.print_header("Test 5: Grad-CAM Generation")
        
        try:
            model = self.test_model
            model.eval()
            model = model.to(self.device)  # Ensure model is on correct device
            
            # Get a sample image
            _, _, test_loader = create_dataloaders(batch_size=1, num_workers=0)
            images, labels = next(iter(test_loader))
            images = images.to(self.device)
            
            # Generate Grad-CAM
            target_layer = model.get_cam_target_layer()
            grad_cam = GradCAM(model, target_layer)
            
            cam = grad_cam.generate_cam(images)
            
            assert cam.shape[0] > 0, "CAM should have height"
            assert cam.shape[1] > 0, "CAM should have width"
            assert cam.min() >= 0.0, "CAM values should be >= 0"
            assert cam.max() <= 1.0, "CAM values should be <= 1"
            
            # Generate overlay
            original_image = images[0, 0].cpu().numpy()
            overlay = grad_cam.generate_overlay(original_image, cam)
            
            assert overlay.shape[2] == 3, "Overlay should be RGB"
            
            return self.print_test(
                "Grad-CAM Generation",
                True,
                f"Generated {cam.shape} heatmap"
            )
        
        except Exception as e:
            return self.print_test("Grad-CAM Generation", False, str(e))
    
    def test_6_inference(self):
        """Test 6: Inference on new images."""
        self.print_header("Test 6: Inference Pipeline")
        
        try:
            # Create predictor
            predictor = ClassifierPredictor(
                self.checkpoint_path,
                model_name='efficientnet',
                device='cpu'  # Use CPU for consistency
            )
            
            # Test single image prediction
            dummy_image = np.random.rand(256, 256).astype(np.float32)
            result = predictor.predict(dummy_image)
            
            assert 'predicted_class' in result
            assert 'predicted_label' in result
            assert 'confidence' in result
            assert 'probabilities' in result
            
            # Test batch prediction
            images = [np.random.rand(256, 256).astype(np.float32) for _ in range(3)]
            results = predictor.predict_batch(images)
            
            assert len(results) == 3
            
            return self.print_test(
                "Inference Pipeline",
                True,
                f"Predicted: {result['predicted_label']} ({result['confidence']:.2%})"
            )
        
        except Exception as e:
            return self.print_test("Inference Pipeline", False, str(e))
    
    def test_7_api_endpoints(self):
        """Test 7: API endpoints (if backend is running)."""
        self.print_header("Test 7: API Endpoints")
        
        try:
            import requests
            from PIL import Image
            import io
            
            # Check if API is running
            try:
                response = requests.get("http://localhost:8000/healthz", timeout=2)
                if response.status_code != 200:
                    return self.print_test(
                        "API Endpoints",
                        False,
                        "Backend not running. Start with: python scripts/run_backend.py"
                    )
            except:
                return self.print_test(
                    "API Endpoints",
                    False,
                    "Backend not running. Start with: python scripts/run_backend.py"
                )
            
            # Test /model/info endpoint
            response = requests.get("http://localhost:8000/model/info")
            assert response.status_code == 200
            model_info = response.json()
            
            # Test /classify_slice endpoint
            dummy_image = (np.random.rand(256, 256) * 255).astype(np.uint8)
            pil_image = Image.fromarray(dummy_image)
            
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            files = {'file': ('test.png', img_byte_arr, 'image/png')}
            response = requests.post("http://localhost:8000/classify_slice", files=files)
            
            assert response.status_code == 200
            result = response.json()
            assert 'predicted_class' in result
            
            return self.print_test(
                "API Endpoints",
                True,
                f"API working, model: {model_info['model_name']}"
            )
        
        except Exception as e:
            return self.print_test(
                "API Endpoints",
                False,
                f"Skipped (backend not running): {str(e)}"
            )
    
    def test_8_integration(self):
        """Test 8: Full integration test."""
        self.print_header("Test 8: Full Integration")
        
        try:
            # Load real data
            _, _, test_loader = create_dataloaders(batch_size=1, num_workers=0)
            images, labels = next(iter(test_loader))
            
            # Convert to numpy
            image_np = images[0, 0].cpu().numpy()
            true_label = labels[0].item()
            
            # Run through predictor (uses CPU)
            predictor = ClassifierPredictor(
                self.checkpoint_path,
                model_name='efficientnet',
                device='cpu'
            )
            
            result = predictor.predict(image_np)
            pred_label = result['predicted_class']
            confidence = result['confidence']
            
            # Generate Grad-CAM (move model to self.device for Grad-CAM)
            model = predictor.model.to(self.device)
            model.eval()
            images = images.to(self.device)
            target_layer = model.get_cam_target_layer()
            grad_cam = GradCAM(model, target_layer)
            cam = grad_cam.generate_cam(images)
            
            is_correct = pred_label == true_label
            status_str = "✓ Correct" if is_correct else "✗ Incorrect"
            
            return self.print_test(
                "Full Integration",
                True,
                f"{status_str} prediction with {confidence:.2%} confidence, Grad-CAM generated"
            )
        
        except Exception as e:
            return self.print_test("Full Integration", False, str(e))
    
    def run_all_tests(self):
        """Run all end-to-end tests."""
        print("\n" + "="*70)
        print("  PHASE 2 END-TO-END TEST SUITE")
        print("="*70)
        print(f"  Timestamp: {self.test_results['timestamp']}")
        print(f"  Device: {self.device}")
        print("="*70)
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_1_data_pipeline,
            self.test_2_model_creation,
            self.test_3_mini_training,
            self.test_4_evaluation,
            self.test_5_gradcam,
            self.test_6_inference,
            self.test_7_api_endpoints,
            self.test_8_integration,
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
        
        results_file = self.project_root / "phase2_e2e_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Final status
        print("\n" + "="*70)
        if passed == total:
            print("✅ ALL END-TO-END TESTS PASSED!")
            print("🎉 Phase 2 is fully functional and production-ready!")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Review errors above.")
        print("="*70 + "\n")
        
        return passed == total


if __name__ == "__main__":
    tester = Phase2E2ETest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
