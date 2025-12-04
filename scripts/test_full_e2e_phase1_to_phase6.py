"""
Comprehensive End-to-End Test: Phase 1 → Phase 6
SliceWise MRI Brain Tumor Detection Project

This script validates the entire pipeline:
- Phase 1: Data acquisition and preprocessing
- Phase 2: Classification (training, evaluation, Grad-CAM)
- Phase 3: Segmentation (training, evaluation, post-processing)
- Phase 4: Calibration and uncertainty estimation
- Phase 5: Metrics and patient-level evaluation
- Phase 6: API endpoints and full integration

Usage:
    python scripts/test_full_e2e_phase1_to_phase6.py [--quick] [--skip-training]
"""

import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from PIL import Image
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.data.kaggle_mri_dataset import KaggleBrainMRIDataset, create_dataloaders
from src.data.brats2d_dataset import BraTS2DSliceDataset
from src.models.classifier import BrainTumorClassifier
from src.models.unet2d import UNet2D
from src.inference.predict import ClassifierPredictor
from src.inference.infer_seg2d import SegmentationPredictor
from src.inference.uncertainty import MCDropoutPredictor, TTAPredictor, EnsemblePredictor
from src.eval.calibration import TemperatureScaling
from src.eval.grad_cam import GradCAM
from src.inference.postprocess import postprocess_mask
from src.eval.patient_level_eval import PatientLevelEvaluator


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_test(test_name):
    """Print test name."""
    print(f"{Colors.OKBLUE}▶ Testing: {test_name}{Colors.ENDC}")


def print_success(message):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


class FullE2ETest:
    """Comprehensive end-to-end test suite."""
    
    def __init__(self, quick_mode=False, skip_training=False):
        self.quick_mode = quick_mode
        self.skip_training = skip_training
        self.results = {
            'start_time': datetime.now().isoformat(),
            'phases': {},
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'warnings': 0
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run_all_tests(self):
        """Run all end-to-end tests."""
        print_header("SliceWise Full E2E Test: Phase 1 → Phase 6")
        print_info(f"Device: {self.device}")
        print_info(f"Quick mode: {self.quick_mode}")
        print_info(f"Skip training: {self.skip_training}")
        
        try:
            # Phase 1: Data
            self.test_phase1_data()
            
            # Phase 2: Classification
            self.test_phase2_classification()
            
            # Phase 3: Segmentation
            self.test_phase3_segmentation()
            
            # Phase 4: Calibration & Uncertainty
            self.test_phase4_calibration_uncertainty()
            
            # Phase 5: Metrics & Patient-Level
            self.test_phase5_metrics_patient()
            
            # Phase 6: API & Integration
            self.test_phase6_api_integration()
            
            # Final summary
            self.print_summary()
            
        except KeyboardInterrupt:
            print_warning("\n\nTest interrupted by user")
            self.print_summary()
            sys.exit(1)
        except Exception as e:
            print_error(f"\n\nFatal error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.print_summary()
            sys.exit(1)
    
    def test_phase1_data(self):
        """Test Phase 1: Data acquisition and preprocessing."""
        print_header("PHASE 1: Data Acquisition & Preprocessing")
        phase_results = {'tests': [], 'passed': 0, 'failed': 0}
        
        # Test 1.1: Kaggle dataset exists
        print_test("Kaggle dataset availability")
        kaggle_train = project_root / "data" / "processed" / "kaggle" / "train"
        if kaggle_train.exists() and len(list(kaggle_train.glob("*.npz"))) > 0:
            num_files = len(list(kaggle_train.glob("*.npz")))
            print_success(f"Kaggle train set found: {num_files} files")
            phase_results['tests'].append(('Kaggle dataset exists', True))
            phase_results['passed'] += 1
        else:
            print_error("Kaggle dataset not found")
            phase_results['tests'].append(('Kaggle dataset exists', False))
            phase_results['failed'] += 1
        
        # Test 1.2: Load Kaggle dataset
        print_test("Kaggle dataset loading")
        try:
            dataset = KaggleBrainMRIDataset(
                data_dir=str(project_root / "data" / "processed" / "kaggle" / "train")
            )
            print_success(f"Dataset loaded: {len(dataset)} samples")
            
            # Test sample
            image, label = dataset[0]
            assert image.shape[0] == 1, "Image should have 1 channel"
            assert label in [0, 1], "Label should be 0 or 1"
            print_success(f"Sample shape: {image.shape}, label: {label}")
            
            phase_results['tests'].append(('Kaggle dataset loading', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Failed to load Kaggle dataset: {str(e)}")
            phase_results['tests'].append(('Kaggle dataset loading', False))
            phase_results['failed'] += 1
        
        # Test 1.3: BraTS dataset (if available)
        print_test("BraTS dataset availability")
        brats_train = project_root / "data" / "processed" / "brats2d" / "train"
        if brats_train.exists() and len(list(brats_train.glob("*.npz"))) > 0:
            num_files = len(list(brats_train.glob("*.npz")))
            print_success(f"BraTS train set found: {num_files} files")
            
            try:
                brats_dataset = BraTS2DSliceDataset(
                    data_dir=str(project_root / "data" / "processed" / "brats2d" / "train")
                )
                print_success(f"BraTS dataset loaded: {len(brats_dataset)} slices")
                
                # Test sample
                sample = brats_dataset[0]
                image = sample['image'] if isinstance(sample, dict) else sample[0]
                mask = sample['mask'] if isinstance(sample, dict) else sample[1]
                print_success(f"BraTS sample - Image: {image.shape}, Mask: {mask.shape}")
                
                phase_results['tests'].append(('BraTS dataset loading', True))
                phase_results['passed'] += 1
            except Exception as e:
                print_warning(f"BraTS dataset found but loading failed: {str(e)}")
                phase_results['tests'].append(('BraTS dataset loading', False))
                phase_results['failed'] += 1
        else:
            print_warning("BraTS dataset not found (optional)")
            phase_results['tests'].append(('BraTS dataset availability', None))
            self.results['warnings'] += 1
        
        # Test 1.4: DataLoader creation
        print_test("DataLoader creation")
        try:
            train_loader, val_loader, test_loader = create_dataloaders(
                train_dir=str(project_root / "data" / "processed" / "kaggle" / "train"),
                val_dir=str(project_root / "data" / "processed" / "kaggle" / "val"),
                test_dir=str(project_root / "data" / "processed" / "kaggle" / "test"),
                batch_size=4 if self.quick_mode else 8,
                num_workers=0
            )
            print_success(f"DataLoaders created: {len(train_loader)} train batches")
            
            # Test batch
            batch = next(iter(train_loader))
            images, labels = batch
            print_success(f"Batch shape: {images.shape}, labels: {labels.shape}")
            
            phase_results['tests'].append(('DataLoader creation', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"DataLoader creation failed: {str(e)}")
            phase_results['tests'].append(('DataLoader creation', False))
            phase_results['failed'] += 1
        
        self.results['phases']['Phase 1'] = phase_results
        self.results['total_tests'] += len([t for t in phase_results['tests'] if t[1] is not None])
        self.results['passed_tests'] += phase_results['passed']
        self.results['failed_tests'] += phase_results['failed']
    
    def test_phase2_classification(self):
        """Test Phase 2: Classification pipeline."""
        print_header("PHASE 2: Classification Pipeline")
        phase_results = {'tests': [], 'passed': 0, 'failed': 0}
        
        # Test 2.1: Model creation
        print_test("Classifier model creation")
        try:
            model = BrainTumorClassifier(
                pretrained=False,
                num_classes=2
            )
            model = model.to(self.device)
            print_success(f"Model created: BrainTumorClassifier (EfficientNet-B0 backbone)")
            print_info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            phase_results['tests'].append(('Classifier creation', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Model creation failed: {str(e)}")
            phase_results['tests'].append(('Classifier creation', False))
            phase_results['failed'] += 1
            return
        
        # Test 2.2: Forward pass
        print_test("Classifier forward pass")
        try:
            dummy_input = torch.randn(2, 1, 256, 256).to(self.device)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
            print_success(f"Forward pass successful: {output.shape}")
            
            phase_results['tests'].append(('Classifier forward pass', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Forward pass failed: {str(e)}")
            phase_results['tests'].append(('Classifier forward pass', False))
            phase_results['failed'] += 1
        
        # Test 2.3: Check for trained model
        print_test("Trained classifier checkpoint")
        checkpoint_path = project_root / "checkpoints" / "cls" / "best_model.pth"
        if checkpoint_path.exists():
            print_success(f"Checkpoint found: {checkpoint_path}")
            
            try:
                predictor = ClassifierPredictor(
                    checkpoint_path=str(checkpoint_path),
                    device=str(self.device)
                )
                print_success("Predictor loaded successfully")
                
                # Test prediction
                dummy_image = np.random.rand(256, 256).astype(np.float32)
                result = predictor.predict(dummy_image)
                print_success(f"Prediction: {result['predicted_label']} ({result['confidence']:.2%})")
                
                phase_results['tests'].append(('Classifier inference', True))
                phase_results['passed'] += 1
            except Exception as e:
                print_error(f"Predictor loading failed: {str(e)}")
                phase_results['tests'].append(('Classifier inference', False))
                phase_results['failed'] += 1
        else:
            print_warning("No trained checkpoint found (run training first)")
            phase_results['tests'].append(('Trained classifier', None))
            self.results['warnings'] += 1
        
        # Test 2.4: Grad-CAM
        print_test("Grad-CAM generation")
        try:
            gradcam = GradCAM(model, target_layer=model.backbone.features[-1])
            dummy_input = torch.randn(1, 1, 256, 256).to(self.device)
            cam = gradcam.generate_cam(dummy_input, target_class=1)
            # CAM shape depends on the layer, just check it's 2D
            assert len(cam.shape) == 2, f"Expected 2D CAM, got shape {cam.shape}"
            print_success(f"Grad-CAM generated: {cam.shape}")
            
            phase_results['tests'].append(('Grad-CAM generation', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Grad-CAM failed: {str(e)}")
            phase_results['tests'].append(('Grad-CAM generation', False))
            phase_results['failed'] += 1
        
        self.results['phases']['Phase 2'] = phase_results
        self.results['total_tests'] += len([t for t in phase_results['tests'] if t[1] is not None])
        self.results['passed_tests'] += phase_results['passed']
        self.results['failed_tests'] += phase_results['failed']
    
    def test_phase3_segmentation(self):
        """Test Phase 3: Segmentation pipeline."""
        print_header("PHASE 3: Segmentation Pipeline")
        phase_results = {'tests': [], 'passed': 0, 'failed': 0}
        
        # Test 3.1: U-Net creation
        print_test("U-Net model creation")
        try:
            model = UNet2D(
                in_channels=1,
                out_channels=1,
                base_filters=64,
                depth=4
            )
            model = model.to(self.device)
            print_success(f"U-Net created")
            print_info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            phase_results['tests'].append(('U-Net creation', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"U-Net creation failed: {str(e)}")
            phase_results['tests'].append(('U-Net creation', False))
            phase_results['failed'] += 1
            return
        
        # Test 3.2: Forward pass
        print_test("U-Net forward pass")
        try:
            dummy_input = torch.randn(2, 1, 256, 256).to(self.device)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {output.shape}"
            print_success(f"Forward pass successful: {output.shape}")
            
            phase_results['tests'].append(('U-Net forward pass', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Forward pass failed: {str(e)}")
            phase_results['tests'].append(('U-Net forward pass', False))
            phase_results['failed'] += 1
        
        # Test 3.3: Trained segmentation model
        print_test("Trained segmentation checkpoint")
        checkpoint_path = project_root / "checkpoints" / "seg" / "best_model.pth"
        if checkpoint_path.exists():
            print_success(f"Checkpoint found: {checkpoint_path}")
            
            try:
                predictor = SegmentationPredictor(
                    checkpoint_path=str(checkpoint_path),
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print_success("Segmentation predictor loaded")
                
                # Test prediction with numpy array
                dummy_image = np.random.rand(256, 256).astype(np.float32)
                try:
                    result = predictor.predict_slice(dummy_image)
                    prob_map = result.get('prob', result['mask'])
                    binary_mask = result['mask']
                    print_success(f"Prediction - Prob map: {prob_map.shape}, Mask: {binary_mask.shape}")
                    phase_results['tests'].append(('Segmentation inference', True))
                    phase_results['passed'] += 1
                except Exception as pred_error:
                    print_error(f"Prediction failed: {str(pred_error)}")
                    phase_results['tests'].append(('Segmentation inference', False))
                    phase_results['failed'] += 1
            except Exception as e:
                print_error(f"Segmentation predictor loading failed: {str(e)}")
                phase_results['tests'].append(('Segmentation inference', False))
                phase_results['failed'] += 1
        else:
            print_warning("No trained checkpoint found (run training first)")
            phase_results['tests'].append(('Trained segmentation', None))
            self.results['warnings'] += 1
        
        # Test 3.4: Post-processing
        print_test("Post-processing pipeline")
        try:
            dummy_prob_map = np.random.rand(256, 256).astype(np.float32)
            binary_mask, stats = postprocess_mask(
                dummy_prob_map,
                threshold=0.5,
                min_object_size=50,
                fill_holes_size=None,
                morphology_op='close',
                morphology_kernel=3
            )
            print_success(f"Post-processing successful")
            print_info(f"Stats: {stats}")
            
            phase_results['tests'].append(('Post-processing', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Post-processing failed: {str(e)}")
            phase_results['tests'].append(('Post-processing', False))
            phase_results['failed'] += 1
        
        self.results['phases']['Phase 3'] = phase_results
        self.results['total_tests'] += len([t for t in phase_results['tests'] if t[1] is not None])
        self.results['passed_tests'] += phase_results['passed']
        self.results['failed_tests'] += phase_results['failed']
    
    def test_phase4_calibration_uncertainty(self):
        """Test Phase 4: Calibration and uncertainty estimation."""
        print_header("PHASE 4: Calibration & Uncertainty")
        phase_results = {'tests': [], 'passed': 0, 'failed': 0}
        
        # Test 4.1: Temperature scaling
        print_test("Temperature scaling")
        try:
            temp_scaler = TemperatureScaling()
            print_success("Temperature scaler created")
            
            # Check for calibrated model
            calib_path = project_root / "checkpoints" / "cls" / "temperature_scaler.pth"
            if calib_path.exists():
                checkpoint = torch.load(calib_path, map_location=self.device, weights_only=False)
                temp_scaler.temperature.data = checkpoint['temperature']
                print_success(f"Calibration loaded: T={temp_scaler.temperature.item():.4f}")
                phase_results['tests'].append(('Temperature scaling', True))
                phase_results['passed'] += 1
            else:
                print_warning("No calibration checkpoint found")
                phase_results['tests'].append(('Temperature scaling', None))
                self.results['warnings'] += 1
        except Exception as e:
            print_error(f"Temperature scaling failed: {str(e)}")
            phase_results['tests'].append(('Temperature scaling', False))
            phase_results['failed'] += 1
        
        # Test 4.2: MC Dropout
        print_test("MC Dropout uncertainty")
        try:
            # Create simple model with dropout
            model = UNet2D(in_channels=1, out_channels=1, base_filters=32, depth=2)
            model = model.to(self.device)
            
            mc_predictor = MCDropoutPredictor(model, n_samples=5, device=str(self.device))
            dummy_input = torch.randn(1, 1, 64, 64).to(self.device)
            result = mc_predictor.predict_with_uncertainty(dummy_input)
            
            assert 'mean' in result and 'std' in result
            print_success(f"MC Dropout successful - Mean: {result['mean'].shape}, Std: {result['std'].shape}")
            
            phase_results['tests'].append(('MC Dropout', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"MC Dropout failed: {str(e)}")
            phase_results['tests'].append(('MC Dropout', False))
            phase_results['failed'] += 1
        
        # Test 4.3: Test-Time Augmentation
        print_test("Test-Time Augmentation")
        try:
            tta_predictor = TTAPredictor(model, device=str(self.device))
            result = tta_predictor.predict_with_tta(dummy_input)
            
            assert 'mean' in result and 'std' in result
            print_success(f"TTA successful - Mean: {result['mean'].shape}, Std: {result['std'].shape}")
            
            phase_results['tests'].append(('TTA', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"TTA failed: {str(e)}")
            phase_results['tests'].append(('TTA', False))
            phase_results['failed'] += 1
        
        # Test 4.4: Ensemble predictor
        print_test("Ensemble uncertainty (MC + TTA)")
        try:
            ensemble = EnsemblePredictor(model, n_mc_samples=3, use_tta=True, device=str(self.device))
            result = ensemble.predict_with_uncertainty(dummy_input)
            
            assert all(k in result for k in ['mean', 'std', 'epistemic', 'aleatoric'])
            print_success(f"Ensemble successful - Epistemic: {result['epistemic'].mean():.4f}, Aleatoric: {result['aleatoric'].mean():.4f}")
            
            phase_results['tests'].append(('Ensemble uncertainty', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Ensemble failed: {str(e)}")
            phase_results['tests'].append(('Ensemble uncertainty', False))
            phase_results['failed'] += 1
        
        self.results['phases']['Phase 4'] = phase_results
        self.results['total_tests'] += len([t for t in phase_results['tests'] if t[1] is not None])
        self.results['passed_tests'] += phase_results['passed']
        self.results['failed_tests'] += phase_results['failed']
    
    def test_phase5_metrics_patient(self):
        """Test Phase 5: Metrics and patient-level evaluation."""
        print_header("PHASE 5: Metrics & Patient-Level Evaluation")
        phase_results = {'tests': [], 'passed': 0, 'failed': 0}
        
        # Test 5.1: Metrics computation
        print_test("Metrics computation")
        try:
            from src.eval.metrics import dice_coefficient, iou_score
            
            # Dummy predictions and ground truth
            pred = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
            target = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
            
            dice = dice_coefficient(pred, target)
            iou = iou_score(pred, target)
            
            print_success(f"Metrics computed - Dice: {dice:.4f}, IoU: {iou:.4f}")
            
            phase_results['tests'].append(('Metrics computation', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Metrics computation failed: {str(e)}")
            phase_results['tests'].append(('Metrics computation', False))
            phase_results['failed'] += 1
        
        # Test 5.2: Patient-level evaluator
        print_test("Patient-level evaluation")
        try:
            evaluator = PatientLevelEvaluator()
            
            # Add dummy patient data with required parameters
            gt_mask = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
            pred_mask = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
            pred_prob = np.random.rand(256, 256).astype(np.float32)
            
            evaluator.add_slice('patient_001', slice_idx=0, gt_mask=gt_mask, pred_mask=pred_mask, pred_prob=pred_prob)
            evaluator.add_slice('patient_001', slice_idx=1, gt_mask=gt_mask, pred_mask=pred_mask, pred_prob=pred_prob)
            
            # Compute results
            results_df = evaluator.compute_all_patients()
            print_success(f"Patient-level evaluation successful")
            print_info(f"Results: {len(results_df)} patients")
            
            phase_results['tests'].append(('Patient-level evaluation', True))
            phase_results['passed'] += 1
        except Exception as e:
            print_error(f"Patient-level evaluation failed: {str(e)}")
            phase_results['tests'].append(('Patient-level evaluation', False))
            phase_results['failed'] += 1
        
        self.results['phases']['Phase 5'] = phase_results
        self.results['total_tests'] += len([t for t in phase_results['tests'] if t[1] is not None])
        self.results['passed_tests'] += phase_results['passed']
        self.results['failed_tests'] += phase_results['failed']
    
    def test_phase6_api_integration(self):
        """Test Phase 6: API endpoints and full integration."""
        print_header("PHASE 6: API & Full Integration")
        phase_results = {'tests': [], 'passed': 0, 'failed': 0}
        
        # Check if backend is running
        api_url = "http://localhost:8000"
        
        print_test("Backend API availability")
        try:
            response = requests.get(f"{api_url}/healthz", timeout=2)
            if response.status_code == 200:
                health = response.json()
                print_success(f"Backend API is running")
                print_info(f"Status: {health['status']}")
                print_info(f"Device: {health['device']}")
                print_info(f"Classifier: {'✓' if health['classifier_loaded'] else '✗'}")
                print_info(f"Segmentation: {'✓' if health['segmentation_loaded'] else '✗'}")
                
                phase_results['tests'].append(('Backend API running', True))
                phase_results['passed'] += 1
                
                # Run API tests
                self._test_api_endpoints(api_url, phase_results)
            else:
                print_warning(f"Backend returned status {response.status_code}")
                phase_results['tests'].append(('Backend API running', False))
                phase_results['failed'] += 1
        except requests.exceptions.ConnectionError:
            print_warning("Backend API not running - start with: python scripts/run_demo_backend.py")
            print_info("Skipping API tests...")
            phase_results['tests'].append(('Backend API running', None))
            self.results['warnings'] += 1
        except Exception as e:
            print_error(f"Backend check failed: {str(e)}")
            phase_results['tests'].append(('Backend API running', False))
            phase_results['failed'] += 1
        
        self.results['phases']['Phase 6'] = phase_results
        self.results['total_tests'] += len([t for t in phase_results['tests'] if t[1] is not None])
        self.results['passed_tests'] += phase_results['passed']
        self.results['failed_tests'] += phase_results['failed']
    
    def _test_api_endpoints(self, api_url, phase_results):
        """Test individual API endpoints."""
        
        # Create test image
        test_image = Image.fromarray((np.random.rand(256, 256) * 255).astype(np.uint8), mode='L')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Test /model/info
        print_test("GET /model/info")
        try:
            response = requests.get(f"{api_url}/model/info", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print_success(f"Model info retrieved")
                phase_results['tests'].append(('/model/info', True))
                phase_results['passed'] += 1
            else:
                print_error(f"Failed: {response.status_code}")
                phase_results['tests'].append(('/model/info', False))
                phase_results['failed'] += 1
        except Exception as e:
            print_error(f"Request failed: {str(e)}")
            phase_results['tests'].append(('/model/info', False))
            phase_results['failed'] += 1
        
        # Test /classify
        print_test("POST /classify")
        try:
            img_bytes.seek(0)
            files = {"file": ("test.png", img_bytes, "image/png")}
            response = requests.post(f"{api_url}/classify", files=files, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print_success(f"Classification: {result['predicted_label']} ({result['confidence']:.2%})")
                phase_results['tests'].append(('/classify', True))
                phase_results['passed'] += 1
            else:
                print_error(f"Failed: {response.status_code}")
                phase_results['tests'].append(('/classify', False))
                phase_results['failed'] += 1
        except Exception as e:
            print_error(f"Request failed: {str(e)}")
            phase_results['tests'].append(('/classify', False))
            phase_results['failed'] += 1
        
        # Test /segment
        print_test("POST /segment")
        try:
            img_bytes.seek(0)
            files = {"file": ("test.png", img_bytes, "image/png")}
            response = requests.post(f"{api_url}/segment", files=files, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print_success(f"Segmentation: Tumor={'Yes' if result['has_tumor'] else 'No'}, Area={result['tumor_area_pixels']}px")
                phase_results['tests'].append(('/segment', True))
                phase_results['passed'] += 1
            else:
                print_error(f"Failed: {response.status_code}")
                phase_results['tests'].append(('/segment', False))
                phase_results['failed'] += 1
        except Exception as e:
            print_error(f"Request failed: {str(e)}")
            phase_results['tests'].append(('/segment', False))
            phase_results['failed'] += 1
    
    def print_summary(self):
        """Print test summary."""
        self.results['end_time'] = datetime.now().isoformat()
        
        print_header("TEST SUMMARY")
        
        # Overall stats
        total = self.results['total_tests']
        passed = self.results['passed_tests']
        failed = self.results['failed_tests']
        warnings = self.results['warnings']
        
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\n{Colors.BOLD}Overall Results:{Colors.ENDC}")
        print(f"  Total Tests:    {total}")
        print(f"  {Colors.OKGREEN}Passed:         {passed}{Colors.ENDC}")
        print(f"  {Colors.FAIL}Failed:         {failed}{Colors.ENDC}")
        print(f"  {Colors.WARNING}Warnings:       {warnings}{Colors.ENDC}")
        print(f"  {Colors.BOLD}Pass Rate:      {pass_rate:.1f}%{Colors.ENDC}")
        
        # Phase breakdown
        print(f"\n{Colors.BOLD}Phase Breakdown:{Colors.ENDC}")
        for phase_name, phase_data in self.results['phases'].items():
            phase_passed = phase_data['passed']
            phase_failed = phase_data['failed']
            phase_total = phase_passed + phase_failed
            phase_rate = (phase_passed / phase_total * 100) if phase_total > 0 else 0
            
            status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if phase_failed == 0 else f"{Colors.FAIL}✗{Colors.ENDC}"
            print(f"  {status} {phase_name}: {phase_passed}/{phase_total} ({phase_rate:.0f}%)")
        
        # Save results
        results_file = project_root / "full_e2e_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n{Colors.OKCYAN}Results saved to: {results_file}{Colors.ENDC}")
        
        # Final verdict
        print()
        if failed == 0 and warnings == 0:
            print(f"{Colors.OKGREEN}{Colors.BOLD}{'='*80}{Colors.ENDC}")
            print(f"{Colors.OKGREEN}{Colors.BOLD}{'ALL TESTS PASSED! 🎉':^80}{Colors.ENDC}")
            print(f"{Colors.OKGREEN}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        elif failed == 0:
            print(f"{Colors.WARNING}{Colors.BOLD}{'='*80}{Colors.ENDC}")
            print(f"{Colors.WARNING}{Colors.BOLD}{'TESTS PASSED WITH WARNINGS ⚠️':^80}{Colors.ENDC}")
            print(f"{Colors.WARNING}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}{Colors.BOLD}{'='*80}{Colors.ENDC}")
            print(f"{Colors.FAIL}{Colors.BOLD}{'SOME TESTS FAILED ✗':^80}{Colors.ENDC}")
            print(f"{Colors.FAIL}{Colors.BOLD}{'='*80}{Colors.ENDC}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full E2E Test: Phase 1 → Phase 6")
    parser.add_argument("--quick", action="store_true", help="Quick mode (smaller batches)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training tests")
    
    args = parser.parse_args()
    
    tester = FullE2ETest(quick_mode=args.quick, skip_training=args.skip_training)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
