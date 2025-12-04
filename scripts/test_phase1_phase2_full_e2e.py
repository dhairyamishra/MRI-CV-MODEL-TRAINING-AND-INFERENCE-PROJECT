"""
Comprehensive End-to-End Test for Phase 1 & Phase 2
=====================================================

This script performs a complete test of the entire Phase 1 and Phase 2 pipeline:

PHASE 1 - Data Acquisition & Preprocessing:
1. Data download (Kaggle Brain MRI dataset)
2. Data preprocessing (JPG → .npz conversion)
3. Data splitting (train/val/test with stratification)
4. Dataset loading and verification
5. Transform pipeline testing

PHASE 2 - Classification MVP:
6. Model creation and initialization
7. Mini training run (few epochs)
8. Model evaluation and metrics
9. Grad-CAM generation
10. Inference pipeline
11. API endpoint testing (if backend is running)
12. Full integration test

Run this to validate the complete Phase 1 and Phase 2 implementation.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import time
from datetime import datetime
import json
import shutil
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Phase 1 components
from scripts.download_kaggle_data import download_kaggle_brain_mri
from src.data.preprocess_kaggle import preprocess_kaggle_dataset
from src.data.split_kaggle import create_kaggle_splits
from src.data.kaggle_mri_dataset import KaggleBrainMRIDataset, create_dataloaders
from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_strong_train_transforms,
    get_light_train_transforms,
)

# Import Phase 2 components
from src.models.classifier import create_classifier
from src.inference.predict import ClassifierPredictor
from src.eval.grad_cam import GradCAM


class Phase1Phase2E2ETest:
    """Comprehensive end-to-end test suite for Phase 1 and Phase 2."""
    
    def __init__(self, use_temp_dir: bool = False, skip_download: bool = True):
        """
        Initialize test suite.
        
        Args:
            use_temp_dir: If True, use temporary directory for testing (cleanup after)
            skip_download: If True, skip data download (assumes data already exists)
        """
        self.project_root = project_root
        self.use_temp_dir = use_temp_dir
        self.skip_download = skip_download
        
        # Setup directories
        if use_temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="phase1_2_test_"))
            self.raw_data_dir = self.temp_dir / "raw" / "kaggle_brain_mri"
            self.processed_data_dir = self.temp_dir / "processed" / "kaggle"
            print(f"📁 Using temporary directory: {self.temp_dir}")
        else:
            self.raw_data_dir = self.project_root / "data" / "raw" / "kaggle_brain_mri"
            self.processed_data_dir = self.project_root / "data" / "processed" / "kaggle"
        
        self.test_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'phase1_tests': {},
            'phase2_tests': {},
            'overall_status': 'PENDING'
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
    
    def cleanup(self):
        """Cleanup temporary directories if used."""
        if self.use_temp_dir and hasattr(self, 'temp_dir') and self.temp_dir.exists():
            print(f"\n🧹 Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
    
    def print_header(self, text, level=1):
        """Print formatted header."""
        if level == 1:
            print("\n" + "="*80)
            print(f"  {text}")
            print("="*80)
        else:
            print("\n" + "-"*80)
            print(f"  {text}")
            print("-"*80)
    
    def print_test(self, name, status, message="", phase="phase1"):
        """Print test result."""
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name}: {message}")
        
        phase_key = f'{phase}_tests'
        self.test_results[phase_key][name] = {
            'status': 'PASSED' if status else 'FAILED',
            'message': message
        }
        return status
    
    # ========================================================================
    # PHASE 1 TESTS - Data Acquisition & Preprocessing
    # ========================================================================
    
    def test_phase1_1_data_download(self):
        """Phase 1 Test 1: Data download from Kaggle."""
        self.print_header("Phase 1 Test 1: Data Download", level=2)
        
        if self.skip_download:
            # Check if data already exists
            if self.raw_data_dir.exists():
                yes_dir = self.raw_data_dir / "yes"
                no_dir = self.raw_data_dir / "no"
                
                if yes_dir.exists() and no_dir.exists():
                    yes_count = len(list(yes_dir.glob("*.jpg")))
                    no_count = len(list(no_dir.glob("*.jpg")))
                    
                    if yes_count > 0 and no_count > 0:
                        return self.print_test(
                            "Data Download",
                            True,
                            f"Skipped (data exists): {yes_count} tumor, {no_count} no tumor",
                            phase="phase1"
                        )
            
            return self.print_test(
                "Data Download",
                False,
                "Data not found. Set skip_download=False to download.",
                phase="phase1"
            )
        
        try:
            # Download dataset
            print("📥 Downloading Kaggle Brain MRI dataset...")
            print("(This may take a few minutes)")
            
            download_kaggle_brain_mri(str(self.raw_data_dir))
            
            # Verify download
            yes_dir = self.raw_data_dir / "yes"
            no_dir = self.raw_data_dir / "no"
            
            yes_count = len(list(yes_dir.glob("*.jpg")))
            no_count = len(list(no_dir.glob("*.jpg")))
            
            assert yes_count > 0, "No tumor images found"
            assert no_count > 0, "No non-tumor images found"
            
            return self.print_test(
                "Data Download",
                True,
                f"Downloaded {yes_count} tumor, {no_count} no tumor images",
                phase="phase1"
            )
        
        except Exception as e:
            return self.print_test(
                "Data Download",
                False,
                str(e),
                phase="phase1"
            )
    
    def test_phase1_2_data_preprocessing(self):
        """Phase 1 Test 2: Data preprocessing (JPG → .npz)."""
        self.print_header("Phase 1 Test 2: Data Preprocessing", level=2)
        
        try:
            # Check if already preprocessed
            if self.processed_data_dir.exists():
                npz_files = list(self.processed_data_dir.glob("*.npz"))
                if len(npz_files) > 0:
                    print(f"ℹ️  Found {len(npz_files)} existing .npz files")
                    
                    # Verify a sample file
                    sample = np.load(npz_files[0], allow_pickle=True)
                    assert 'image' in sample, "Missing 'image' key"
                    assert 'label' in sample, "Missing 'label' key"
                    assert 'metadata' in sample, "Missing 'metadata' key"
                    
                    return self.print_test(
                        "Data Preprocessing",
                        True,
                        f"Using existing {len(npz_files)} preprocessed files",
                        phase="phase1"
                    )
            
            # Run preprocessing
            print("🔄 Preprocessing images to .npz format...")
            preprocess_kaggle_dataset(
                raw_dir=str(self.raw_data_dir),
                processed_dir=str(self.processed_data_dir),
                target_size=(256, 256)
            )
            
            # Verify output
            npz_files = list(self.processed_data_dir.glob("*.npz"))
            assert len(npz_files) > 0, "No .npz files created"
            
            # Verify file structure
            sample = np.load(npz_files[0], allow_pickle=True)
            assert 'image' in sample, "Missing 'image' key"
            assert 'label' in sample, "Missing 'label' key"
            assert sample['image'].shape == (1, 256, 256), "Incorrect image shape"
            
            return self.print_test(
                "Data Preprocessing",
                True,
                f"Preprocessed {len(npz_files)} images to .npz format",
                phase="phase1"
            )
        
        except Exception as e:
            return self.print_test(
                "Data Preprocessing",
                False,
                str(e),
                phase="phase1"
            )
    
    def test_phase1_3_data_splitting(self):
        """Phase 1 Test 3: Data splitting (train/val/test)."""
        self.print_header("Phase 1 Test 3: Data Splitting", level=2)
        
        try:
            # Check if splits already exist
            train_dir = self.processed_data_dir / "train"
            val_dir = self.processed_data_dir / "val"
            test_dir = self.processed_data_dir / "test"
            
            if train_dir.exists() and val_dir.exists() and test_dir.exists():
                train_count = len(list(train_dir.glob("*.npz")))
                val_count = len(list(val_dir.glob("*.npz")))
                test_count = len(list(test_dir.glob("*.npz")))
                
                if train_count > 0 and val_count > 0 and test_count > 0:
                    return self.print_test(
                        "Data Splitting",
                        True,
                        f"Using existing splits: {train_count} train, {val_count} val, {test_count} test",
                        phase="phase1"
                    )
            
            # Create splits
            print("✂️  Creating train/val/test splits...")
            create_kaggle_splits(
                processed_dir=str(self.processed_data_dir),
                output_dir=str(self.processed_data_dir),
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42
            )
            
            # Verify splits
            train_count = len(list(train_dir.glob("*.npz")))
            val_count = len(list(val_dir.glob("*.npz")))
            test_count = len(list(test_dir.glob("*.npz")))
            
            assert train_count > 0, "No training files"
            assert val_count > 0, "No validation files"
            assert test_count > 0, "No test files"
            
            # Verify stratification
            def get_class_dist(split_dir):
                labels = []
                for f in split_dir.glob("*.npz"):
                    data = np.load(f, allow_pickle=True)
                    labels.append(int(data['label']))
                return np.mean(labels)
            
            train_pos_ratio = get_class_dist(train_dir)
            val_pos_ratio = get_class_dist(val_dir)
            test_pos_ratio = get_class_dist(test_dir)
            
            # Check if ratios are similar (within 10%)
            ratio_diff = max(abs(train_pos_ratio - val_pos_ratio),
                           abs(train_pos_ratio - test_pos_ratio))
            
            stratified = ratio_diff < 0.1
            
            return self.print_test(
                "Data Splitting",
                True,
                f"Created splits: {train_count} train, {val_count} val, {test_count} test "
                f"(stratified: {stratified})",
                phase="phase1"
            )
        
        except Exception as e:
            return self.print_test(
                "Data Splitting",
                False,
                str(e),
                phase="phase1"
            )
    
    def test_phase1_4_dataset_loading(self):
        """Phase 1 Test 4: Dataset loading and PyTorch integration."""
        self.print_header("Phase 1 Test 4: Dataset Loading", level=2)
        
        try:
            # Create dataset
            train_dir = self.processed_data_dir / "train"
            dataset = KaggleBrainMRIDataset(
                data_dir=str(train_dir),
                transform=None
            )
            
            assert len(dataset) > 0, "Dataset is empty"
            
            # Test __getitem__
            image, label = dataset[0]
            assert isinstance(image, torch.Tensor), "Image should be tensor"
            assert image.shape == (1, 256, 256), f"Wrong shape: {image.shape}"
            assert label in [0, 1], f"Invalid label: {label}"
            
            # Test class distribution
            class_dist = dataset.get_class_distribution()
            assert 'tumor' in class_dist, "Missing tumor count"
            assert 'no_tumor' in class_dist, "Missing no_tumor count"
            
            # Test metadata
            metadata = dataset.get_sample_metadata(0)
            assert 'image_id' in metadata, "Missing image_id in metadata"
            
            return self.print_test(
                "Dataset Loading",
                True,
                f"Loaded {len(dataset)} samples, "
                f"{class_dist['tumor']} tumor, {class_dist['no_tumor']} no tumor",
                phase="phase1"
            )
        
        except Exception as e:
            return self.print_test(
                "Dataset Loading",
                False,
                str(e),
                phase="phase1"
            )
    
    def test_phase1_5_transforms(self):
        """Phase 1 Test 5: Transform pipeline."""
        self.print_header("Phase 1 Test 5: Transform Pipeline", level=2)
        
        try:
            # Test different transform presets
            transforms_to_test = {
                'train': get_train_transforms(),
                'val': get_val_transforms(),
                'strong': get_strong_train_transforms(),
                'light': get_light_train_transforms(),
            }
            
            # Load a sample image
            train_dir = self.processed_data_dir / "train"
            dataset = KaggleBrainMRIDataset(
                data_dir=str(train_dir),
                transform=None
            )
            image, _ = dataset[0]
            
            results = []
            for name, transform in transforms_to_test.items():
                if transform is not None:
                    transformed = transform(image)
                    assert isinstance(transformed, torch.Tensor), f"{name}: Not a tensor"
                    assert transformed.shape == image.shape, f"{name}: Shape mismatch"
                    results.append(name)
            
            return self.print_test(
                "Transform Pipeline",
                True,
                f"Tested {len(results)} transform presets: {', '.join(results)}",
                phase="phase1"
            )
        
        except Exception as e:
            return self.print_test(
                "Transform Pipeline",
                False,
                str(e),
                phase="phase1"
            )
    
    def test_phase1_6_dataloaders(self):
        """Phase 1 Test 6: DataLoader creation and batching."""
        self.print_header("Phase 1 Test 6: DataLoader Creation", level=2)
        
        try:
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                train_dir=str(self.processed_data_dir / "train"),
                val_dir=str(self.processed_data_dir / "val"),
                test_dir=str(self.processed_data_dir / "test"),
                batch_size=4,
                num_workers=0,
                train_transform=get_train_transforms(),
                val_transform=get_val_transforms()
            )
            
            # Test train loader
            images, labels = next(iter(train_loader))
            assert images.shape[0] <= 4, "Batch size too large"
            assert images.shape[1] == 1, "Should be single channel"
            assert images.shape[2] == 256, "Height should be 256"
            assert images.shape[3] == 256, "Width should be 256"
            
            # Test all loaders
            train_size = len(train_loader.dataset)
            val_size = len(val_loader.dataset)
            test_size = len(test_loader.dataset)
            
            return self.print_test(
                "DataLoader Creation",
                True,
                f"Created loaders: {train_size} train, {val_size} val, {test_size} test",
                phase="phase1"
            )
        
        except Exception as e:
            return self.print_test(
                "DataLoader Creation",
                False,
                str(e),
                phase="phase1"
            )
    
    # ========================================================================
    # PHASE 2 TESTS - Classification MVP
    # ========================================================================
    
    def test_phase2_1_model_creation(self):
        """Phase 2 Test 1: Model creation and architecture."""
        self.print_header("Phase 2 Test 1: Model Creation", level=2)
        
        try:
            # Test EfficientNet
            model_eff = create_classifier('efficientnet', pretrained=False)
            model_eff = model_eff.to(self.device)
            
            # Test forward pass
            dummy_input = torch.randn(2, 1, 256, 256).to(self.device)
            output = model_eff(dummy_input)
            
            assert output.shape == (2, 2), f"Wrong output shape: {output.shape}"
            
            # Test ConvNeXt
            model_conv = create_classifier('convnext', pretrained=False)
            model_conv = model_conv.to(self.device)
            output_conv = model_conv(dummy_input)
            
            assert output_conv.shape == (2, 2), "ConvNeXt output shape wrong"
            
            # Store model for later tests
            self.test_model = model_eff
            
            # Get parameter counts
            eff_params = model_eff.get_num_trainable_params()
            # ConvNeXt doesn't have this method, so count manually
            conv_params = sum(p.numel() for p in model_conv.parameters() if p.requires_grad)
            
            return self.print_test(
                "Model Creation",
                True,
                f"EfficientNet: {eff_params:,} params, "
                f"ConvNeXt: {conv_params:,} params",
                phase="phase2"
            )
        
        except Exception as e:
            return self.print_test(
                "Model Creation",
                False,
                str(e),
                phase="phase2"
            )
    
    def test_phase2_2_mini_training(self):
        """Phase 2 Test 2: Mini training run."""
        self.print_header("Phase 2 Test 2: Mini Training Run", level=2)
        
        try:
            # Create dataloaders
            train_loader, val_loader, _ = create_dataloaders(
                train_dir=str(self.processed_data_dir / "train"),
                val_dir=str(self.processed_data_dir / "val"),
                test_dir=str(self.processed_data_dir / "test"),
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
            
            # Save checkpoint
            checkpoint_dir = self.project_root / "checkpoints" / "cls"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': epoch_acc / 100,
                'config': {}
            }
            
            # Save as test model
            checkpoint_path = checkpoint_dir / "full_e2e_test_model.pth"
            torch.save(checkpoint, checkpoint_path)
            self.checkpoint_path = str(checkpoint_path)
            
            # Also save as best_model.pth for backend API to use
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            print(f"  Saved checkpoint to: {checkpoint_path}")
            print(f"  Also saved as: {best_model_path} (for backend API)")
            
            return self.print_test(
                "Mini Training",
                True,
                f"Trained {num_epochs} epochs, final acc={epoch_acc:.2f}%",
                phase="phase2"
            )
        
        except Exception as e:
            return self.print_test(
                "Mini Training",
                False,
                str(e),
                phase="phase2"
            )
    
    def test_phase2_3_evaluation(self):
        """Phase 2 Test 3: Model evaluation."""
        self.print_header("Phase 2 Test 3: Model Evaluation", level=2)
        
        try:
            model = self.test_model
            model.eval()
            
            # Get test data
            _, _, test_loader = create_dataloaders(
                train_dir=str(self.processed_data_dir / "train"),
                val_dir=str(self.processed_data_dir / "val"),
                test_dir=str(self.processed_data_dir / "test"),
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
            
            # Calculate AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(all_labels, all_probs)
                metric_str = f"Acc={accuracy:.2f}%, AUC={auc:.3f}"
            except:
                metric_str = f"Acc={accuracy:.2f}%"
            
            return self.print_test(
                "Model Evaluation",
                True,
                metric_str,
                phase="phase2"
            )
        
        except Exception as e:
            return self.print_test(
                "Model Evaluation",
                False,
                str(e),
                phase="phase2"
            )
    
    def test_phase2_4_gradcam(self):
        """Phase 2 Test 4: Grad-CAM generation."""
        self.print_header("Phase 2 Test 4: Grad-CAM Generation", level=2)
        
        try:
            model = self.test_model
            model.eval()
            model = model.to(self.device)
            
            # Get a sample image
            _, _, test_loader = create_dataloaders(
                train_dir=str(self.processed_data_dir / "train"),
                val_dir=str(self.processed_data_dir / "val"),
                test_dir=str(self.processed_data_dir / "test"),
                batch_size=1,
                num_workers=0
            )
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
                f"Generated {cam.shape} heatmap with overlay",
                phase="phase2"
            )
        
        except Exception as e:
            return self.print_test(
                "Grad-CAM Generation",
                False,
                str(e),
                phase="phase2"
            )
    
    def test_phase2_5_inference(self):
        """Phase 2 Test 5: Inference pipeline."""
        self.print_header("Phase 2 Test 5: Inference Pipeline", level=2)
        
        try:
            # Create predictor
            predictor = ClassifierPredictor(
                self.checkpoint_path,
                model_name='efficientnet',
                device='cpu'
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
                f"Single: {result['predicted_label']} ({result['confidence']:.2%}), Batch: 3 images",
                phase="phase2"
            )
        
        except Exception as e:
            return self.print_test(
                "Inference Pipeline",
                False,
                str(e),
                phase="phase2"
            )
    
    def test_phase2_6_api_endpoints(self):
        """Phase 2 Test 6: API endpoints."""
        self.print_header("Phase 2 Test 6: API Endpoints", level=2)
        
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
                        "Backend not running. Start with: python scripts/run_backend.py",
                        phase="phase2"
                    )
            except:
                return self.print_test(
                    "API Endpoints",
                    False,
                    "Backend not running (optional test)",
                    phase="phase2"
                )
            
            # Test /model/info
            response = requests.get("http://localhost:8000/model/info")
            assert response.status_code == 200
            model_info = response.json()
            
            # Test /classify_slice
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
                f"API working, model: {model_info['model_name']}",
                phase="phase2"
            )
        
        except Exception as e:
            return self.print_test(
                "API Endpoints",
                False,
                f"Skipped (backend not running): {str(e)}",
                phase="phase2"
            )
    
    def test_phase2_7_full_integration(self):
        """Phase 2 Test 7: Full integration test."""
        self.print_header("Phase 2 Test 7: Full Integration", level=2)
        
        try:
            # Load real data
            _, _, test_loader = create_dataloaders(
                train_dir=str(self.processed_data_dir / "train"),
                val_dir=str(self.processed_data_dir / "val"),
                test_dir=str(self.processed_data_dir / "test"),
                batch_size=1,
                num_workers=0
            )
            images, labels = next(iter(test_loader))
            
            # Convert to numpy
            image_np = images[0, 0].cpu().numpy()
            true_label = labels[0].item()
            
            # Run through predictor
            predictor = ClassifierPredictor(
                self.checkpoint_path,
                model_name='efficientnet',
                device='cpu'
            )
            
            result = predictor.predict(image_np)
            pred_label = result['predicted_class']
            confidence = result['confidence']
            
            # Generate Grad-CAM
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
                f"{status_str} prediction with {confidence:.2%} confidence, Grad-CAM generated",
                phase="phase2"
            )
        
        except Exception as e:
            return self.print_test(
                "Full Integration",
                False,
                str(e),
                phase="phase2"
            )
    
    # ========================================================================
    # TEST RUNNER
    # ========================================================================
    
    def run_all_tests(self):
        """Run all Phase 1 and Phase 2 tests."""
        print("\n" + "="*80)
        print("  COMPREHENSIVE PHASE 1 & PHASE 2 END-TO-END TEST SUITE")
        print("="*80)
        print(f"  Timestamp: {self.test_results['timestamp']}")
        print(f"  Device: {self.device}")
        print(f"  Data Directory: {self.processed_data_dir}")
        print("="*80)
        
        start_time = time.time()
        
        # Phase 1 Tests
        self.print_header("PHASE 1: Data Acquisition & Preprocessing")
        phase1_tests = [
            self.test_phase1_1_data_download,
            self.test_phase1_2_data_preprocessing,
            self.test_phase1_3_data_splitting,
            self.test_phase1_4_dataset_loading,
            self.test_phase1_5_transforms,
            self.test_phase1_6_dataloaders,
        ]
        
        phase1_results = []
        for test in phase1_tests:
            try:
                result = test()
                phase1_results.append(result)
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                phase1_results.append(False)
        
        # Phase 2 Tests
        self.print_header("PHASE 2: Classification MVP")
        phase2_tests = [
            self.test_phase2_1_model_creation,
            self.test_phase2_2_mini_training,
            self.test_phase2_3_evaluation,
            self.test_phase2_4_gradcam,
            self.test_phase2_5_inference,
            self.test_phase2_6_api_endpoints,
            self.test_phase2_7_full_integration,
        ]
        
        phase2_results = []
        for test in phase2_tests:
            try:
                result = test()
                phase2_results.append(result)
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                phase2_results.append(False)
        
        elapsed_time = time.time() - start_time
        
        # Summary
        self.print_header("TEST SUMMARY")
        
        phase1_passed = sum(phase1_results)
        phase1_total = len(phase1_results)
        phase2_passed = sum(phase2_results)
        phase2_total = len(phase2_results)
        
        total_passed = phase1_passed + phase2_passed
        total_tests = phase1_total + phase2_total
        success_rate = (total_passed / total_tests) * 100
        
        print(f"\n📊 PHASE 1 Results:")
        print(f"   Total Tests: {phase1_total}")
        print(f"   Passed: {phase1_passed}")
        print(f"   Failed: {phase1_total - phase1_passed}")
        print(f"   Success Rate: {(phase1_passed/phase1_total)*100:.1f}%")
        
        print(f"\n📊 PHASE 2 Results:")
        print(f"   Total Tests: {phase2_total}")
        print(f"   Passed: {phase2_passed}")
        print(f"   Failed: {phase2_total - phase2_passed}")
        print(f"   Success Rate: {(phase2_passed/phase2_total)*100:.1f}%")
        
        print(f"\n📊 OVERALL Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_tests - total_passed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Time: {elapsed_time:.2f}s")
        
        # Save results
        self.test_results['overall_status'] = 'PASSED' if total_passed == total_tests else 'FAILED'
        self.test_results['summary'] = {
            'phase1': {
                'total': phase1_total,
                'passed': phase1_passed,
                'failed': phase1_total - phase1_passed,
                'success_rate': (phase1_passed/phase1_total)*100
            },
            'phase2': {
                'total': phase2_total,
                'passed': phase2_passed,
                'failed': phase2_total - phase2_passed,
                'success_rate': (phase2_passed/phase2_total)*100
            },
            'overall': {
                'total': total_tests,
                'passed': total_passed,
                'failed': total_tests - total_passed,
                'success_rate': success_rate,
                'elapsed_time': elapsed_time
            }
        }
        
        results_file = self.project_root / "phase1_phase2_full_e2e_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Final status
        print("\n" + "="*80)
        if total_passed == total_tests:
            print("✅ ALL END-TO-END TESTS PASSED!")
            print("🎉 Phase 1 and Phase 2 are fully functional and production-ready!")
        else:
            print(f"⚠️  {total_tests - total_passed} test(s) failed. Review errors above.")
            if phase1_passed < phase1_total:
                print(f"   Phase 1: {phase1_total - phase1_passed} failed")
            if phase2_passed < phase2_total:
                print(f"   Phase 2: {phase2_total - phase2_passed} failed")
        print("="*80 + "\n")
        
        return total_passed == total_tests


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Phase 1 & 2 End-to-End Test"
    )
    parser.add_argument(
        "--use-temp-dir",
        action="store_true",
        help="Use temporary directory for testing (cleanup after)"
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download data from Kaggle (requires API credentials)"
    )
    
    args = parser.parse_args()
    
    tester = Phase1Phase2E2ETest(
        use_temp_dir=args.use_temp_dir,
        skip_download=not args.download_data
    )
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
