#!/usr/bin/env python3
"""
SliceWise Production Training Orchestration Script
===================================================

This script orchestrates the complete training pipeline:
1. Data download (Kaggle + BraTS)
2. Data preprocessing
3. Model training (Classification and/or Segmentation)
4. Comprehensive evaluation
5. Visualization generation
6. Model calibration

Usage:
    # Train classification model
    python scripts/train_production.py --task classification --epochs 100
    
    # Train segmentation model
    python scripts/train_production.py --task segmentation --epochs 100
    
    # Train both models
    python scripts/train_production.py --task both --epochs 100
    
    # Resume training from checkpoint
    python scripts/train_production.py --task classification --resume checkpoints/cls_production/best_model.pth
    
    # Quick test run (10 epochs)
    python scripts/train_production.py --task classification --epochs 10 --quick-test
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import time
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ProductionTrainer:
    """Orchestrates the complete production training pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.project_root = project_root
        self.start_time = datetime.now()
        self.results = {
            "start_time": self.start_time.isoformat(),
            "task": args.task,
            "epochs": args.epochs,
            "steps_completed": [],
            "errors": []
        }
        
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = "🚀" if level == "INFO" else "⚠️" if level == "WARN" else "❌"
        print(f"[{timestamp}] {prefix} {message}")
        
    def run_command(self, cmd, description, critical=True):
        """Run a command and handle errors."""
        self.log(f"Starting: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=False,
                text=True
            )
            self.log(f"✓ Completed: {description}")
            self.results["steps_completed"].append(description)
            return True
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed: {description} (exit code {e.returncode})"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            if critical:
                raise
            return False
        except Exception as e:
            error_msg = f"Error in {description}: {str(e)}"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            if critical:
                raise
            return False
    
    def check_data_exists(self, task):
        """Check if required data exists."""
        if task in ["classification", "both"]:
            cls_data = self.project_root / "data" / "processed" / "kaggle"
            if not cls_data.exists() or not any(cls_data.iterdir()):
                return False, "classification"
        
        if task in ["segmentation", "both"]:
            seg_data = self.project_root / "data" / "processed" / "brats2d"
            if not seg_data.exists() or not any(seg_data.iterdir()):
                return False, "segmentation"
        
        return True, None
    
    def download_and_preprocess_data(self):
        """Download and preprocess all required data."""
        self.log("=" * 80)
        self.log("STEP 1: DATA ACQUISITION & PREPROCESSING")
        self.log("=" * 80)
        
        # Check what data we need
        data_exists, missing_task = self.check_data_exists(self.args.task)
        
        if data_exists and not self.args.force_download:
            self.log("Data already exists. Skipping download. Use --force-download to re-download.")
            return True
        
        # Download Kaggle data for classification
        if self.args.task in ["classification", "both"]:
            self.log("Downloading Kaggle brain MRI dataset...")
            self.run_command(
                [sys.executable, "scripts/download_kaggle_data.py"],
                "Download Kaggle dataset",
                critical=True
            )
            
            self.log("Preprocessing Kaggle data...")
            self.run_command(
                [sys.executable, "scripts/preprocess_kaggle.py"],
                "Preprocess Kaggle dataset",
                critical=True
            )
        
        # Download BraTS data for segmentation
        if self.args.task in ["segmentation", "both"]:
            self.log("Downloading BraTS dataset...")
            self.run_command(
                [sys.executable, "scripts/download_brats_data.py"],
                "Download BraTS dataset",
                critical=True
            )
            
            self.log("Preprocessing BraTS data to 2D slices...")
            self.run_command(
                [sys.executable, "scripts/preprocess_all_brats.py"],
                "Preprocess BraTS dataset",
                critical=True
            )
        
        return True
    
    def train_classification(self):
        """Train classification model."""
        self.log("=" * 80)
        self.log("STEP 2: CLASSIFICATION MODEL TRAINING")
        self.log("=" * 80)
        
        # Select config
        if self.args.quick_test:
            config = "configs/config_cls.yaml"
            self.log("Using quick test config (50 epochs)")
        else:
            config = "configs/config_cls_production.yaml"
            self.log(f"Using production config ({self.args.epochs} epochs)")
        
        # Modify config if custom epochs specified
        if self.args.epochs != 100:
            self.modify_config_epochs(config, self.args.epochs)
        
        # Build training command
        cmd = [sys.executable, "scripts/train_classifier.py", "--config", config]
        
        if self.args.resume:
            self.log(f"Resuming from checkpoint: {self.args.resume}")
            # Note: Need to modify train_cls.py to support --resume flag
        
        self.run_command(
            cmd,
            f"Train classification model ({self.args.epochs} epochs)",
            critical=True
        )
        
        return True
    
    def train_segmentation(self):
        """Train segmentation model."""
        self.log("=" * 80)
        self.log("STEP 2: SEGMENTATION MODEL TRAINING")
        self.log("=" * 80)
        
        # Select config
        if self.args.quick_test:
            config = "configs/seg2d_baseline.yaml"
            self.log("Using quick test config (10 epochs)")
        else:
            config = "configs/seg2d_production.yaml"
            self.log(f"Using production config ({self.args.epochs} epochs)")
        
        # Modify config if custom epochs specified
        if self.args.epochs != 100:
            self.modify_config_epochs(config, self.args.epochs)
        
        # Build training command
        cmd = [sys.executable, "scripts/train_segmentation.py", "--config", config]
        
        self.run_command(
            cmd,
            f"Train segmentation model ({self.args.epochs} epochs)",
            critical=True
        )
        
        return True
    
    def evaluate_models(self):
        """Run comprehensive evaluation."""
        self.log("=" * 80)
        self.log("STEP 3: MODEL EVALUATION")
        self.log("=" * 80)
        
        if self.args.task in ["classification", "both"]:
            self.log("Evaluating classification model...")
            
            # Find best checkpoint
            checkpoint_dir = self.project_root / "checkpoints" / "cls_production"
            best_checkpoint = checkpoint_dir / "best_model.pth"
            
            if best_checkpoint.exists():
                self.run_command(
                    [
                        sys.executable,
                        "scripts/evaluate_classifier.py",
                        "--checkpoint", str(best_checkpoint),
                        "--output-dir", "outputs/classification_production/evaluation"
                    ],
                    "Evaluate classification model",
                    critical=False
                )
            else:
                self.log("No classification checkpoint found, skipping evaluation", "WARN")
        
        if self.args.task in ["segmentation", "both"]:
            self.log("Evaluating segmentation model...")
            
            # Find best checkpoint
            checkpoint_dir = self.project_root / "checkpoints" / "seg_production"
            best_checkpoint = checkpoint_dir / "best_model.pth"
            
            if best_checkpoint.exists():
                self.run_command(
                    [
                        sys.executable,
                        "scripts/evaluate_segmentation.py",
                        "--checkpoint", str(best_checkpoint),
                        "--output-dir", "outputs/seg_production/evaluation"
                    ],
                    "Evaluate segmentation model",
                    critical=False
                )
            else:
                self.log("No segmentation checkpoint found, skipping evaluation", "WARN")
        
        return True
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        self.log("=" * 80)
        self.log("STEP 4: VISUALIZATION GENERATION")
        self.log("=" * 80)
        
        if self.args.task in ["classification", "both"]:
            self.log("Generating Grad-CAM visualizations...")
            self.run_command(
                [
                    sys.executable,
                    "scripts/generate_gradcam.py",
                    "--num-samples", "50",
                    "--output-dir", "visualizations/classification_production/gradcam"
                ],
                "Generate Grad-CAM visualizations",
                critical=False
            )
        
        if self.args.task in ["segmentation", "both"]:
            self.log("Generating segmentation visualizations...")
            self.run_command(
                [
                    sys.executable,
                    "scripts/visualize_segmentation_results.py",
                    "--num-samples", "50",
                    "--output-dir", "visualizations/seg_production/predictions"
                ],
                "Generate segmentation visualizations",
                critical=False
            )
        
        return True
    
    def calibrate_models(self):
        """Calibrate model predictions."""
        self.log("=" * 80)
        self.log("STEP 5: MODEL CALIBRATION")
        self.log("=" * 80)
        
        if self.args.task in ["classification", "both"]:
            self.log("Calibrating classification model...")
            self.run_command(
                [sys.executable, "scripts/calibrate_classifier.py"],
                "Calibrate classification model",
                critical=False
            )
        
        return True
    
    def modify_config_epochs(self, config_path, epochs):
        """Modify config file to set custom number of epochs."""
        config_file = self.project_root / config_path
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'training' in config:
                config['training']['epochs'] = epochs
                self.log(f"Modified {config_path} to use {epochs} epochs")
            
            # Save to temporary config
            temp_config = config_file.parent / f"temp_{config_file.name}"
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
            
            return str(temp_config)
        except Exception as e:
            self.log(f"Could not modify config: {e}", "WARN")
            return config_path
    
    def save_results(self):
        """Save training results summary."""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["duration_seconds"] = (datetime.now() - self.start_time).total_seconds()
        self.results["duration_human"] = str(datetime.now() - self.start_time)
        
        # Save results
        results_file = self.project_root / "outputs" / f"training_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Results saved to: {results_file}")
        
    def print_summary(self):
        """Print training summary."""
        duration = datetime.now() - self.start_time
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Task: {self.args.task}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Duration: {duration}")
        print(f"Steps completed: {len(self.results['steps_completed'])}")
        
        if self.results['errors']:
            print(f"\n⚠️  Errors encountered: {len(self.results['errors'])}")
            for error in self.results['errors']:
                print(f"  - {error}")
        else:
            print("\n✓ All steps completed successfully!")
        
        print("\nNext steps:")
        print("  1. Check W&B dashboard for training curves")
        print("  2. View visualizations in visualizations/ directory")
        print("  3. Test models using the demo app: python scripts/run_demo.py")
        print("  4. Run calibration: python scripts/calibrate_classifier.py")
        print("=" * 80 + "\n")
    
    def run(self):
        """Run the complete training pipeline."""
        try:
            self.log("=" * 80)
            self.log("SliceWise Production Training Pipeline")
            self.log("=" * 80)
            self.log(f"Task: {self.args.task}")
            self.log(f"Epochs: {self.args.epochs}")
            self.log(f"Quick test: {self.args.quick_test}")
            self.log(f"Force download: {self.args.force_download}")
            self.log("=" * 80 + "\n")
            
            # Step 1: Data acquisition
            if not self.args.skip_data:
                self.download_and_preprocess_data()
            else:
                self.log("Skipping data download (--skip-data)")
            
            # Step 2: Training
            if self.args.task == "classification":
                self.train_classification()
            elif self.args.task == "segmentation":
                self.train_segmentation()
            elif self.args.task == "both":
                self.train_classification()
                self.train_segmentation()
            
            # Step 3: Evaluation
            if not self.args.skip_eval:
                self.evaluate_models()
            else:
                self.log("Skipping evaluation (--skip-eval)")
            
            # Step 4: Visualizations
            if not self.args.skip_viz:
                self.generate_visualizations()
            else:
                self.log("Skipping visualizations (--skip-viz)")
            
            # Step 5: Calibration
            if not self.args.skip_calibration:
                self.calibrate_models()
            else:
                self.log("Skipping calibration (--skip-calibration)")
            
            # Save results and print summary
            self.save_results()
            self.print_summary()
            
            return True
            
        except KeyboardInterrupt:
            self.log("\n\nTraining interrupted by user", "WARN")
            self.save_results()
            return False
        except Exception as e:
            self.log(f"\n\nFatal error: {str(e)}", "ERROR")
            self.save_results()
            raise


def main():
    parser = argparse.ArgumentParser(
        description="SliceWise Production Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation", "both"],
        default="classification",
        help="Which model(s) to train"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test run with fewer epochs (10-50)"
    )
    
    # Data options
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data download and preprocessing"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of data even if it exists"
    )
    
    # Pipeline options
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation step"
    )
    
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip model calibration"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    # Quick test mode adjustments
    if args.quick_test:
        if args.epochs == 100:  # Only adjust if using default
            args.epochs = 10 if args.task == "segmentation" else 20
    
    # Run training pipeline
    trainer = ProductionTrainer(args)
    success = trainer.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
