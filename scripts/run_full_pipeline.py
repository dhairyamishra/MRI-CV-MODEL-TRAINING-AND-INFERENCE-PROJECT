"""
Full End-to-End Multi-Task Pipeline Controller

This script orchestrates the complete pipeline for training and deploying
the multi-task brain tumor detection model:

1. Data Download (BraTS + Kaggle datasets)
2. Data Preprocessing (3D→2D conversion, normalization)
3. Data Splitting (patient-level train/val/test)
4. Multi-Task Training (3-stage: seg warmup → cls head → joint fine-tuning)
5. Comprehensive Evaluation (metrics, Grad-CAM, phase comparison)
6. Demo Application Launch (FastAPI + Streamlit)

Usage:
    # Full pipeline with production training (100+ epochs)
    python scripts/run_full_pipeline.py --mode full --training-mode production
    
    # Quick test pipeline (5 epochs, 10 patients)
    python scripts/run_full_pipeline.py --mode full --training-mode quick
    
    # Skip data download (if already downloaded)
    python scripts/run_full_pipeline.py --mode full --skip-download
    
    # Only training and evaluation (skip data preparation)
    python scripts/run_full_pipeline.py --mode train-eval
    
    # Only demo application (requires trained model)
    python scripts/run_full_pipeline.py --mode demo

Author: SliceWise Team
Date: December 7, 2025
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

# ============================================================================
# Configuration Constants
# ============================================================================

# Config file paths - Using new hierarchical config system
# Configs are generated from base/stages/modes using scripts/utils/merge_configs.py
CONFIG_DIR = "configs/final"

# Quick test configs (3 epochs, minimal augmentation, ~30 minutes)
QUICK_SEG_CONFIG = f"{CONFIG_DIR}/stage1_quick.yaml"
QUICK_CLS_CONFIG = f"{CONFIG_DIR}/stage2_quick.yaml"
QUICK_JOINT_CONFIG = f"{CONFIG_DIR}/stage3_quick.yaml"

# Baseline configs (50 epochs, moderate augmentation, ~2-4 hours)
BASELINE_SEG_CONFIG = f"{CONFIG_DIR}/stage1_baseline.yaml"
BASELINE_CLS_CONFIG = f"{CONFIG_DIR}/stage2_baseline.yaml"
BASELINE_JOINT_CONFIG = f"{CONFIG_DIR}/stage3_baseline.yaml"

# Production configs (100 epochs, aggressive augmentation, ~8-12 hours)
PRODUCTION_SEG_CONFIG = f"{CONFIG_DIR}/stage1_production.yaml"
PRODUCTION_CLS_CONFIG = f"{CONFIG_DIR}/stage2_production.yaml"
PRODUCTION_JOINT_CONFIG = f"{CONFIG_DIR}/stage3_production.yaml"

# Timeout values (in seconds)
QUICK_TIMEOUT_SEG = 600      # 10 minutes
QUICK_TIMEOUT_CLS = 300      # 5 minutes
QUICK_TIMEOUT_JOINT = 600    # 10 minutes

BASELINE_TIMEOUT_SEG = 7200   # 2 hours
BASELINE_TIMEOUT_CLS = 3600   # 1 hour
BASELINE_TIMEOUT_JOINT = 7200 # 2 hours

PRODUCTION_TIMEOUT_SEG = 21600   # 6 hours
PRODUCTION_TIMEOUT_CLS = 10800   # 3 hours
PRODUCTION_TIMEOUT_JOINT = 21600 # 6 hours

# Checkpoint paths
CHECKPOINT_DIR = "checkpoints"
MULTITASK_CHECKPOINT = f"{CHECKPOINT_DIR}/multitask_joint/best_model.pth"
SEG_WARMUP_CHECKPOINT = f"{CHECKPOINT_DIR}/multitask_seg_warmup/best_model.pth"
CLS_HEAD_CHECKPOINT = f"{CHECKPOINT_DIR}/multitask_cls_head/best_model.pth"

# Demo URLs
FRONTEND_URL = "http://localhost:8501"
BACKEND_URL = "http://localhost:8000"
API_DOCS_URL = f"{BACKEND_URL}/docs"

# ============================================================================
# ANSI color codes for terminal output
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PipelineController:
    """Controller for managing the full end-to-end pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent.parent
        self.start_time = datetime.now()
        self.results = {
            "start_time": self.start_time.isoformat(),
            "mode": args.mode,
            "training_mode": args.training_mode,
            "steps": {}
        }
        
        # Validate project structure
        self._validate_project_structure()
    
    def _validate_project_structure(self):
        """Validate that required directories exist."""
        required_dirs = [
            "scripts/data/collection",
            "scripts/data/preprocessing",
            "scripts/data/splitting",
            "scripts/training/multitask",
            "scripts/evaluation/multitask",
            "scripts/demo",
            "configs",
            "src"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                self._print_error(f"Required directory not found: {dir_path}")
                sys.exit(1)
    
    def _print_header(self, message: str):
        """Print a formatted header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    def _print_step(self, step_num: int, total_steps: int, message: str):
        """Print a formatted step message."""
        print(f"\n{Colors.OKCYAN}{Colors.BOLD}[Step {step_num}/{total_steps}] {message}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'-'*80}{Colors.ENDC}")
    
    def _print_success(self, message: str):
        """Print a success message."""
        print(f"{Colors.OKGREEN}[OK] {message}{Colors.ENDC}")
    
    def _print_warning(self, message: str):
        """Print a warning message."""
        print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")
    
    def _print_error(self, message: str):
        """Print an error message."""
        print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")
    
    def _print_info(self, message: str):
        """Print an info message."""
        print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")
    
    def _run_command(self, cmd: List[str], step_name: str, timeout: Optional[int] = None, shell: bool = False) -> bool:
        """
        Run a command and track its execution.
        
        Args:
            cmd: Command to run as list of strings
            step_name: Name of the step for logging
            timeout: Optional timeout in seconds
            shell: Whether to run command through shell (needed for npm/pm2 on Windows)
            
        Returns:
            True if successful, False otherwise
        """
        step_start = time.time()
        self._print_info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd if not shell else ' '.join(cmd),
                cwd=str(self.project_root),
                check=True,
                capture_output=False,
                text=True,
                timeout=timeout,
                shell=shell
            )
            
            duration = time.time() - step_start
            self.results["steps"][step_name] = {
                "status": "success",
                "duration_seconds": round(duration, 2),
                "command": " ".join(cmd)
            }
            
            self._print_success(f"{step_name} completed in {duration:.1f}s")
            return True
            
        except subprocess.TimeoutExpired:
            self._print_error(f"{step_name} timed out after {timeout}s")
            self.results["steps"][step_name] = {
                "status": "timeout",
                "timeout_seconds": timeout,
                "command": " ".join(cmd)
            }
            return False
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - step_start
            self._print_error(f"{step_name} failed after {duration:.1f}s")
            self.results["steps"][step_name] = {
                "status": "failed",
                "duration_seconds": round(duration, 2),
                "command": " ".join(cmd),
                "error": str(e)
            }
            return False
    
    def _check_file_exists(self, file_path: Path, description: str) -> bool:
        """Check if a file exists and print status."""
        if file_path.exists():
            self._print_success(f"{description} found: {file_path}")
            return True
        else:
            self._print_warning(f"{description} not found: {file_path}")
            return False
    
    def run_data_download(self) -> bool:
        """Step 1: Download datasets."""
        self._print_step(1, self._get_total_steps(), "Data Download")
        
        if self.args.skip_download:
            self._print_warning("Skipping data download (--skip-download flag)")
            return True
        
        # Check if data already exists
        brats_dir = self.project_root / "data" / "raw" / "brats2020"
        kaggle_dir = self.project_root / "data" / "raw" / "kaggle_brain_mri"
        
        # Check BraTS dataset
        download_brats = True
        if brats_dir.exists():
            brats_folders = list(brats_dir.glob("BraTS*"))
            if len(brats_folders) > 0:
                self._print_warning(f"BraTS dataset already exists ({len(brats_folders)} patient folders found)")
                user_input = input("Do you want to re-download BraTS? (y/N): ").strip().lower()
                download_brats = (user_input == 'y')
        
        # Check Kaggle dataset
        download_kaggle = True
        if kaggle_dir.exists():
            yes_dir = kaggle_dir / "yes"
            no_dir = kaggle_dir / "no"
            if yes_dir.exists() and no_dir.exists():
                yes_count = len(list(yes_dir.glob("*.jpg")))
                no_count = len(list(no_dir.glob("*.jpg")))
                total_images = yes_count + no_count
                if total_images > 0:
                    self._print_warning(f"Kaggle MRI dataset already exists ({total_images} images found)")
                    user_input = input("Do you want to re-download Kaggle dataset? (y/N): ").strip().lower()
                    download_kaggle = (user_input == 'y')
        
        # Download BraTS dataset if needed
        if download_brats:
            self._print_info("Downloading BraTS 2020 dataset (~15GB, may take 10-30 minutes)...")
            if not self._run_command(
                ["python", "scripts/data/collection/download_brats_data.py", "--version", "2020"],
                "download_brats",
                timeout=3600  # 1 hour timeout
            ):
                return False
        else:
            self._print_info("Skipping BraTS download (using existing data)")
        
        # Download Kaggle dataset if needed
        if download_kaggle:
            self._print_info("Downloading Kaggle brain MRI dataset (~500MB, may take 2-5 minutes)...")
            if not self._run_command(
                ["python", "scripts/data/collection/download_kaggle_data.py"],
                "download_kaggle",
                timeout=600  # 10 minutes timeout
            ):
                return False
        else:
            self._print_info("Skipping Kaggle download (using existing data)")
        
        return True
    
    def run_data_preprocessing(self) -> bool:
        """Step 2: Preprocess datasets."""
        self._print_step(2, self._get_total_steps(), "Data Preprocessing")
        
        if self.args.skip_preprocessing:
            self._print_warning("Skipping data preprocessing (--skip-preprocessing flag)")
            return True
        
        # Dynamically determine number of patients based on actual dataset size
        brats_raw_dir = self.project_root / "data" / "raw" / "brats2020"
        
        # Count available BraTS patient folders
        total_patients = 0
        if brats_raw_dir.exists():
            patient_folders = list(brats_raw_dir.glob("BraTS*"))
            total_patients = len(patient_folders)
            self._print_info(f"Found {total_patients} BraTS patient folders")
        else:
            self._print_warning(f"BraTS directory not found: {brats_raw_dir}")
            total_patients = 500  # Fallback estimate
        
        # Calculate percentages based on actual dataset size
        if self.args.training_mode == "quick":
            num_patients = max(2, int(total_patients * 0.05))  # 5% with minimum of 2
            self._print_info(f"Quick mode: Processing {num_patients} patients (~5% of {total_patients} for faster loading)")
        elif self.args.training_mode == "baseline":
            num_patients = max(50, int(total_patients * 0.30))  # 30% with minimum of 50
            self._print_info(f"Baseline mode: Processing {num_patients} patients (~30% of {total_patients})")
        else:  # production
            num_patients = None  # Process all patients
            self._print_info(f"Production mode: Processing ALL {total_patients} patients")
        
        # Check for existing preprocessed BraTS data
        brats_processed_dir = self.project_root / "data" / "processed" / "brats2d"
        preprocess_brats = True
        if brats_processed_dir.exists():
            # Count existing processed files to detect stale data
            existing_train = list((brats_processed_dir / "train").glob("*.npz")) if (brats_processed_dir / "train").exists() else []
            if len(existing_train) > 0:
                self._print_warning(f"BraTS preprocessed data already exists ({len(existing_train)} train files)")
                self._print_warning(f"Current mode will process {num_patients} patients, but existing data may have different count")
                user_input = input("Do you want to re-preprocess BraTS? (y/N): ").strip().lower()
                preprocess_brats = (user_input == 'y')
        
        # Check for existing preprocessed Kaggle data
        kaggle_processed_dir = self.project_root / "data" / "processed" / "kaggle"
        preprocess_kaggle = True
        if kaggle_processed_dir.exists():
            existing_train = list((kaggle_processed_dir / "train").glob("*.npz")) if (kaggle_processed_dir / "train").exists() else []
            if len(existing_train) > 0:
                self._print_warning(f"Kaggle preprocessed data already exists ({len(existing_train)} train files)")
                user_input = input("Do you want to re-preprocess Kaggle dataset? (y/N): ").strip().lower()
                preprocess_kaggle = (user_input == 'y')
        
        # Preprocess BraTS data if needed
        if preprocess_brats:
            # Build preprocessing command - use direct script with new no-tumor slice support
            cmd = ["python", "src/data/preprocess_brats_2d.py"]
            
            # Add preprocessing options
            cmd.extend([
                "--input", "data/raw/brats2020",
                "--output", "data/processed/brats2d",
                "--modality", "flair",
                "--normalize", "zscore",
                "--min-tumor-pixels", "100",
                "--save-all-slices",  # NEW: Save tumor + no-tumor slices
                "--no-tumor-sample-rate", "0.3"  # NEW: Keep 30% of no-tumor slices
            ])
            
            if num_patients:
                cmd.extend(["--max-patients", str(num_patients)])
            
            self._print_info(f"Preprocessing BraTS data (3D→2D conversion, normalization)...")
            timeout = 300 if self.args.training_mode == "quick" else 7200  # 5 min or 2 hours
            
            if not self._run_command(cmd, "preprocess_brats", timeout=timeout):
                return False
        else:
            self._print_info("Skipping BraTS preprocessing (using existing data)")
        
        # Preprocess Kaggle data if needed
        if preprocess_kaggle:
            self._print_info("Preprocessing Kaggle data (JPG→NPZ conversion, normalization)...")
            if not self._run_command(
                ["python", "src/data/preprocess_kaggle.py",
                 "--raw-dir", "data/raw/kaggle_brain_mri",
                 "--processed-dir", "data/processed/kaggle",
                 "--target-size", "256", "256"],
                "preprocess_kaggle",
                timeout=300  # 5 minutes should be enough for 245 images
            ):
                return False
        else:
            self._print_info("Skipping Kaggle preprocessing (using existing data)")
        
        return True
    
    def run_data_splitting(self) -> bool:
        """Step 3: Split datasets into train/val/test."""
        self._print_step(3, self._get_total_steps(), "Data Splitting")
        
        # Split BraTS data (patient-level splitting)
        self._print_info("Splitting BraTS data (patient-level, 70/15/15)...")
        if not self._run_command(
            ["python", "src/data/split_brats.py",
             "--input", "data/processed/brats2d",
             "--train-ratio", "0.7",
             "--val-ratio", "0.15",
             "--test-ratio", "0.15",
             "--seed", "42"],
            "split_brats",
            timeout=300
        ):
            return False
        
        # Split Kaggle data
        self._print_info("Splitting Kaggle data (patient-level, 70/15/15)...")
        if not self._run_command(
            ["python", "scripts/data/splitting/split_kaggle_data.py",
             "--train", "0.7",
             "--val", "0.15",
             "--test", "0.15",
             "--seed", "42"],
            "split_kaggle",
            timeout=300
        ):
            return False
        
        # Generate example dataset for testing
        self._print_info("Generating example dataset for demo testing...")
        if not self._run_command(
            ["python", "scripts/data/preprocessing/export_dataset_examples.py",
             "--kaggle-with-tumor", "10",
             "--kaggle-without-tumor", "10", 
             "--brats-with-tumor", "10",
             "--brats-without-tumor", "10"],
            "export_examples",
            timeout=600  # 10 minutes should be enough
        ):
            self._print_warning("Example dataset generation failed, but continuing...")
        
        return True
    
    def run_multitask_training(self) -> bool:
        """Step 4: Multi-task training (3 stages)."""
        self._print_step(4, self._get_total_steps(), "Multi-Task Training (3 Stages)")
        
        # Determine config files based on training mode
        if self.args.training_mode == "quick":
            seg_config = QUICK_SEG_CONFIG
            cls_config = QUICK_CLS_CONFIG
            joint_config = QUICK_JOINT_CONFIG
            timeout_seg = QUICK_TIMEOUT_SEG
            timeout_cls = QUICK_TIMEOUT_CLS
            timeout_joint = QUICK_TIMEOUT_JOINT
        elif self.args.training_mode == "baseline":
            seg_config = BASELINE_SEG_CONFIG
            cls_config = BASELINE_CLS_CONFIG
            joint_config = BASELINE_JOINT_CONFIG
            timeout_seg = BASELINE_TIMEOUT_SEG
            timeout_cls = BASELINE_TIMEOUT_CLS
            timeout_joint = BASELINE_TIMEOUT_JOINT
        else:  # production
            seg_config = PRODUCTION_SEG_CONFIG
            cls_config = PRODUCTION_CLS_CONFIG
            joint_config = PRODUCTION_JOINT_CONFIG
            timeout_seg = PRODUCTION_TIMEOUT_SEG
            timeout_cls = PRODUCTION_TIMEOUT_CLS
            timeout_joint = PRODUCTION_TIMEOUT_JOINT
        
        # Stage 1: Segmentation Warm-up
        self._print_info("Stage 1/3: Segmentation warm-up (encoder + decoder on BraTS)")
        if not self._run_command(
            ["python", "scripts/training/multitask/train_multitask_seg_warmup.py",
             "--config", seg_config],
            "train_stage1_seg_warmup",
            timeout=timeout_seg
        ):
            self._print_error("Stage 1 failed. Cannot proceed to Stage 2.")
            return False
        
        # Check if checkpoint exists
        seg_checkpoint = self.project_root / SEG_WARMUP_CHECKPOINT
        if not self._check_file_exists(seg_checkpoint, "Stage 1 checkpoint"):
            return False
        
        # Stage 2: Classification Head Training
        self._print_info("Stage 2/3: Classification head training (frozen encoder)")
        if not self._run_command(
            ["python", "scripts/training/multitask/train_multitask_cls_head.py",
             "--config", cls_config,
             "--encoder-init", str(seg_checkpoint)],
            "train_stage2_cls_head",
            timeout=timeout_cls
        ):
            self._print_error("Stage 2 failed. Cannot proceed to Stage 3.")
            return False
        
        # Check if checkpoint exists
        cls_checkpoint = self.project_root / CLS_HEAD_CHECKPOINT
        if not self._check_file_exists(cls_checkpoint, "Stage 2 checkpoint"):
            return False
        
        # Stage 3: Joint Fine-tuning
        self._print_info("Stage 3/3: Joint fine-tuning (all parameters unfrozen)")
        if not self._run_command(
            ["python", "scripts/training/multitask/train_multitask_joint.py",
             "--config", joint_config,
             "--init-from", str(cls_checkpoint)],
            "train_stage3_joint",
            timeout=timeout_joint
        ):
            self._print_error("Stage 3 failed.")
            return False
        
        # Check if final checkpoint exists
        joint_checkpoint = self.project_root / MULTITASK_CHECKPOINT
        if not self._check_file_exists(joint_checkpoint, "Stage 3 (final) checkpoint"):
            return False
        
        # Create model_config.json for demo to load the correct architecture
        self._print_info("Creating model_config.json for demo...")
        import json
        import yaml
        
        # Read architecture params from the config file
        with open(joint_config, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = {
            "base_filters": config['model']['base_filters'],
            "depth": config['model']['depth'],
            "in_channels": config['model']['in_channels'],
            "seg_out_channels": config['model']['seg_out_channels'],
            "cls_num_classes": config['model']['cls_num_classes']
        }
        
        config_path = self.project_root / "checkpoints" / "multitask_joint" / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f)
        
        self._print_success(f"Model config saved: {config_path}")
        self._print_success("All 3 training stages completed successfully!")
        return True
    
    def run_evaluation(self) -> bool:
        """Step 5: Comprehensive evaluation."""
        self._print_step(5, self._get_total_steps(), "Comprehensive Evaluation")
        
        # Evaluate multi-task model
        self._print_info("Evaluating multi-task model on test set...")
        if not self._run_command(
            ["python", "scripts/evaluation/multitask/evaluate_multitask.py",
             "--checkpoint", "checkpoints/multitask_joint/best_model.pth",
             "--output", "results/multitask_evaluation.json"],
            "evaluate_multitask",
            timeout=1800  # 30 minutes
        ):
            return False
        
        # Generate Grad-CAM visualizations
        num_samples = 20 if self.args.training_mode == "quick" else 50
        self._print_info(f"Generating Grad-CAM visualizations ({num_samples} samples)...")
        if not self._run_command(
            ["python", "scripts/evaluation/multitask/generate_multitask_gradcam.py",
             "--checkpoint", "checkpoints/multitask_joint/best_model.pth",
             "--num-samples", str(num_samples),
             "--output-dir", "visualizations/multitask_gradcam"],
            "generate_gradcam",
            timeout=1800  # 30 minutes
        ):
            self._print_warning("Grad-CAM generation failed, but continuing...")
        
        # Compare all training phases
        self._print_info("Comparing all 3 training phases...")
        if not self._run_command(
            ["python", "scripts/evaluation/multitask/compare_all_phases.py",
             "--phase-21-checkpoint", "checkpoints/multitask_seg_warmup/best_model.pth",
             "--phase-22-checkpoint", "checkpoints/multitask_cls_head/best_model.pth",
             "--phase-23-checkpoint", "checkpoints/multitask_joint/best_model.pth",
             "--output", "results/phase_comparison.json"],
            "compare_phases",
            timeout=3600  # 1 hour
        ):
            self._print_warning("Phase comparison failed, but continuing...")
        
        return True
    
    def run_demo(self) -> bool:
        """Step 6: Launch demo application."""
        self._print_step(6, self._get_total_steps(), "Launch Demo Application")
        
        # Check if model checkpoint exists
        model_checkpoint = self.project_root / MULTITASK_CHECKPOINT
        if not self._check_file_exists(model_checkpoint, "Trained model checkpoint"):
            self._print_error("Cannot launch demo without trained model!")
            return False
        
        self._print_info("Launching multi-task demo application using PM2...")
        self._print_info("")
        self._print_info("PM2 provides robust process management with:")
        self._print_info("  Automatic restart on failure")
        self._print_info("  Centralized logging in logs/ directory")
        self._print_info("  Background execution (no terminal windows)")
        self._print_info("  Easy monitoring and control")
        self._print_info("")
        
        # Try running PM2 directly (works on Windows)
        if self._run_command(
            ["pm2", "start", "configs/pm2-ecosystem/ecosystem.config.js"],
            "launch_pm2_direct",
            timeout=60,  # 1 minute for startup
            shell=True
        ):
            self._print_success("Demo launched successfully with PM2!")
            self._print_info("")
            self._print_info("🌐 Access the demo:")
            self._print_info(f"  Frontend: {FRONTEND_URL}")
            self._print_info(f"  Backend:  {BACKEND_URL}")
            self._print_info(f"  API Docs: {API_DOCS_URL}")
            self._print_info("")
            self._print_info("📊 Manage with PM2:")
            self._print_info("  pm2 status              - View process status")
            self._print_info("  pm2 logs                - View all logs (live)")
            self._print_info("  pm2 monit               - Interactive monitoring")
            self._print_info("  pm2 stop all            - Stop the demo")
            self._print_info("  pm2 delete all          - Stop and remove processes")
            self._print_info("")
            self._print_info("💡 Tip: Processes run in background. You can close this terminal.")
            self._print_info("")
            return True
        
        # PM2 command failed - show alternatives
        self._print_warning("PM2 command failed or PM2 is not installed.")
        self._print_info("")
        self._print_info("To install PM2:")
        self._print_info("  npm install -g pm2")
        self._print_info("")
        self._print_info("Then run:")
        self._print_info("  pm2 start configs/pm2-ecosystem/ecosystem.config.js")
        self._print_info("")
        self._print_info("Alternative: Run manually in separate terminals:")
        self._print_info("  Terminal 1: python scripts/demo/run_demo_backend.py")
        self._print_info("  Terminal 2: python scripts/demo/run_demo_frontend.py")
        self._print_info("")
        return False
    
    def _get_total_steps(self) -> int:
        """Get total number of steps based on mode."""
        if self.args.mode == "full":
            return 7  # Added export_examples step
        elif self.args.mode == "data-only":
            return 4  # Added export_examples step
        elif self.args.mode == "train-eval":
            return 2
        elif self.args.mode == "demo":
            return 1
        return 0
    
    def run(self):
        """Run the full pipeline based on mode."""
        self._print_header(f"SliceWise Multi-Task Pipeline Controller")
        self._print_info(f"Mode: {self.args.mode}")
        self._print_info(f"Training Mode: {self.args.training_mode}")
        self._print_info(f"Project Root: {self.project_root}")
        
        success = True
        
        if self.args.mode == "full":
            # Full pipeline: data → training → evaluation → demo
            success = (
                self.run_data_download() and
                self.run_data_preprocessing() and
                self.run_data_splitting() and
                self.run_multitask_training() and
                self.run_evaluation() and
                self.run_demo()
            )
        
        elif self.args.mode == "data-only":
            # Only data preparation
            success = (
                self.run_data_download() and
                self.run_data_preprocessing() and
                self.run_data_splitting()
            )
        
        elif self.args.mode == "train-eval":
            # Only training and evaluation
            success = (
                self.run_multitask_training() and
                self.run_evaluation()
            )
        
        elif self.args.mode == "demo":
            # Only demo application
            success = self.run_demo()
        
        # Print final summary
        self._print_summary(success)
        
        # Save results to JSON
        self._save_results(success)
        
        return success
    
    def _print_summary(self, success: bool):
        """Print final summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        self._print_header("Pipeline Summary")
        
        if success:
            self._print_success(f"Pipeline completed successfully in {duration:.1f}s ({duration/60:.1f} minutes)")
        else:
            self._print_error(f"Pipeline failed after {duration:.1f}s ({duration/60:.1f} minutes)")
        
        # Print step results
        print(f"\n{Colors.BOLD}Step Results:{Colors.ENDC}")
        for step_name, step_info in self.results["steps"].items():
            status = step_info["status"]
            duration = step_info.get("duration_seconds", 0)
            
            if status == "success":
                self._print_success(f"{step_name}: {duration:.1f}s")
            elif status == "failed":
                self._print_error(f"{step_name}: Failed after {duration:.1f}s")
            elif status == "timeout":
                self._print_error(f"{step_name}: Timeout")
        
        # Print next steps
        if success:
            print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
            if self.args.mode == "full":
                self._print_info("Pipeline complete! Demo application is running.")
            elif self.args.mode == "data-only":
                self._print_info("Run training: python scripts/run_full_pipeline.py --mode train-eval")
            elif self.args.mode == "train-eval":
                self._print_info("Launch demo: python scripts/run_full_pipeline.py --mode demo")
    
    def _save_results(self, success: bool):
        """Save pipeline results to JSON file."""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["success"] = success
        self.results["total_duration_seconds"] = (datetime.now() - self.start_time).total_seconds()
        
        results_file = self.project_root / "pipeline_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        self._print_info(f"Results saved to: {results_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full End-to-End Multi-Task Pipeline Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with production training (100+ epochs)
  python scripts/run_full_pipeline.py --mode full --training-mode production
  
  # Quick test pipeline (5 epochs, 10 patients)
  python scripts/run_full_pipeline.py --mode full --training-mode quick
  
  # Skip data download (if already downloaded)
  python scripts/run_full_pipeline.py --mode full --skip-download
  
  # Only data preparation
  python scripts/run_full_pipeline.py --mode data-only
  
  # Only training and evaluation
  python scripts/run_full_pipeline.py --mode train-eval
  
  # Only demo application
  python scripts/run_full_pipeline.py --mode demo

Modes:
  full        - Complete pipeline: data → training → evaluation → demo
  data-only   - Only data download, preprocessing, and splitting
  train-eval  - Only training and evaluation (requires prepared data)
  demo        - Only launch demo application (requires trained model)

Training Modes:
  quick       - Quick test (5 epochs, 10 patients, ~30 minutes)
  baseline    - Baseline training (50 epochs, 100 patients, ~2-4 hours)
  production  - Full production (100+ epochs, all 988 patients, ~8-12 hours)
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "data-only", "train-eval", "demo"],
        default="full",
        help="Pipeline mode (default: full)"
    )
    
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["quick", "baseline", "production"],
        default="baseline",
        help="Training mode (default: baseline)"
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download step (use existing data)"
    )
    
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing step (use existing processed data)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create controller and run pipeline
    controller = PipelineController(args)
    success = controller.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
