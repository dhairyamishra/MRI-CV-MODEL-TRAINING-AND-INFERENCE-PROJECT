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

# ANSI color codes for terminal output
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
    
    def _run_command(self, cmd: List[str], step_name: str, timeout: Optional[int] = None) -> bool:
        """
        Run a command and track its execution.
        
        Args:
            cmd: Command to run as list of strings
            step_name: Name of the step for logging
            timeout: Optional timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        step_start = time.time()
        self._print_info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                check=True,
                capture_output=False,
                text=True,
                timeout=timeout
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
        kaggle_dir = self.project_root / "data" / "raw" / "kaggle"
        
        if brats_dir.exists() and kaggle_dir.exists():
            self._print_warning("Data directories already exist. Skipping download.")
            user_input = input("Do you want to re-download? (y/N): ").strip().lower()
            if user_input != 'y':
                return True
        
        # Download BraTS dataset
        self._print_info("Downloading BraTS 2020 dataset (~15GB, may take 10-30 minutes)...")
        if not self._run_command(
            ["python", "scripts/data/collection/download_brats_data.py", "--version", "2020"],
            "download_brats",
            timeout=3600  # 1 hour timeout
        ):
            return False
        
        # Download Kaggle dataset
        self._print_info("Downloading Kaggle brain MRI dataset (~500MB, may take 2-5 minutes)...")
        if not self._run_command(
            ["python", "scripts/data/collection/download_kaggle_data.py"],
            "download_kaggle",
            timeout=600  # 10 minutes timeout
        ):
            return False
        
        return True
    
    def run_data_preprocessing(self) -> bool:
        """Step 2: Preprocess datasets."""
        self._print_step(2, self._get_total_steps(), "Data Preprocessing")
        
        # Determine number of patients based on training mode
        if self.args.training_mode == "quick":
            num_patients = 10
            self._print_info(f"Quick mode: Processing {num_patients} patients")
        elif self.args.training_mode == "baseline":
            num_patients = 100
            self._print_info(f"Baseline mode: Processing {num_patients} patients")
        else:  # production
            num_patients = None  # Process all patients
            self._print_info("Production mode: Processing ALL patients (988)")
        
        # Build preprocessing command
        cmd = ["python", "scripts/data/preprocessing/preprocess_all_brats.py"]
        
        # Add preprocessing options
        cmd.extend([
            "--input", "data/raw/brats2020",  # Specify input directory
            "--modality", "flair",
            "--normalization", "zscore",  # Correct argument name
            "--min-tumor-pixels", "100"
        ])
        
        if num_patients:
            cmd.extend(["--num-patients", str(num_patients)])  # Correct argument name
        
        self._print_info(f"Preprocessing BraTS data (3D→2D conversion, normalization)...")
        timeout = 300 if self.args.training_mode == "quick" else 7200  # 5 min or 2 hours
        
        if not self._run_command(cmd, "preprocess_brats", timeout=timeout):
            return False
        
        return True
    
    def run_data_splitting(self) -> bool:
        """Step 3: Split datasets into train/val/test."""
        self._print_step(3, self._get_total_steps(), "Data Splitting")
        
        # BraTS data is already split by preprocess_all_brats.py
        self._print_info("BraTS data already split by preprocessing step")
        self._print_success("BraTS splits: train/val/test (70/15/15)")
        
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
        
        return True
    
    def run_multitask_training(self) -> bool:
        """Step 4: Multi-task training (3 stages)."""
        self._print_step(4, self._get_total_steps(), "Multi-Task Training (3 Stages)")
        
        # Determine config files based on training mode
        if self.args.training_mode == "quick":
            seg_config = "configs/multitask_seg_warmup_quick_test.yaml"
            cls_config = "configs/multitask_cls_head_quick_test.yaml"
            joint_config = "configs/multitask_joint_quick_test.yaml"
            timeout_seg = 600  # 10 minutes
            timeout_cls = 300  # 5 minutes
            timeout_joint = 600  # 10 minutes
        elif self.args.training_mode == "baseline":
            seg_config = "configs/multitask_seg_warmup.yaml"
            cls_config = "configs/multitask_cls_head_quick_test.yaml"
            joint_config = "configs/multitask_joint_quick_test.yaml"
            timeout_seg = 7200  # 2 hours
            timeout_cls = 3600  # 1 hour
            timeout_joint = 7200  # 2 hours
        else:  # production
            seg_config = "configs/multitask_seg_warmup_production.yaml"
            cls_config = "configs/multitask_cls_head_production.yaml"
            joint_config = "configs/multitask_joint_production.yaml"
            timeout_seg = 21600  # 6 hours
            timeout_cls = 10800  # 3 hours
            timeout_joint = 21600  # 6 hours
        
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
        seg_checkpoint = self.project_root / "checkpoints" / "multitask_seg_warmup" / "best_model.pth"
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
        cls_checkpoint = self.project_root / "checkpoints" / "multitask_cls_head" / "best_model.pth"
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
        joint_checkpoint = self.project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
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
            "seg_out_channels": 1,
            "cls_num_classes": config['model']['num_classes']
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
        model_checkpoint = self.project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
        if not self._check_file_exists(model_checkpoint, "Trained model checkpoint"):
            self._print_error("Cannot launch demo without trained model!")
            return False
        
        self._print_info("Launching multi-task demo application using PM2...")
        self._print_info("")
        self._print_info("PM2 provides robust process management with:")
        self._print_info("  Automatic restart on failure")
        self._print_info("  Centralized logging")
        self._print_info("  Easy monitoring and control")
        self._print_info("")
        
        # Launch using PM2 script
        if not self._run_command(
            ["python", "scripts/demo/run_demo_pm2.py"],
            "launch_demo_pm2",
            timeout=120  # 2 minutes for startup
        ):
            self._print_warning("PM2 launcher failed or PM2 is not installed.")
            self._print_info("")
            self._print_info("Alternative: Run the demo manually in separate terminals:")
            self._print_info("")
            self._print_info("  Terminal 1 (Backend):")
            self._print_info("    python app/backend/main_v2.py")
            self._print_info("")
            self._print_info("  Terminal 2 (Frontend - New Modular Version):")
            self._print_info("    streamlit run app/frontend/app.py --server.port 8501")
            self._print_info("")
            self._print_info("  Terminal 2 (Frontend - Legacy Version):")
            self._print_info("    streamlit run app/frontend/app_v2.py --server.port 8501")
            self._print_info("")
            self._print_info("Or install PM2 and try again:")
            self._print_info("    npm install -g pm2")
            self._print_info("    python scripts/demo/run_demo_pm2.py")
            self._print_info("")
            return False
        
        self._print_success("Demo launched successfully with PM2!")
        self._print_info("")
        self._print_info("To manage the demo:")
        self._print_info("  pm2 status              - View process status")
        self._print_info("  pm2 logs                - View all logs")
        self._print_info("  pm2 monit               - Monitor processes")
        self._print_info("  pm2 stop all            - Stop the demo")
        self._print_info("  pm2 delete all          - Stop and remove processes")
        self._print_info("")
        
        return True
    
    def _get_total_steps(self) -> int:
        """Get total number of steps based on mode."""
        if self.args.mode == "full":
            return 6
        elif self.args.mode == "data-only":
            return 3
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
