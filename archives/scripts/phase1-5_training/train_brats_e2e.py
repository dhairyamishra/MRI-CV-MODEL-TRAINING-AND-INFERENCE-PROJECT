#!/usr/bin/env python3
"""
BraTS End-to-End Training Pipeline
===================================

Complete pipeline using ONLY BraTS dataset for:
1. Classification (derived from segmentation masks)
2. Grad-CAM visualization
3. Segmentation
4. Comparison and analysis

Usage:
    python scripts/train_brats_e2e.py                    # Full pipeline
    python scripts/train_brats_e2e.py --skip-cls         # Skip classification
    python scripts/train_brats_e2e.py --skip-seg         # Skip segmentation
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(text):
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*len(text)}{Colors.ENDC}")


def print_success(text):
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print_section(description)
    print_info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, cwd=str(project_root))
        elapsed = time.time() - start_time
        print_success(f"{description} completed in {elapsed/60:.1f} minutes")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print_error(f"{description} failed after {elapsed/60:.1f} minutes")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="BraTS End-to-End Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (classification + segmentation)
  python scripts/train_brats_e2e.py
  
  # Skip classification, only segmentation
  python scripts/train_brats_e2e.py --skip-cls
  
  # Skip segmentation, only classification + Grad-CAM
  python scripts/train_brats_e2e.py --skip-seg
  
  # Test dataset only
  python scripts/train_brats_e2e.py --test-dataset-only

Pipeline Steps:
  1. Test BraTS classification dataset
  2. Train classification model on BraTS
  3. Evaluate classification model
  4. Generate Grad-CAM on BraTS images
  5. Train segmentation model on BraTS
  6. Evaluate segmentation model
  7. Compare classification vs segmentation
        """
    )
    
    parser.add_argument(
        '--skip-cls',
        action='store_true',
        help='Skip classification training'
    )
    
    parser.add_argument(
        '--skip-seg',
        action='store_true',
        help='Skip segmentation training'
    )
    
    parser.add_argument(
        '--test-dataset-only',
        action='store_true',
        help='Only test the dataset, do not train'
    )
    
    args = parser.parse_args()
    
    print_header("BraTS End-to-End Training Pipeline")
    print_info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'start_time': datetime.now().isoformat(),
        'steps': {},
    }
    
    pipeline_start = time.time()
    
    # Step 1: Test BraTS classification dataset
    print_header("Step 1: Test BraTS Classification Dataset")
    cmd = ['python', 'src/data/brats_classification_dataset.py']
    success, elapsed = run_command(cmd, "Dataset testing")
    results['steps']['dataset_test'] = {'success': success, 'time': elapsed}
    
    if not success:
        print_error("Dataset test failed. Exiting.")
        return 1
    
    if args.test_dataset_only:
        print_success("Dataset test complete. Exiting (--test-dataset-only)")
        return 0
    
    # Step 2: Train classification model
    if not args.skip_cls:
        print_header("Step 2: Train BraTS Classification Model")
        
        # First, we need to modify the training script to use BraTS dataset
        print_warning("Note: Using modified training that loads BraTS classification dataset")
        
        cmd = ['python', 'scripts/train_brats_classifier.py', '--config', 'configs/config_cls_brats.yaml']
        success, elapsed = run_command(cmd, "Classification training")
        results['steps']['classification_training'] = {'success': success, 'time': elapsed}
        
        if not success:
            print_warning("Classification training failed, continuing...")
        
        # Step 3: Evaluate classification
        if success:
            print_header("Step 3: Evaluate BraTS Classification")
            cmd = ['python', 'scripts/evaluate_brats_classifier.py']
            success, elapsed = run_command(cmd, "Classification evaluation")
            results['steps']['classification_eval'] = {'success': success, 'time': elapsed}
        
        # Step 4: Generate Grad-CAM
        if success:
            print_header("Step 4: Generate Grad-CAM on BraTS")
            cmd = ['python', 'scripts/generate_brats_gradcam.py']
            success, elapsed = run_command(cmd, "Grad-CAM generation")
            results['steps']['gradcam'] = {'success': success, 'time': elapsed}
    
    # Step 5: Train segmentation model
    if not args.skip_seg:
        print_header("Step 5: Train BraTS Segmentation Model")
        cmd = ['python', 'scripts/train_segmentation.py', '--config', 'configs/seg2d_quick_test.yaml']
        success, elapsed = run_command(cmd, "Segmentation training")
        results['steps']['segmentation_training'] = {'success': success, 'time': elapsed}
        
        # Step 6: Evaluate segmentation
        if success:
            print_header("Step 6: Evaluate BraTS Segmentation")
            cmd = [
                'python', 'scripts/evaluate_segmentation.py',
                '--checkpoint', 'checkpoints/seg_test/best_model.pth',
                '--split', 'test'
            ]
            success, elapsed = run_command(cmd, "Segmentation evaluation")
            results['steps']['segmentation_eval'] = {'success': success, 'time': elapsed}
    
    # Summary
    total_elapsed = time.time() - pipeline_start
    results['end_time'] = datetime.now().isoformat()
    results['total_time'] = total_elapsed
    
    print_header("Pipeline Summary")
    
    print_section("Completed Steps")
    for step_name, step_data in results['steps'].items():
        status = "[OK]" if step_data['success'] else "✗"
        color = Colors.OKGREEN if step_data['success'] else Colors.FAIL
        print(f"{color}{status} {step_name}: {step_data['time']/60:.1f} minutes{Colors.ENDC}")
    
    print_section("Total Time")
    print_info(f"Pipeline completed in {total_elapsed/60:.1f} minutes")
    
    print_section("Next Steps")
    if not args.skip_cls:
        print("• View Grad-CAM: explorer assets\\grad_cam_brats")
        print("• View classification results: explorer results\\classification_brats")
    if not args.skip_seg:
        print("• View segmentation results: explorer outputs\\seg\\evaluation_test")
    print("• Run demo: python scripts/run_demo.py")
    print("• Compare results: python scripts/compare_cls_vs_seg.py")
    
    print_header("BraTS E2E Pipeline Complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
