#!/usr/bin/env python3
"""
SliceWise Training Controller
==============================

Interactive training orchestrator for classification and segmentation models.
Allows easy selection of training type, configuration, and parameter overrides.

Usage:
    python scripts/train_controller.py                    # Interactive mode
    python scripts/train_controller.py --mode cls         # Quick classification
    python scripts/train_controller.py --mode seg         # Quick segmentation
    python scripts/train_controller.py --mode both        # Train both models
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ANSI Colors for beautiful output
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


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(text: str):
    """Print a section header."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*len(text)}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


# Available configurations
CONFIGS = {
    'classification': {
        'quick_test': 'configs/config_cls.yaml',  # Kaggle dataset
        'production': 'configs/config_cls_production.yaml',
    },
    'classification_brats': {
        'quick_test': 'configs/config_cls_brats.yaml',  # BraTS dataset
    },
    'segmentation': {
        'quick_test': 'configs/seg2d_quick_test.yaml',
        'baseline': 'configs/seg2d_baseline.yaml',
        'production': 'configs/seg2d_production.yaml',
    }
}


class TrainingController:
    """Interactive training controller."""
    
    def __init__(self):
        self.project_root = project_root
        self.results = {
            'classification': None,
            'classification_brats': None,
            'segmentation': None,
            'start_time': None,
            'end_time': None,
        }
    
    def run_interactive(self):
        """Run interactive training selection."""
        print_header("SliceWise Training Controller")
        print_info("Interactive training orchestrator for brain tumor detection models")
        
        # Step 1: Select training mode
        mode = self._select_training_mode()
        
        if mode == 'both':
            # Train both models
            self._train_classification_interactive()
            self._train_segmentation_interactive()
        elif mode == 'classification':
            self._train_classification_interactive()
        elif mode == 'classification_brats':
            self._train_classification_brats_interactive()
        elif mode == 'segmentation':
            self._train_segmentation_interactive()
        elif mode == 'brats_e2e':
            self._train_brats_e2e_interactive()
        else:
            print_error("Invalid mode selected")
            return
        
        # Summary
        self._print_summary()
    
    def _select_training_mode(self) -> str:
        """Select training mode interactively."""
        print_section("Step 1: Select Training Mode")
        print("1. Classification only (Kaggle dataset)")
        print("2. Classification only (BraTS dataset)")
        print("3. Segmentation only (BraTS dataset)")
        print("4. Both Kaggle (Classification) + BraTS (Segmentation)")
        print("5. Both BraTS (Classification + Segmentation) - E2E")
        print("6. Exit")
        
        while True:
            choice = input(f"\n{Colors.BOLD}Enter choice [1-6]: {Colors.ENDC}").strip()
            if choice == '1':
                return 'classification'
            elif choice == '2':
                return 'classification_brats'
            elif choice == '3':
                return 'segmentation'
            elif choice == '4':
                return 'both'
            elif choice == '5':
                return 'brats_e2e'
            elif choice == '6':
                print_info("Exiting...")
                sys.exit(0)
            else:
                print_warning("Invalid choice. Please enter 1-6.")
    
    def _select_config(self, model_type: str) -> str:
        """Select configuration file."""
        configs = CONFIGS[model_type]
        
        print_section(f"Select {model_type.capitalize()} Configuration")
        for i, (name, path) in enumerate(configs.items(), 1):
            print(f"{i}. {name:15s} - {path}")
        
        while True:
            choice = input(f"\n{Colors.BOLD}Enter choice [1-{len(configs)}]: {Colors.ENDC}").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(configs):
                    config_name = list(configs.keys())[idx]
                    config_path = configs[config_name]
                    print_success(f"Selected: {config_name} ({config_path})")
                    return config_path
                else:
                    print_warning(f"Invalid choice. Please enter 1-{len(configs)}.")
            except ValueError:
                print_warning("Please enter a number.")
    
    def _get_parameter_overrides(self, model_type: str) -> Dict[str, any]:
        """Get parameter overrides from user."""
        print_section(f"Customize {model_type.capitalize()} Parameters (Optional)")
        print("Press Enter to skip and use config defaults")
        
        overrides = {}
        
        # Common parameters
        params = {
            'epochs': ('Number of epochs', int),
            'batch_size': ('Batch size', int),
            'learning_rate': ('Learning rate', float),
        }
        
        for param, (description, dtype) in params.items():
            value = input(f"{description} [{param}]: ").strip()
            if value:
                try:
                    overrides[param] = dtype(value)
                    print_success(f"Set {param} = {overrides[param]}")
                except ValueError:
                    print_warning(f"Invalid value for {param}, skipping...")
        
        return overrides
    
    def _train_classification_interactive(self):
        """Train classification model interactively."""
        print_header("Classification Training")
        
        # Select config
        config_path = self._select_config('classification')
        
        # Get parameter overrides
        overrides = self._get_parameter_overrides('classification')
        
        # Confirm
        print_section("Training Summary")
        print(f"Model:  Classification (EfficientNet-B0)")
        print(f"Config: {config_path}")
        if overrides:
            print(f"Overrides: {overrides}")
        
        confirm = input(f"\n{Colors.BOLD}Start training? [Y/n]: {Colors.ENDC}").strip().lower()
        if confirm in ['', 'y', 'yes']:
            self._run_classification_training(config_path, overrides)
        else:
            print_warning("Classification training skipped")
    
    def _train_classification_brats_interactive(self):
        """Train classification model interactively."""
        print_header("Classification Training")
        
        # Select config
        config_path = self._select_config('classification_brats')
        
        # Get parameter overrides
        overrides = self._get_parameter_overrides('classification_brats')
        
        # Confirm
        print_section("Training Summary")
        print(f"Model:  Classification (EfficientNet-B0)")
        print(f"Config: {config_path}")
        if overrides:
            print(f"Overrides: {overrides}")
        
        confirm = input(f"\n{Colors.BOLD}Start training? [Y/n]: {Colors.ENDC}").strip().lower()
        if confirm in ['', 'y', 'yes']:
            self._run_classification_brats_training(config_path, overrides)
        else:
            print_warning("Classification training skipped")
    
    def _train_segmentation_interactive(self):
        """Train segmentation model interactively."""
        print_header("Segmentation Training")
        
        # Select config
        config_path = self._select_config('segmentation')
        
        # Get parameter overrides
        overrides = self._get_parameter_overrides('segmentation')
        
        # Confirm
        print_section("Training Summary")
        print(f"Model:  Segmentation (U-Net 2D)")
        print(f"Config: {config_path}")
        if overrides:
            print(f"Overrides: {overrides}")
        
        confirm = input(f"\n{Colors.BOLD}Start training? [Y/n]: {Colors.ENDC}").strip().lower()
        if confirm in ['', 'y', 'yes']:
            self._run_segmentation_training(config_path, overrides)
        else:
            print_warning("Segmentation training skipped")
    
    def _train_brats_e2e_interactive(self):
        """Train BraTS E2E model interactively."""
        print_header("BraTS E2E Training")
        
        # Select config
        cls_config_path = self._select_config('classification_brats')
        seg_config_path = self._select_config('segmentation')
        
        # Get parameter overrides
        cls_overrides = self._get_parameter_overrides('classification_brats')
        seg_overrides = self._get_parameter_overrides('segmentation')
        
        # Confirm
        print_section("Training Summary")
        print(f"Model:  BraTS E2E (Classification + Segmentation)")
        print(f"Classification Config: {cls_config_path}")
        print(f"Segmentation Config: {seg_config_path}")
        if cls_overrides:
            print(f"Classification Overrides: {cls_overrides}")
        if seg_overrides:
            print(f"Segmentation Overrides: {seg_overrides}")
        
        confirm = input(f"\n{Colors.BOLD}Start training? [Y/n]: {Colors.ENDC}").strip().lower()
        if confirm in ['', 'y', 'yes']:
            self._run_brats_e2e_training(cls_config_path, cls_overrides, seg_config_path, seg_overrides)
        else:
            print_warning("BraTS E2E training skipped")
    
    def _run_classification_training(self, config_path: str, overrides: Dict = None):
        """Run classification training."""
        print_section("Starting Classification Training")
        
        start_time = time.time()
        
        cmd = ['python', 'scripts/train_classifier.py', '--config', config_path]
        
        # Note: Parameter overrides would require modifying the config file
        # or adding CLI arguments to train_classifier.py
        if overrides:
            print_warning("Parameter overrides require config file modification")
            print_info("Using config file as-is for now")
        
        try:
            print_info(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, cwd=str(self.project_root))
            
            elapsed = time.time() - start_time
            self.results['classification'] = {
                'status': 'success',
                'elapsed_time': elapsed,
                'config': config_path,
            }
            print_success(f"Classification training completed in {elapsed/60:.1f} minutes")
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            self.results['classification'] = {
                'status': 'failed',
                'elapsed_time': elapsed,
                'error': str(e),
            }
            print_error(f"Classification training failed: {e}")
    
    def _run_classification_brats_training(self, config_path: str, overrides: Dict = None):
        """Run classification training."""
        print_section("Starting Classification Training")
        
        start_time = time.time()
        
        cmd = ['python', 'scripts/train_classifier_brats.py', '--config', config_path]
        
        # Note: Parameter overrides would require modifying the config file
        # or adding CLI arguments to train_classifier.py
        if overrides:
            print_warning("Parameter overrides require config file modification")
            print_info("Using config file as-is for now")
        
        try:
            print_info(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, cwd=str(self.project_root))
            
            elapsed = time.time() - start_time
            self.results['classification_brats'] = {
                'status': 'success',
                'elapsed_time': elapsed,
                'config': config_path,
            }
            print_success(f"Classification training completed in {elapsed/60:.1f} minutes")
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            self.results['classification_brats'] = {
                'status': 'failed',
                'elapsed_time': elapsed,
                'error': str(e),
            }
            print_error(f"Classification training failed: {e}")
    
    def _run_segmentation_training(self, config_path: str, overrides: Dict = None):
        """Run segmentation training."""
        print_section("Starting Segmentation Training")
        
        start_time = time.time()
        
        cmd = ['python', 'scripts/train_segmentation.py', '--config', config_path]
        
        if overrides:
            print_warning("Parameter overrides require config file modification")
            print_info("Using config file as-is for now")
        
        try:
            print_info(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, cwd=str(self.project_root))
            
            elapsed = time.time() - start_time
            self.results['segmentation'] = {
                'status': 'success',
                'elapsed_time': elapsed,
                'config': config_path,
            }
            print_success(f"Segmentation training completed in {elapsed/60:.1f} minutes")
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            self.results['segmentation'] = {
                'status': 'failed',
                'elapsed_time': elapsed,
                'error': str(e),
            }
            print_error(f"Segmentation training failed: {e}")
    
    def _run_brats_e2e_training(self, cls_config_path: str, cls_overrides: Dict, seg_config_path: str, seg_overrides: Dict):
        """Run BraTS E2E training."""
        print_section("Starting BraTS E2E Training")
        
        start_time = time.time()
        
        cls_cmd = ['python', 'scripts/train_classifier_brats.py', '--config', cls_config_path]
        seg_cmd = ['python', 'scripts/train_segmentation.py', '--config', seg_config_path]
        
        if cls_overrides:
            print_warning("Parameter overrides require config file modification")
            print_info("Using config file as-is for now")
        if seg_overrides:
            print_warning("Parameter overrides require config file modification")
            print_info("Using config file as-is for now")
        
        try:
            print_info(f"Command: {' '.join(cls_cmd)}")
            result = subprocess.run(cls_cmd, check=True, cwd=str(self.project_root))
            
            print_info(f"Command: {' '.join(seg_cmd)}")
            result = subprocess.run(seg_cmd, check=True, cwd=str(self.project_root))
            
            elapsed = time.time() - start_time
            self.results['classification_brats'] = {
                'status': 'success',
                'elapsed_time': elapsed,
                'config': cls_config_path,
            }
            self.results['segmentation'] = {
                'status': 'success',
                'elapsed_time': elapsed,
                'config': seg_config_path,
            }
            print_success(f"BraTS E2E training completed in {elapsed/60:.1f} minutes")
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            self.results['classification_brats'] = {
                'status': 'failed',
                'elapsed_time': elapsed,
                'error': str(e),
            }
            self.results['segmentation'] = {
                'status': 'failed',
                'elapsed_time': elapsed,
                'error': str(e),
            }
            print_error(f"BraTS E2E training failed: {e}")
    
    def _print_summary(self):
        """Print training summary."""
        print_header("Training Summary")
        
        if self.results['classification']:
            print_section("Classification Results")
            cls_result = self.results['classification']
            if cls_result['status'] == 'success':
                print_success(f"Status: Completed successfully")
                print_info(f"Time: {cls_result['elapsed_time']/60:.1f} minutes")
                print_info(f"Config: {cls_result['config']}")
            else:
                print_error(f"Status: Failed")
                print_error(f"Error: {cls_result.get('error', 'Unknown')}")
        
        if self.results['classification_brats']:
            print_section("Classification (BraTS) Results")
            cls_brats_result = self.results['classification_brats']
            if cls_brats_result['status'] == 'success':
                print_success(f"Status: Completed successfully")
                print_info(f"Time: {cls_brats_result['elapsed_time']/60:.1f} minutes")
                print_info(f"Config: {cls_brats_result['config']}")
            else:
                print_error(f"Status: Failed")
                print_error(f"Error: {cls_brats_result.get('error', 'Unknown')}")
        
        if self.results['segmentation']:
            print_section("Segmentation Results")
            seg_result = self.results['segmentation']
            if seg_result['status'] == 'success':
                print_success(f"Status: Completed successfully")
                print_info(f"Time: {seg_result['elapsed_time']/60:.1f} minutes")
                print_info(f"Config: {seg_result['config']}")
            else:
                print_error(f"Status: Failed")
                print_error(f"Error: {seg_result.get('error', 'Unknown')}")
        
        print_section("Next Steps")
        if self.results['classification'] and self.results['classification']['status'] == 'success':
            print("• Evaluate classifier: python scripts/evaluate_classifier.py")
            print("• Generate Grad-CAM: python scripts/generate_gradcam.py")
        
        if self.results['classification_brats'] and self.results['classification_brats']['status'] == 'success':
            print("• Evaluate classifier (BraTS): python scripts/evaluate_classifier_brats.py")
        
        if self.results['segmentation'] and self.results['segmentation']['status'] == 'success':
            print("• Evaluate segmentation: python scripts/evaluate_segmentation.py")
        
        print("• Run demo app: python scripts/run_demo.py")
        print("• Run full E2E test: python scripts/test_full_e2e_phase1_to_phase6.py")
    
    def run_quick_mode(self, mode: str, config_type: str = 'quick_test'):
        """Run training in quick mode without interaction."""
        print_header(f"Quick Mode: {mode.upper()}")
        
        if mode == 'classification' or mode == 'cls':
            config_path = CONFIGS['classification'].get(config_type)
            if not config_path:
                print_error(f"Config type '{config_type}' not found for classification")
                return
            self._run_classification_training(config_path)
        
        elif mode == 'classification_brats' or mode == 'cls_brats':
            config_path = CONFIGS['classification_brats'].get(config_type)
            if not config_path:
                print_error(f"Config type '{config_type}' not found for classification_brats")
                return
            self._run_classification_brats_training(config_path)
        
        elif mode == 'segmentation' or mode == 'seg':
            config_path = CONFIGS['segmentation'].get(config_type)
            if not config_path:
                print_error(f"Config type '{config_type}' not found for segmentation")
                return
            self._run_segmentation_training(config_path)
        
        elif mode == 'both':
            cls_config = CONFIGS['classification'].get(config_type)
            seg_config = CONFIGS['segmentation'].get(config_type)
            
            if cls_config:
                self._run_classification_training(cls_config)
            if seg_config:
                self._run_segmentation_training(seg_config)
        
        elif mode == 'brats_e2e':
            cls_config = CONFIGS['classification_brats'].get(config_type)
            seg_config = CONFIGS['segmentation'].get(config_type)
            
            if cls_config:
                self._run_classification_brats_training(cls_config)
            if seg_config:
                self._run_segmentation_training(seg_config)
        
        self._print_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SliceWise Training Controller - Interactive training orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python scripts/train_controller.py
  
  # Quick classification training
  python scripts/train_controller.py --mode cls
  
  # Quick segmentation training
  python scripts/train_controller.py --mode seg
  
  # Train both models
  python scripts/train_controller.py --mode both
  
  # Production training
  python scripts/train_controller.py --mode both --config production

Available Configurations:
  Classification:
    - quick_test: Fast testing config (configs/config_cls.yaml)
    - production: Full production config (configs/config_cls_production.yaml)
  
  Classification (BraTS):
    - quick_test: Fast testing config (configs/config_cls_brats.yaml)
  
  Segmentation:
    - quick_test: Fast testing config (configs/seg2d_quick_test.yaml)
    - baseline: Baseline config (configs/seg2d_baseline.yaml)
    - production: Full production config (configs/seg2d_production.yaml)
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['cls', 'classification', 'cls_brats', 'classification_brats', 'seg', 'segmentation', 'both', 'brats_e2e', 'interactive'],
        default='interactive',
        help='Training mode (default: interactive)',
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='quick_test',
        help='Config type to use (default: quick_test)',
    )
    
    args = parser.parse_args()
    
    controller = TrainingController()
    
    if args.mode == 'interactive':
        controller.run_interactive()
    else:
        controller.run_quick_mode(args.mode, args.config)


if __name__ == "__main__":
    main()
