"""
Full End-to-End Run Script for SliceWise Project
Orchestrates the complete pipeline validation:
1. Prerequisites check
2. Run comprehensive E2E tests (Phases 1-6)
3. Start backend API (optional)
4. Run demo application (optional)
5. Generate comprehensive report

Usage:
    python scripts/run_full_e2e.py [--with-api] [--with-demo] [--quick]
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def print_step(step_num, total_steps, message):
    """Print step message."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}[Step {step_num}/{total_steps}] {message}{Colors.ENDC}")


def check_prerequisites():
    """Check if all prerequisites are met."""
    print_header("Prerequisites Check")
    
    all_good = True
    
    # Check data
    print_info("Checking data availability...")
    kaggle_train = project_root / "data" / "processed" / "kaggle" / "train"
    if kaggle_train.exists() and len(list(kaggle_train.glob("*.npz"))) > 0:
        num_files = len(list(kaggle_train.glob("*.npz")))
        print_success(f"Kaggle train set: {num_files} files")
    else:
        print_error("Kaggle train set not found")
        all_good = False
    
    kaggle_val = project_root / "data" / "processed" / "kaggle" / "val"
    if kaggle_val.exists() and len(list(kaggle_val.glob("*.npz"))) > 0:
        num_files = len(list(kaggle_val.glob("*.npz")))
        print_success(f"Kaggle val set: {num_files} files")
    else:
        print_error("Kaggle val set not found")
        all_good = False
    
    kaggle_test = project_root / "data" / "processed" / "kaggle" / "test"
    if kaggle_test.exists() and len(list(kaggle_test.glob("*.npz"))) > 0:
        num_files = len(list(kaggle_test.glob("*.npz")))
        print_success(f"Kaggle test set: {num_files} files")
    else:
        print_error("Kaggle test set not found")
        all_good = False
    
    # Check checkpoints
    print_info("\nChecking model checkpoints...")
    cls_checkpoint = project_root / "checkpoints" / "cls" / "best_model.pth"
    if cls_checkpoint.exists():
        print_success(f"Classifier checkpoint: {cls_checkpoint.name}")
    else:
        print_warning("Classifier checkpoint not found (some tests will be skipped)")
    
    seg_checkpoint = project_root / "checkpoints" / "seg" / "best_model.pth"
    if seg_checkpoint.exists():
        print_success(f"Segmentation checkpoint: {seg_checkpoint.name}")
    else:
        print_warning("Segmentation checkpoint not found (some tests will be skipped)")
    
    # Check production checkpoints
    cls_prod_checkpoint = project_root / "checkpoints" / "cls_production" / "best_model.pth"
    if cls_prod_checkpoint.exists():
        print_success(f"Production classifier checkpoint: {cls_prod_checkpoint.name}")
    else:
        print_warning("Production classifier checkpoint not found")
    
    seg_prod_checkpoint = project_root / "checkpoints" / "seg_production" / "best_model.pth"
    if seg_prod_checkpoint.exists():
        print_success(f"Production segmentation checkpoint: {seg_prod_checkpoint.name}")
    else:
        print_warning("Production segmentation checkpoint not found")
    
    # Check calibration
    calibration_checkpoint = project_root / "outputs" / "calibration" / "temperature_scaler.pth"
    if calibration_checkpoint.exists():
        print_success("Calibration checkpoint found")
    else:
        print_warning("Calibration checkpoint not found (Phase 4 tests may be limited)")
    
    print()
    return all_good


def run_e2e_tests(quick_mode=False):
    """Run the comprehensive E2E test suite."""
    print_header("Running Comprehensive E2E Tests")
    
    cmd = [sys.executable, str(project_root / "scripts" / "test_full_e2e_phase1_to_phase6.py")]
    if quick_mode:
        cmd.append("--quick")
    
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, cwd=str(project_root), check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"E2E tests failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"Failed to run E2E tests: {str(e)}")
        return False


def start_backend_api():
    """Start the backend API server."""
    print_header("Starting Backend API")
    
    print_info("Starting backend server on http://localhost:8000")
    print_info("Press Ctrl+C to stop the server")
    print()
    
    cmd = [sys.executable, str(project_root / "scripts" / "run_demo_backend.py")]
    
    try:
        subprocess.run(cmd, cwd=str(project_root))
    except KeyboardInterrupt:
        print_warning("\nBackend server stopped by user")
    except Exception as e:
        print_error(f"Failed to start backend: {str(e)}")


def run_demo_app():
    """Run the full demo application."""
    print_header("Starting Demo Application")
    
    print_info("Starting both backend and frontend...")
    print_info("Backend: http://localhost:8000")
    print_info("Frontend: http://localhost:8501")
    print_info("Press Ctrl+C to stop")
    print()
    
    cmd = [sys.executable, str(project_root / "scripts" / "run_demo.py")]
    
    try:
        subprocess.run(cmd, cwd=str(project_root))
    except KeyboardInterrupt:
        print_warning("\nDemo application stopped by user")
    except Exception as e:
        print_error(f"Failed to start demo: {str(e)}")


def generate_report():
    """Generate a comprehensive test report."""
    print_header("Generating Comprehensive Report")
    
    results_file = project_root / "full_e2e_test_results.json"
    
    if not results_file.exists():
        print_warning("No test results found. Run E2E tests first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Print summary
    print_info("Test Summary:")
    print(f"  Start Time: {results.get('start_time', 'N/A')}")
    print(f"  End Time: {results.get('end_time', 'N/A')}")
    print(f"  Total Tests: {results.get('total_tests', 0)}")
    print(f"  Passed: {Colors.OKGREEN}{results.get('passed_tests', 0)}{Colors.ENDC}")
    print(f"  Failed: {Colors.FAIL}{results.get('failed_tests', 0)}{Colors.ENDC}")
    print(f"  Warnings: {Colors.WARNING}{results.get('warnings', 0)}{Colors.ENDC}")
    
    total = results.get('total_tests', 0)
    passed = results.get('passed_tests', 0)
    pass_rate = (passed / total * 100) if total > 0 else 0
    print(f"  Pass Rate: {Colors.BOLD}{pass_rate:.1f}%{Colors.ENDC}")
    
    # Phase breakdown
    print(f"\n{Colors.BOLD}Phase Results:{Colors.ENDC}")
    for phase_name, phase_data in results.get('phases', {}).items():
        phase_passed = phase_data.get('passed', 0)
        phase_failed = phase_data.get('failed', 0)
        phase_total = phase_passed + phase_failed
        
        if phase_total > 0:
            phase_rate = (phase_passed / phase_total * 100)
            status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if phase_failed == 0 else f"{Colors.FAIL}✗{Colors.ENDC}"
            print(f"  {status} {phase_name}: {phase_passed}/{phase_total} ({phase_rate:.0f}%)")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Full End-to-End Run for SliceWise Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run E2E tests only
  python scripts/run_full_e2e.py
  
  # Run E2E tests in quick mode
  python scripts/run_full_e2e.py --quick
  
  # Run E2E tests and start backend API
  python scripts/run_full_e2e.py --with-api
  
  # Run E2E tests and start full demo
  python scripts/run_full_e2e.py --with-demo
  
  # Just check prerequisites
  python scripts/run_full_e2e.py --check-only
        """
    )
    
    parser.add_argument("--quick", action="store_true", 
                       help="Quick mode (smaller batches for faster testing)")
    parser.add_argument("--with-api", action="store_true",
                       help="Start backend API after tests")
    parser.add_argument("--with-demo", action="store_true",
                       help="Start full demo application after tests")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check prerequisites, don't run tests")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip E2E tests (useful with --with-api or --with-demo)")
    
    args = parser.parse_args()
    
    # Print banner
    print_header("SliceWise - Full End-to-End Run")
    print_info(f"Project Root: {project_root}")
    print_info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_steps = 1  # Prerequisites
    if not args.check_only and not args.skip_tests:
        total_steps += 1  # E2E tests
    if args.with_api or args.with_demo:
        total_steps += 1  # API/Demo
    
    current_step = 0
    
    # Step 1: Check prerequisites
    current_step += 1
    print_step(current_step, total_steps, "Checking Prerequisites")
    prereqs_ok = check_prerequisites()
    
    if not prereqs_ok:
        print_error("\nPrerequisites check failed. Please fix the issues above.")
        print_info("\nTo setup the project:")
        print("  1. Download and preprocess data:")
        print("     python scripts/download_kaggle_data.py")
        print("     python src/data/preprocess_kaggle.py")
        print("     python src/data/split_kaggle.py")
        print("  2. Train models:")
        print("     python scripts/train_classifier.py")
        print("     python scripts/train_segmentation.py")
        sys.exit(1)
    
    if args.check_only:
        print_success("\nAll prerequisites met! ✓")
        sys.exit(0)
    
    # Step 2: Run E2E tests
    if not args.skip_tests:
        current_step += 1
        print_step(current_step, total_steps, "Running E2E Tests")
        
        tests_passed = run_e2e_tests(quick_mode=args.quick)
        
        if tests_passed:
            print_success("\nE2E tests completed successfully!")
        else:
            print_warning("\nE2E tests completed with issues. Check the output above.")
        
        # Generate report
        generate_report()
    
    # Step 3: Start API or Demo
    if args.with_demo:
        current_step += 1
        print_step(current_step, total_steps, "Starting Demo Application")
        run_demo_app()
    elif args.with_api:
        current_step += 1
        print_step(current_step, total_steps, "Starting Backend API")
        start_backend_api()
    
    # Final message
    print_header("End-to-End Run Complete")
    print_success("All steps completed!")
    print()
    print_info("Next steps:")
    print("  • Review test results in: full_e2e_test_results.json")
    print("  • Start demo: python scripts/run_demo.py")
    print("  • View documentation: documentation/FULL_E2E_TEST_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
