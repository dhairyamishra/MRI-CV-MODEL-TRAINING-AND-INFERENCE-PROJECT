"""
Comprehensive test runner for Phase 2 implementations.

This script runs all tests and generates a detailed report.
Results are saved to test_results.txt
"""

import sys
import subprocess
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Output file
output_file = project_root / "test_results.txt"


def print_and_log(text, file_handle):
    """Print to console and write to file."""
    print(text)
    file_handle.write(text + "\n")
    file_handle.flush()


def print_header(text, file_handle):
    """Print a formatted header."""
    line = "\n" + "="*70
    print_and_log(line, file_handle)
    print_and_log(f"  {text}", file_handle)
    print_and_log("="*70 + "\n", file_handle)


def print_section(text, file_handle):
    """Print a formatted section."""
    line = "\n" + "-"*70
    print_and_log(line, file_handle)
    print_and_log(f"  {text}", file_handle)
    print_and_log("-"*70, file_handle)


def run_test_file(test_file, verbose, file_handle):
    """Run a single test file and return results."""
    print_section(f"Running: {test_file.name}", file_handle)
    
    cmd = [sys.executable, "-m", "pytest", str(test_file), "-v" if verbose else ""]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    print_and_log(result.stdout, file_handle)
    if result.stderr:
        print_and_log("STDERR: " + result.stderr, file_handle)
    
    print_and_log(f"\n⏱️  Time: {elapsed_time:.2f}s", file_handle)
    
    return result.returncode == 0, elapsed_time


def main():
    """Main test runner."""
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print_and_log(f"Test Run: {timestamp}", f)
        
        print_header("SliceWise Phase 2 - Comprehensive Test Suite", f)
        
        print_and_log("🔍 Discovering tests...", f)
        
        # Define test files
        test_files = [
            project_root / "tests" / "test_smoke.py",
            project_root / "tests" / "test_classifier.py",
            project_root / "tests" / "test_predictor.py",
            project_root / "tests" / "test_gradcam.py",
            project_root / "tests" / "test_data_pipeline.py",
        ]
        
        # Filter existing test files
        existing_tests = [f for f in test_files if f.exists()]
        
        print_and_log(f"📝 Found {len(existing_tests)} test files:\n", f)
        for test_file in existing_tests:
            print_and_log(f"   - {test_file.name}", f)
        
        # Run tests
        print_header("Running Tests", f)
        
        results = {}
        total_time = 0
        
        for test_file in existing_tests:
            success, elapsed = run_test_file(test_file, verbose=True, file_handle=f)
            results[test_file.name] = success
            total_time += elapsed
        
        # Summary
        print_header("Test Summary", f)
        
        passed = sum(1 for success in results.values() if success)
        failed = len(results) - passed
        
        print_and_log(f"📊 Results:\n", f)
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print_and_log(f"   {status}  {test_name}", f)
        
        print_and_log(f"\n📈 Statistics:", f)
        print_and_log(f"   Total Tests: {len(results)}", f)
        print_and_log(f"   Passed: {passed}", f)
        print_and_log(f"   Failed: {failed}", f)
        print_and_log(f"   Success Rate: {(passed/len(results)*100):.1f}%", f)
        print_and_log(f"   Total Time: {total_time:.2f}s", f)
        
        # Component-specific tests
        print_header("Component Test Status", f)
        
        components = {
            "Classifier Models": "test_classifier.py",
            "Inference Predictor": "test_predictor.py",
            "Grad-CAM": "test_gradcam.py",
            "Data Pipeline": "test_data_pipeline.py",
            "Smoke Tests": "test_smoke.py"
        }
        
        for component, test_file in components.items():
            if test_file in results:
                status = "✅" if results[test_file] else "❌"
                print_and_log(f"   {status} {component}", f)
            else:
                print_and_log(f"   ⚠️  {component} (test not found)", f)
        
        # Recommendations
        print_header("Recommendations", f)
        
        if failed == 0:
            print_and_log("🎉 All tests passed! Your Phase 2 implementation is working correctly.", f)
            print_and_log("\n✅ Ready to proceed with:", f)
            print_and_log("   - Training the classifier", f)
            print_and_log("   - Running the demo application", f)
            print_and_log("   - Moving to Phase 3", f)
        else:
            print_and_log("⚠️  Some tests failed. Please review the errors above.", f)
            print_and_log("\n🔧 Troubleshooting steps:", f)
            print_and_log("   1. Check error messages in failed tests", f)
            print_and_log("   2. Verify all dependencies are installed", f)
            print_and_log("   3. Ensure data is preprocessed (if data tests failed)", f)
            print_and_log("   4. Run individual tests for more details:", f)
            print_and_log(f"      python -m pytest tests/test_<name>.py -v", f)
        
        # Additional checks
        print_header("Additional Checks", f)
        
        # Check if data is available
        data_dir = project_root / "data" / "processed" / "kaggle"
        if data_dir.exists():
            train_files = list((data_dir / "train").glob("*.npz")) if (data_dir / "train").exists() else []
            print_and_log(f"   ✅ Data available: {len(train_files)} training files", f)
        else:
            print_and_log("   ⚠️  Data not preprocessed. Run:", f)
            print_and_log("      python src/data/preprocess_kaggle.py", f)
            print_and_log("      python src/data/split_kaggle.py", f)
        
        # Check if model checkpoint exists
        checkpoint_dir = project_root / "checkpoints" / "cls"
        if checkpoint_dir.exists() and (checkpoint_dir / "best_model.pth").exists():
            print_and_log("   ✅ Model checkpoint available", f)
        else:
            print_and_log("   ⚠️  No trained model found. Train a model:", f)
            print_and_log("      python scripts/train_classifier.py", f)
        
        # Exit code
        print_and_log("\n" + "="*70, f)
        if failed == 0:
            print_and_log("✅ ALL TESTS PASSED", f)
            print_and_log("="*70 + "\n", f)
            print_and_log(f"\n📄 Results saved to: {output_file}", f)
            sys.exit(0)
        else:
            print_and_log(f"❌ {failed} TEST(S) FAILED", f)
            print_and_log("="*70 + "\n", f)
            print_and_log(f"\n📄 Results saved to: {output_file}", f)
            sys.exit(1)


if __name__ == "__main__":
    main()
