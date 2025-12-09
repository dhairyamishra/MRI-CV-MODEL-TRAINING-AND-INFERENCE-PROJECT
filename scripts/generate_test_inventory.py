"""
Generate Test Inventory CSV

This script extracts all test functions from the tests/ directory
and generates a comprehensive CSV inventory with test details.

Usage:
    python scripts/generate_test_inventory.py
"""

import ast
import csv
from pathlib import Path
import re


def extract_docstring(node):
    """Extract docstring from a function node."""
    if (isinstance(node, ast.FunctionDef) and 
        node.body and 
        isinstance(node.body[0], ast.Expr) and
        isinstance(node.body[0].value, ast.Constant)):
        return node.body[0].value.value
    return ""


def extract_tests_from_file(file_path):
    """Extract all test functions from a Python file."""
    tests = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find all classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                class_docstring = extract_docstring(node)
                
                # Find all test methods in the class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                        test_name = item.name
                        test_docstring = extract_docstring(item)
                        
                        tests.append({
                            'file': file_path.name,
                            'class': class_name,
                            'function': test_name,
                            'class_docstring': class_docstring,
                            'test_docstring': test_docstring
                        })
        
        # Also check for test functions not in classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Check if it's not inside a class
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    test_name = node.name
                    test_docstring = extract_docstring(node)
                    
                    tests.append({
                        'file': file_path.name,
                        'class': '',
                        'function': test_name,
                        'class_docstring': '',
                        'test_docstring': test_docstring
                    })
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return tests


def categorize_test(file_path):
    """Categorize test based on directory structure."""
    parts = file_path.parts
    
    if 'unit' in parts:
        return 'Unit Tests'
    elif 'integration' in parts:
        return 'Integration Tests'
    elif 'e2e' in parts:
        return 'E2E Tests'
    elif 'performance' in parts:
        return 'Performance Tests'
    elif 'production' in parts:
        return 'Production Tests'
    else:
        return 'Other Tests'


def get_subcategory(file_name):
    """Get subcategory based on file name."""
    subcategories = {
        'data_preprocessing': 'Data Preprocessing Safety',
        'dataset_integrity': 'Dataset Integrity',
        'model_validation': 'Model Validation',
        'individual_model': 'Individual Model Validation',
        'multitask_model': 'Multi-Task Model Validation',
        'loss_function': 'Loss Function Validation',
        'results_validation': 'Results Validation',
        'transform_pipeline': 'Transform Pipeline',
        'visualizations': 'Visualizations Generation',
        'logging_system': 'Logging System',
        'fastapi_endpoints': 'FastAPI Endpoints',
        'api_integration': 'API Integration & Performance',
        'backend_integration': 'Backend Integration',
        'external_integration': 'External Integration',
        'frontend_backend': 'Frontend-Backend Integration',
        'streamlit_ui': 'Streamlit UI Components',
        'user_workflow': 'User Workflow Validation',
        'inference_performance': 'Inference Performance',
        'cross_platform': 'Cross-Platform Compatibility',
        'deployment': 'Deployment & Production',
        'security_compliance': 'Security & Compliance'
    }
    
    for key, value in subcategories.items():
        if key in file_name.lower():
            return value
    
    return 'General'


def generate_description(test_docstring, class_docstring):
    """Generate a description from docstrings."""
    if test_docstring:
        # Clean up the docstring
        desc = test_docstring.strip().replace('\n', ' ').replace('  ', ' ')
        return desc
    elif class_docstring:
        return class_docstring.strip().replace('\n', ' ').replace('  ', ' ')
    else:
        return "No description available"


def generate_expected_result(test_name, description):
    """Generate expected result based on test name and description."""
    # Common patterns
    if 'validation' in test_name or 'validate' in description.lower():
        return "Validation passes without errors"
    elif 'error' in test_name or 'handling' in test_name:
        return "Errors handled gracefully with appropriate messages"
    elif 'performance' in test_name or 'latency' in test_name:
        return "Performance meets clinical requirements"
    elif 'security' in test_name or 'compliance' in test_name:
        return "Security/compliance requirements met"
    elif 'integration' in test_name:
        return "Integration works correctly end-to-end"
    elif 'workflow' in test_name:
        return "Workflow completes successfully"
    else:
        return "Test passes with expected behavior"


def main():
    """Main function to generate test inventory."""
    # Get project root
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / 'tests'
    output_file = tests_dir / 'TEST_INVENTORY.csv'
    
    print(f"Scanning tests directory: {tests_dir}")
    
    # Find all Python test files
    test_files = list(tests_dir.rglob('test_*.py'))
    print(f"Found {len(test_files)} test files")
    
    # Extract all tests
    all_tests = []
    test_id = 1
    
    for test_file in sorted(test_files):
        print(f"Processing: {test_file.relative_to(tests_dir)}")
        tests = extract_tests_from_file(test_file)
        
        for test in tests:
            category = categorize_test(test_file.relative_to(tests_dir))
            subcategory = get_subcategory(test['file'])
            description = generate_description(test['test_docstring'], test['class_docstring'])
            expected_result = generate_expected_result(test['function'], description)
            
            all_tests.append({
                'Test ID': test_id,
                'Test File': test['file'],
                'Test Class': test['class'],
                'Test Function': test['function'],
                'Category': category,
                'Subcategory': subcategory,
                'Description': description,
                'Expected Result': expected_result
            })
            test_id += 1
    
    # Write to CSV
    print(f"\nWriting {len(all_tests)} tests to {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Test ID',
            'Test File',
            'Test Class',
            'Test Function',
            'Category',
            'Subcategory',
            'Description',
            'Expected Result'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_tests)
    
    print(f"\n✅ Test inventory generated successfully!")
    print(f"📊 Total tests: {len(all_tests)}")
    print(f"📁 Output file: {output_file}")
    
    # Print summary by category
    print("\n📋 Summary by Category:")
    categories = {}
    for test in all_tests:
        cat = test['Category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} tests")


if __name__ == '__main__':
    main()
