#!/usr/bin/env python3
"""Test script to verify VLAb installation."""
import sys

def test_import(import_statement, description):
    """Test if an import statement works."""
    try:
        exec(import_statement)
        return True, description
    except ImportError as e:
        return False, f"{description} - Error: {e}"
    except Exception as e:
        return False, f"{description} - Error: {e}"

def main():
    """Run all installation tests."""
    tests = [
        ("from lerobot.configs.train import TrainPipelineConfig", "TrainPipelineConfig"),
        ("from lerobot.policies.factory import make_policy", "Policy factory"),
        ("from lerobot.datasets.factory import make_dataset", "Dataset factory"),
    ]
    
    results = []
    for import_statement, description in tests:
        success, message = test_import(import_statement, description)
        results.append((success, message))
    
    # Print results
    print("\n" + "="*60)
    print("VLAb Installation Test")
    print("="*60)
    
    all_passed = True
    for success, message in results:
        status = "✓" if success else "✗"
        print(f"{status} {message}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check your installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()

