"""
Test Runner Script for Bybit Trading Bot

This script provides easy ways to run different test suites for the trading bot.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Return code: {result.returncode}")
    return result.returncode == 0


def run_unit_tests():
    """Run all unit tests."""
    return run_command(
        "python -m pytest tests/unit/ -v --tb=short",
        "Unit Tests"
    )


def run_integration_tests():
    """Run all integration tests."""
    return run_command(
        "python -m pytest tests/integration/ -v --tb=short",
        "Integration Tests"
    )


def run_e2e_tests():
    """Run all end-to-end tests."""
    return run_command(
        "python -m pytest tests/e2e/ -v --tb=short",
        "End-to-End Tests"
    )


def run_all_tests():
    """Run all tests."""
    return run_command(
        "python -m pytest tests/ -v --tb=short",
        "All Tests"
    )


def run_specific_test(test_path):
    """Run a specific test file or test function."""
    return run_command(
        f"python -m pytest {test_path} -v --tb=short",
        f"Specific Test: {test_path}"
    )


def run_test_with_coverage():
    """Run tests with coverage report."""
    return run_command(
        "python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v",
        "Tests with Coverage"
    )


def run_performance_tests():
    """Run only performance-related tests."""
    return run_command(
        "python -m pytest tests/ -k 'performance' -v --tb=short",
        "Performance Tests"
    )


def check_test_setup():
    """Check if test environment is properly set up."""
    print("Checking test environment setup...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✓ pytest is installed (version: {pytest.__version__})")
    except ImportError:
        print("✗ pytest is not installed")
        return False
    
    # Check if test directories exist
    test_dirs = ['tests', 'tests/unit', 'tests/integration', 'tests/e2e']
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"✓ {test_dir} directory exists")
        else:
            print(f"✗ {test_dir} directory missing")
            return False
    
    # Check if conftest.py exists
    if Path('tests/conftest.py').exists():
        print("✓ tests/conftest.py exists")
    else:
        print("✗ tests/conftest.py missing")
        return False
    
    # Check if source code directories exist
    src_dirs = ['src', 'src/bot', 'src/bot/core', 'src/bot/exchange', 'src/bot/risk_management', 'src/bot/strategies']
    for src_dir in src_dirs:
        if Path(src_dir).exists():
            print(f"✓ {src_dir} directory exists")
        else:
            print(f"✗ {src_dir} directory missing")
            return False
    
    print("\n✓ Test environment appears to be properly set up")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for Bybit Trading Bot")
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--coverage', action='store_true', help='Run tests with coverage')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--check', action='store_true', help='Check test environment setup')
    parser.add_argument('--test', type=str, help='Run specific test file or function')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    success = True
    
    if args.check:
        success = check_test_setup() and success
    
    if args.unit:
        success = run_unit_tests() and success
    
    if args.integration:
        success = run_integration_tests() and success
    
    if args.e2e:
        success = run_e2e_tests() and success
    
    if args.all:
        success = run_all_tests() and success
    
    if args.coverage:
        success = run_test_with_coverage() and success
    
    if args.performance:
        success = run_performance_tests() and success
    
    if args.test:
        success = run_specific_test(args.test) and success
    
    if success:
        print("\n🎉 All requested tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed or encountered errors.")
        sys.exit(1)


if __name__ == '__main__':
    main()