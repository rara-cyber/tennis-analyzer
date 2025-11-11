"""Verification script to check ball tracker installation."""

import os
import sys


def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists


def check_directory_exists(dirpath, description):
    """Check if a directory exists and report status."""
    exists = os.path.isdir(dirpath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {dirpath}")
    return exists


def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"    Syntax error: {e}")
        return False


def main():
    """Run verification checks."""
    print("="*60)
    print("Ball Tracker Module Verification")
    print("="*60)
    print()

    all_checks_passed = True

    # Check directory structure
    print("1. Directory Structure:")
    checks = [
        check_directory_exists("trackers", "Trackers module"),
        check_directory_exists("utils", "Utils module"),
        check_directory_exists("tests", "Tests directory"),
        check_directory_exists("examples", "Examples directory"),
        check_directory_exists("models", "Models directory"),
        check_directory_exists("tracker_stubs", "Tracker stubs directory"),
        check_directory_exists("input_videos", "Input videos directory"),
        check_directory_exists("output_videos", "Output videos directory"),
    ]
    all_checks_passed &= all(checks)
    print()

    # Check core module files
    print("2. Core Module Files:")
    checks = [
        check_file_exists("trackers/__init__.py", "Trackers __init__"),
        check_file_exists("trackers/ball_tracker.py", "BallTracker class"),
        check_file_exists("utils/__init__.py", "Utils __init__"),
        check_file_exists("utils/video_utils.py", "Video utilities"),
    ]
    all_checks_passed &= all(checks)
    print()

    # Check test files
    print("3. Test Files:")
    checks = [
        check_file_exists("tests/__init__.py", "Tests __init__"),
        check_file_exists("tests/test_ball_tracker.py", "BallTracker tests"),
    ]
    all_checks_passed &= all(checks)
    print()

    # Check example files
    print("4. Example Files:")
    checks = [
        check_file_exists("examples/ball_tracking_example.py", "Ball tracking example"),
    ]
    all_checks_passed &= all(checks)
    print()

    # Check configuration files
    print("5. Configuration Files:")
    checks = [
        check_file_exists("requirements.txt", "Requirements file"),
        check_file_exists("download_models.py", "Model download script"),
        check_file_exists("BALL_TRACKER_README.md", "Documentation"),
    ]
    all_checks_passed &= all(checks)
    print()

    # Check Python syntax
    print("6. Python Syntax Validation:")
    python_files = [
        "trackers/ball_tracker.py",
        "utils/video_utils.py",
        "tests/test_ball_tracker.py",
        "examples/ball_tracking_example.py",
        "download_models.py",
    ]

    for filepath in python_files:
        if os.path.exists(filepath):
            valid = check_python_syntax(filepath)
            status = "✓" if valid else "✗"
            print(f"  {status} {filepath}")
            all_checks_passed &= valid
        else:
            print(f"  ✗ {filepath} (file not found)")
            all_checks_passed = False
    print()

    # Check module imports (basic)
    print("7. Module Import Check:")
    try:
        # Just check if files can be compiled, don't import
        # (to avoid dependency issues)
        import py_compile

        files_to_check = [
            "trackers/ball_tracker.py",
            "utils/video_utils.py",
        ]

        for filepath in files_to_check:
            try:
                py_compile.compile(filepath, doraise=True)
                print(f"  ✓ {filepath} can be compiled")
            except py_compile.PyCompileError as e:
                print(f"  ✗ {filepath} has compilation errors")
                print(f"    {e}")
                all_checks_passed = False
    except Exception as e:
        print(f"  ⚠ Could not check imports: {e}")
    print()

    # Summary
    print("="*60)
    if all_checks_passed:
        print("✓ All verification checks passed!")
        print("="*60)
        print()
        print("Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Download models: python download_models.py")
        print("  3. Run tests: pytest tests/test_ball_tracker.py -v")
        print("  4. Run example: python examples/ball_tracking_example.py")
        return 0
    else:
        print("✗ Some verification checks failed!")
        print("="*60)
        print()
        print("Please check the errors above and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
