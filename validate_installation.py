"""
Installation Validation Script

Checks that all modules can be imported and dependencies are available.
"""
import sys
import os


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    print()

    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pytest': 'PyTest'
    }

    missing = []
    installed = []

    for module, name in dependencies.items():
        try:
            __import__(module)
            installed.append(name)
            print(f"  ✓ {name}")
        except ImportError:
            missing.append(name)
            print(f"  ✗ {name} (not installed)")

    print()

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("\nTo install missing dependencies, run:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("✓ All dependencies installed!")
        return True


def check_modules():
    """Check if project modules can be imported."""
    print("\nChecking project modules...")
    print()

    modules = [
        ('constants', 'Constants'),
        ('utils.conversions', 'Coordinate Conversions'),
        ('utils.video_utils', 'Video Utilities'),
        ('utils.bbox_utils', 'Bounding Box Utilities'),
    ]

    # Modules that require torch
    torch_modules = [
        ('court_line_detector', 'Court Line Detector'),
        ('mini_court', 'Mini Court'),
    ]

    success = True

    # Check basic modules
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            success = False

    # Check torch-dependent modules
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    for module, name in torch_modules:
        if not has_torch:
            print(f"  ⊘ {name} (requires PyTorch)")
        else:
            try:
                __import__(module)
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                success = False

    print()

    if success or not has_torch:
        if not has_torch:
            print("⚠ Some modules not checked due to missing PyTorch")
        else:
            print("✓ All modules can be imported!")
        return True
    else:
        print("✗ Some modules failed to import")
        return False


def check_structure():
    """Check if directory structure is correct."""
    print("\nChecking project structure...")
    print()

    required_dirs = [
        'court_line_detector',
        'mini_court',
        'utils',
        'constants',
        'tests',
        'examples',
        'models',
        'input_videos',
        'output_videos',
    ]

    missing_dirs = []

    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ (missing)")
            missing_dirs.append(directory)

    print()

    if missing_dirs:
        print(f"Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("✓ All directories present!")
        return True


def check_model_files():
    """Check if model files are present."""
    print("\nChecking model files...")
    print()

    model_files = [
        'models/court_keypoint_model.pth'
    ]

    all_present = True

    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            size_mb = size / (1024 * 1024)
            print(f"  ✓ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {model_file} (not found)")
            all_present = False

    print()

    if not all_present:
        print("⚠ Some model files are missing")
        print("\nTo create placeholder models for testing:")
        print("  python download_models.py --placeholder")
        return False
    else:
        print("✓ Model files present!")
        return True


def main():
    """Main validation function."""
    print("=" * 60)
    print("Tennis Analysis System - Installation Validation")
    print("=" * 60)
    print()

    checks = [
        ("Dependencies", check_dependencies),
        ("Project Structure", check_structure),
        ("Project Modules", check_modules),
        ("Model Files", check_model_files),
    ]

    results = {}

    for name, check_func in checks:
        result = check_func()
        results[name] = result

    print()
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print()

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print()

    all_passed = all(results.values())

    if all_passed:
        print("✓ All checks passed! System is ready to use.")
        print()
        print("Next steps:")
        print("  1. Place input videos in input_videos/")
        print("  2. Run example: cd examples && python court_analysis_example.py")
        print("  3. Run tests: pytest tests/ -v")
        return 0
    else:
        print("⚠ Some checks failed. Please resolve the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
