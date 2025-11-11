"""
Model Download Script

Downloads pre-trained models for tennis analysis system.
"""
import urllib.request
import os
import sys


def download_court_model():
    """Download court keypoint detection model."""

    os.makedirs('models', exist_ok=True)

    # Model URL (to be replaced with actual URL when available)
    # Example: https://github.com/your-repo/releases/download/v1.0/court_keypoint_model.pth
    model_url = "https://example.com/models/court_keypoint_model.pth"
    model_path = "models/court_keypoint_model.pth"

    if os.path.exists(model_path):
        print(f"✓ Court model already exists: {model_path}")
        return

    print(f"Downloading court keypoint model from {model_url}...")
    print("Note: Please replace the model_url with the actual URL when available.")

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rProgress: {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(model_url, model_path, reporthook=report_progress)
        print(f"\n✓ Court model downloaded: {model_path}")

    except Exception as e:
        print(f"\n✗ Failed to download court model: {e}")
        print("\nManual download instructions:")
        print("1. Download the model from the project releases")
        print(f"2. Place it in: {model_path}")
        sys.exit(1)


def download_ball_model():
    """Download fine-tuned tennis ball detection model."""

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Model configuration
    # NOTE: Replace with actual model URL when available
    model_url = "https://github.com/your-repo/releases/download/v1.0/yolov5_tennis_ball.pt"
    model_path = "models/yolov5_tennis_ball.pt"

    # Check if model already exists
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            return

    print("="*60)
    print("Downloading tennis ball detection model...")
    print("="*60)
    print(f"URL: {model_url}")
    print(f"Destination: {model_path}")
    print()

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '-' * (bar_length - filled)
            sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}%')
            sys.stdout.flush()

        urllib.request.urlretrieve(model_url, model_path, reporthook=report_progress)
        print("\n")
        print(f"✓ Model downloaded successfully: {model_path}")

    except urllib.error.URLError as e:
        print(f"\n✗ Download failed: {e}")
        print("\nThe model URL may not be available yet.")
        print("You can:")
        print("  1. Use the generic YOLOv8n model (automatic fallback)")
        print("  2. Train your own model (see training/README.md)")
        print("  3. Update the model_url in this script with the correct URL")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


def download_player_detector_model():
    """Download player detection model (YOLO or similar)."""

    os.makedirs('models', exist_ok=True)

    # Player detection model (using YOLOv8 as example)
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    model_path = "models/yolov8x.pt"

    if os.path.exists(model_path):
        print(f"✓ Player detector model already exists: {model_path}")
        return

    print(f"Downloading player detection model...")

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(model_url, model_path, reporthook=report_progress)
        print(f"\n✓ Player detector model downloaded: {model_path}")

    except Exception as e:
        print(f"\n✗ Failed to download player detector model: {e}")
        print("\nManual download instructions:")
        print("1. Download YOLOv8 model from Ultralytics")
        print(f"2. Place it in: {model_path}")


def download_yolov8_fallback():
    """Download generic YOLOv8n model as fallback."""
    print("\n" + "="*60)
    print("Downloading generic YOLOv8n model (fallback)...")
    print("="*60)

    try:
        from ultralytics import YOLO

        # This will auto-download YOLOv8n if not present
        model = YOLO('yolov8n.pt')
        print("✓ Generic YOLOv8n model ready")
        print("  Note: Detection accuracy may be lower than fine-tuned model")

    except Exception as e:
        print(f"✗ Failed to download YOLOv8n: {e}")
        sys.exit(1)


def create_placeholder_models():
    """
    Create placeholder model files for testing.
    These should be replaced with actual trained models.
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    os.makedirs('models', exist_ok=True)

    # Create placeholder court model
    court_model_path = "models/court_keypoint_model.pth"
    if not os.path.exists(court_model_path):
        print("\nCreating placeholder court model for testing...")
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 28)

        # Initialize with random weights
        torch.save(model.state_dict(), court_model_path)
        print(f"✓ Created placeholder model: {court_model_path}")
        print("WARNING: This is a placeholder model and will not produce accurate results.")
        print("Please replace with a properly trained model.")


def main():
    """Main function to download all models."""
    print("=" * 60)
    print("Tennis Analysis System - Model Download")
    print("=" * 60)
    print()

    # Check if we want to create placeholders for testing
    if len(sys.argv) > 1 and sys.argv[1] == '--placeholder':
        create_placeholder_models()
        return

    # Download models
    print("Downloading models...\n")

    download_court_model()
    print()

    download_ball_model()
    print()

    download_player_detector_model()
    print()

    print("=" * 60)
    print("Model download complete!")
    print("=" * 60)
    print("\nNote: If downloads failed, please manually download the models")
    print("and place them in the 'models/' directory.")
    print("\nTo create placeholder models for testing, run:")
    print("  python download_models.py --placeholder")


if __name__ == "__main__":
    main()