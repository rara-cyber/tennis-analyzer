"""Download fine-tuned tennis ball detection model."""

import urllib.request
import os
import sys


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


def main():
    """Main function to download models."""
    print("="*60)
    print("Tennis Ball Detection Model Downloader")
    print("="*60)
    print()

    # Try to download fine-tuned model
    print("Option 1: Fine-tuned tennis ball model (recommended)")
    response = input("Download fine-tuned model? (y/n): ")

    if response.lower() == 'y':
        download_ball_model()
    else:
        print("Skipping fine-tuned model download")

    # Offer generic model as fallback
    print("\n" + "="*60)
    print("Option 2: Generic YOLOv8n model (fallback)")
    print("="*60)
    response = input("Download generic YOLOv8n model? (y/n): ")

    if response.lower() == 'y':
        download_yolov8_fallback()
    else:
        print("Skipping generic model download")

    print("\n" + "="*60)
    print("✓ Model setup complete!")
    print("="*60)


if __name__ == "__main__":
    main()
