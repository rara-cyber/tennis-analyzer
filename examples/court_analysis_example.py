"""
Court Analysis Example

Demonstrates court keypoint detection and mini-court visualization.
"""
import sys
import os

# Add parent directory to path
sys.path.append('..')

import cv2
from utils.video_utils import read_video, save_video
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt


def main():
    """Main example function."""

    print("=" * 60)
    print("Court Analysis Example")
    print("=" * 60)
    print()

    # Configuration
    input_video = "input_videos/sample.mp4"
    model_path = "models/court_keypoint_model.pth"
    output_keypoints_image = "output_videos/court_keypoints.jpg"
    output_video = "output_videos/with_mini_court.mp4"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("\nPlease run: python download_models.py --placeholder")
        print("This will create a placeholder model for testing.")
        return

    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"✗ Input video not found: {input_video}")
        print(f"\nPlease place a video file at: {input_video}")
        print("\nAlternatively, creating a demo frame for visualization...")

        # Create a demo frame
        demo_frame = create_demo_frame()
        demo_court_analysis(demo_frame, model_path)
        return

    # Read video
    print("Reading video...")
    frames, metadata = read_video(input_video)
    print(f"✓ Read {len(frames)} frames ({metadata['width']}x{metadata['height']} @ {metadata['fps']} fps)")
    print()

    # Initialize court detector
    print("Initializing court detector...")
    detector = CourtLineDetector(model_path=model_path)
    print()

    # Detect court keypoints (only first frame - static camera)
    print("Detecting court keypoints...")
    court_keypoints = detector.predict(frames[0])
    print(f"✓ Detected keypoints: {court_keypoints[:8]}... (showing first 4 points)")
    print()

    # Draw keypoints on first frame for verification
    print("Saving keypoints visualization...")
    frame_with_keypoints = detector.draw_keypoints(frames[0], court_keypoints)
    os.makedirs("output_videos", exist_ok=True)
    cv2.imwrite(output_keypoints_image, frame_with_keypoints)
    print(f"✓ Saved: {output_keypoints_image}")
    print()

    # Initialize mini-court
    print("Initializing mini-court...")
    mini_court = MiniCourt(frames[0])
    print()

    # Draw mini-court on all frames
    print("Drawing mini-court on frames...")
    output_frames = []
    for i, frame in enumerate(frames):
        # Draw mini-court
        frame_with_court = mini_court.draw_court(frame)

        # Optionally add frame number
        cv2.putText(
            frame_with_court,
            f"Frame: {i+1}/{len(frames)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        output_frames.append(frame_with_court)

        # Show progress
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            progress = (i + 1) / len(frames) * 100
            print(f"\rProgress: {progress:.1f}% ({i+1}/{len(frames)})", end='', flush=True)

    print()
    print()

    # Save output video
    print("Saving output video...")
    save_video(output_frames, output_video, fps=metadata['fps'])
    print(f"✓ Saved: {output_video}")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Keypoints image: {output_keypoints_image}")
    print(f"  - Video with mini-court: {output_video}")


def create_demo_frame():
    """Create a demo frame for testing."""
    # Create a simple tennis court image
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Fill with green (grass)
    frame[:, :] = (34, 139, 34)

    # Draw a simple court outline
    cv2.rectangle(frame, (200, 100), (1080, 620), (255, 255, 255), 3)
    cv2.line(frame, (200, 360), (1080, 360), (255, 255, 255), 3)  # Net
    cv2.rectangle(frame, (350, 100), (930, 620), (255, 255, 255), 2)  # Singles lines

    return frame


def demo_court_analysis(frame, model_path):
    """Demonstrate court analysis with a demo frame."""
    print("\n" + "=" * 60)
    print("Running demo with synthetic frame")
    print("=" * 60)
    print()

    # Initialize detector
    print("Initializing court detector...")
    detector = CourtLineDetector(model_path=model_path)
    print()

    # Detect keypoints
    print("Detecting court keypoints...")
    keypoints = detector.predict(frame)
    print(f"✓ Detected {len(keypoints)//2} keypoints")
    print()

    # Draw keypoints
    frame_with_keypoints = detector.draw_keypoints(frame, keypoints)

    # Initialize mini-court
    print("Initializing mini-court...")
    mini_court = MiniCourt(frame)
    print()

    # Draw mini-court
    frame_with_court = mini_court.draw_court(frame)

    # Draw some example positions on mini-court
    example_positions = {
        1: (mini_court.court_start_x + 50, mini_court.court_start_y + 100),
        2: (mini_court.court_end_x - 50, mini_court.court_end_y - 100)
    }

    frame_with_court = mini_court.draw_points_on_mini_court(
        frame_with_court,
        example_positions,
        color=(0, 0, 255)
    )

    # Save outputs
    os.makedirs("output_videos", exist_ok=True)
    cv2.imwrite("output_videos/demo_keypoints.jpg", frame_with_keypoints)
    cv2.imwrite("output_videos/demo_mini_court.jpg", frame_with_court)

    print("✓ Saved demo outputs:")
    print("  - output_videos/demo_keypoints.jpg")
    print("  - output_videos/demo_mini_court.jpg")
    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()
