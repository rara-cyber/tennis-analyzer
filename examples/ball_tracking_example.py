"""Example script demonstrating ball tracking functionality."""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import read_video, save_video
from trackers.ball_tracker import BallTracker


def main():
    """Run ball tracking example."""
    # Configuration
    input_video = "input_videos/sample.mp4"
    output_video = "output_videos/ball_tracked.mp4"
    model_path = "models/yolov5_tennis_ball.pt"
    cache_path = "tracker_stubs/ball_detections.pkl"

    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"⚠ Input video not found: {input_video}")
        print("  Please place a tennis video in the input_videos/ directory")
        print("  Or update the input_video path in this script")
        return

    # Check if model exists, fallback to generic YOLOv8n
    if not os.path.exists(model_path):
        print(f"⚠ Fine-tuned model not found: {model_path}")
        print("  Using generic YOLOv8n model instead")
        print("  For better results, download the fine-tuned model using:")
        print("  python download_models.py")
        model_path = "yolov8n.pt"

    # Read video
    print("="*60)
    print("STEP 1: Reading video")
    print("="*60)
    frames, metadata = read_video(input_video)

    # Initialize tracker
    print("\n" + "="*60)
    print("STEP 2: Initializing ball tracker")
    print("="*60)
    tracker = BallTracker(model_path=model_path)

    # Detect ball in all frames
    print("\n" + "="*60)
    print("STEP 3: Detecting ball in all frames")
    print("="*60)
    ball_detections = tracker.detect_frames(
        frames,
        read_from_stub=False,  # Set to True after first run to use cache
        stub_path=cache_path
    )

    # Interpolate missing detections
    print("\n" + "="*60)
    print("STEP 4: Interpolating ball positions")
    print("="*60)
    ball_detections = tracker.interpolate_ball_positions(ball_detections)

    # Detect shots
    print("\n" + "="*60)
    print("STEP 5: Detecting shot events")
    print("="*60)
    shot_frames = tracker.get_ball_shot_frames(ball_detections)

    # Draw bounding boxes
    print("\n" + "="*60)
    print("STEP 6: Drawing ball bounding boxes")
    print("="*60)
    output_frames = tracker.draw_bounding_boxes(frames, ball_detections)

    # Save output
    print("\n" + "="*60)
    print("STEP 7: Saving output video")
    print("="*60)
    save_video(output_frames, output_video, fps=metadata['fps'])

    print("\n" + "="*60)
    print("✓ Ball tracking complete!")
    print("="*60)
    print(f"Output saved to: {output_video}")
    print(f"Shots detected at frames: {shot_frames}")
    print(f"Total shots: {len(shot_frames)}")


if __name__ == "__main__":
    main()
