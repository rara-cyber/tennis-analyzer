"""Example usage of PlayerTracker for tennis video analysis."""

from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker


def main():
    """Main function demonstrating player tracking workflow."""
    # Read video
    print("Reading video...")
    frames, metadata = read_video("input_videos/sample.mp4")
    print(f"✓ Loaded {len(frames)} frames at {metadata['fps']} fps")

    # Initialize tracker
    tracker = PlayerTracker(model_path='yolov8x.pt')

    # Detect players in all frames
    print("Detecting players...")
    player_detections = tracker.detect_frames(
        frames,
        read_from_stub=False,  # Set to True after first run
        stub_path='tracker_stubs/player_detections.pkl'
    )

    print(f"✓ Detected players in {len(player_detections)} frames")

    # Example: Print first frame detections
    if player_detections and player_detections[0]:
        print(f"First frame detections: {player_detections[0]}")
        # Output example: {1: [123.4, 234.5, 178.9, 345.6], 2: [567.8, 234.1, 623.4, 345.2]}
    else:
        print("No players detected in first frame")

    # Draw bounding boxes
    print("Drawing bounding boxes...")
    output_frames = tracker.draw_bounding_boxes(frames, player_detections)

    # Save output video
    print("Saving video...")
    save_video(output_frames, "output_videos/players_detected.mp4", fps=metadata['fps'])

    print("✓ Done!")


if __name__ == "__main__":
    main()
