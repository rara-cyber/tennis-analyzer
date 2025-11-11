"""
Example usage of video processing utilities.

This example demonstrates how to:
1. Read a video file
2. Add frame numbers to frames
3. Save the processed video
"""

from utils.video_utils import read_video, save_video, draw_frame_number


def main():
    """Main example function."""
    # Input and output paths
    input_path = "input_videos/sample.mp4"
    output_path = "output_videos/sample_with_frames.mp4"

    print("=" * 60)
    print("Video Processing Example")
    print("=" * 60)

    # Read input video
    print(f"\n📹 Reading video from: {input_path}")
    try:
        frames, metadata = read_video(input_path)
        print(f"✓ Loaded {len(frames)} frames")
        print(f"  - FPS: {metadata['fps']:.2f}")
        print(f"  - Resolution: {metadata['width']}x{metadata['height']}")
        print(f"  - Duration: {metadata['duration']:.2f} seconds")
    except FileNotFoundError:
        print(f"\n❌ Error: Video file not found at {input_path}")
        print("\nTo run this example:")
        print("1. Create an 'input_videos' directory")
        print("2. Add a sample video file named 'sample.mp4'")
        print("3. Run this script again")
        return
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return

    # Draw frame numbers
    print(f"\n🎨 Adding frame numbers...")
    frames_with_numbers = draw_frame_number(
        frames,
        start_frame=0,
        position=(10, 30),
        color=(255, 255, 255),
        font_scale=1.0
    )
    print(f"✓ Frame numbers added to {len(frames_with_numbers)} frames")

    # Save output video
    print(f"\n💾 Saving output video to: {output_path}")
    save_video(
        frames_with_numbers,
        output_path,
        fps=metadata['fps']
    )
    print(f"✓ Video saved successfully!")

    print("\n" + "=" * 60)
    print("✅ Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
