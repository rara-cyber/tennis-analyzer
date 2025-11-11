"""
Video Processing Utilities

Functions for reading, writing, and processing video files.
"""
import cv2
import numpy as np
import pickle
import os
from constants import CACHE_DIR, ENABLE_CACHING


def read_video(video_path: str) -> tuple[list[np.ndarray], dict]:
    """
    Read video file and return frames with metadata.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (frames, metadata)
        - frames: List of video frames as numpy arrays
        - metadata: Dict with 'fps', 'width', 'height', 'frame_count'
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Get metadata
    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': len(frames)
    }

    cap.release()
    print(f"✓ Read {len(frames)} frames from {video_path}")

    return frames, metadata


def save_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 24.0,
    codec: str = 'mp4v'
) -> None:
    """
    Save frames to video file.

    Args:
        frames: List of video frames
        output_path: Path to save video
        fps: Frames per second
        codec: Video codec (e.g., 'mp4v', 'XVID')
    """
    if not frames:
        raise ValueError("No frames to save")

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        out.write(frame)

    out.release()
    print(f"✓ Saved {len(frames)} frames to {output_path}")


def display_progress(current: int, total: int, prefix: str = 'Progress') -> None:
    """
    Display progress bar in console.

    Args:
        current: Current progress value
        total: Total value
        prefix: Prefix text for progress bar
    """
    percent = 100 * (current / float(total))
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)

    if current == total:
        print()  # New line when complete


def save_cache(data: any, cache_name: str) -> None:
    """
    Save data to cache file.

    Args:
        data: Data to cache (must be pickle-able)
        cache_name: Name of cache file
    """
    if not ENABLE_CACHING:
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")

    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Saved cache: {cache_path}")


def load_cache(cache_name: str) -> any:
    """
    Load data from cache file.

    Args:
        cache_name: Name of cache file

    Returns:
        Cached data, or None if cache doesn't exist
    """
    if not ENABLE_CACHING:
        return None

    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")

    if not os.path.exists(cache_path):
        return None

    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    print(f"✓ Loaded cache: {cache_path}")
    return data


def cache_exists(cache_name: str) -> bool:
    """
    Check if cache file exists.

    Args:
        cache_name: Name of cache file

    Returns:
        True if cache exists, False otherwise
    """
    if not ENABLE_CACHING:
        return False

    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    return os.path.exists(cache_path)


def draw_frame_number(frame: np.ndarray, frame_num: int) -> np.ndarray:
    """
    Draw frame number on frame.

    Args:
        frame: Video frame
        frame_num: Frame number to display

    Returns:
        Frame with frame number drawn
    """
    frame_copy = frame.copy()
    cv2.putText(
        frame_copy,
        f"Frame: {frame_num}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    return frame_copy
