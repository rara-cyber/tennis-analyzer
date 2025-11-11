"""Video processing and caching utilities."""

import cv2
import pickle
import os
import sys
import time
import numpy as np
from typing import List, Dict, Any, Tuple


def read_video(video_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Read video file and extract frames.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (frames, metadata) where:
        - frames: List of video frames as numpy arrays
        - metadata: Dictionary with video properties (fps, width, height, frame_count)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    metadata = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': len(frames)
    }

    cap.release()
    return frames, metadata


def save_video(frames: List[np.ndarray], output_path: str, fps: int = 24) -> None:
    """
    Save frames as video file.

    Args:
        frames: List of video frames
        output_path: Path to save video
        fps: Frames per second (default: 24)
    """
    if not frames:
        raise ValueError("No frames to save")

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        out.write(frame)

    out.release()
    print(f"✓ Video saved to {output_path}")


def save_cache(data: Any, cache_path: str) -> None:
    """
    Save data to cache file using pickle.

    Args:
        data: Data to cache
        cache_path: Path to save cache file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Cache saved to {cache_path}")


def load_cache(cache_path: str) -> Any:
    """
    Load data from cache file.

    Args:
        cache_path: Path to cache file

    Returns:
        Cached data
    """
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    return data


def cache_exists(cache_path: str) -> bool:
    """
    Check if cache file exists.

    Args:
        cache_path: Path to cache file

    Returns:
        True if cache exists, False otherwise
    """
    return os.path.exists(cache_path)


def display_progress(current: int, total: int, task_name: str = "Progress", start_time: float = None) -> None:
    """
    Display progress bar in console.

    Args:
        current: Current iteration
        total: Total iterations
        task_name: Name of task being tracked
        start_time: Start time (from time.time())
    """
    progress = current / total
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)

    # Calculate time remaining
    time_str = ""
    if start_time:
        elapsed = time.time() - start_time
        if current > 0:
            eta = (elapsed / current) * (total - current)
            time_str = f" ETA: {eta:.1f}s"

    # Print progress bar
    sys.stdout.write(f'\r{task_name}: [{bar}] {current}/{total} ({progress*100:.1f}%){time_str}')
    sys.stdout.flush()

    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
