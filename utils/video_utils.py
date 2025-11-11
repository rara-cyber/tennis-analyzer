"""Video utilities for reading, writing, and caching video data."""

import cv2
import pickle
import os
import sys
import time
from typing import List, Dict, Tuple, Any
import numpy as np


def read_video(video_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Read video frames and metadata.

    Args:
        video_path: Path to input video file

    Returns:
        Tuple of (frames list, metadata dict)
        metadata contains: fps, width, height, total_frames
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video metadata
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames
    }

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    print(f"✓ Read {len(frames)} frames from {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}")

    return frames, metadata


def save_video(frames: List[np.ndarray], output_path: str, fps: int = 24) -> None:
    """
    Save frames to video file.

    Args:
        frames: List of video frames
        output_path: Path to output video file
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames to save")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        out.write(frame)

    out.release()

    print(f"✓ Saved {len(frames)} frames to {output_path}")


def cache_exists(cache_path: str) -> bool:
    """
    Check if cache file exists.

    Args:
        cache_path: Path to cache file

    Returns:
        True if cache exists, False otherwise
    """
    return os.path.exists(cache_path)


def load_cache(cache_path: str) -> Any:
    """
    Load cached data from pickle file.

    Args:
        cache_path: Path to cache file

    Returns:
        Cached data
    """
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    return data


def save_cache(data: Any, cache_path: str) -> None:
    """
    Save data to cache file.

    Args:
        data: Data to cache
        cache_path: Path to cache file
    """
    # Create directory if it doesn't exist
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Cached data saved to {cache_path}")


def display_progress(current: int, total: int, prefix: str = "Progress", start_time: float = None) -> None:
    """
    Display progress bar in console.

    Args:
        current: Current iteration
        total: Total iterations
        prefix: Prefix string
        start_time: Start time for ETA calculation
    """
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = 100 * (current / float(total))

    # Calculate ETA
    eta_str = ""
    if start_time and current > 0:
        elapsed = time.time() - start_time
        eta = elapsed * (total - current) / current
        eta_str = f" ETA: {eta:.1f}s"

    sys.stdout.write(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total}){eta_str}')
    sys.stdout.flush()

    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
