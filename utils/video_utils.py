"""
Video processing utilities for tennis analysis system.

This module handles all video input/output operations, frame extraction,
buffering, and result caching.
"""

import os
import sys
import time
import pickle
from typing import Any, Tuple, List

import cv2
import numpy as np

from constants import MIN_VIDEO_RESOLUTION


def read_video(video_path: str) -> Tuple[List[np.ndarray], dict]:
    """
    Read video file and extract all frames.

    Args:
        video_path: Absolute or relative path to video file

    Returns:
        tuple containing:
        - List of frames (each frame is np.ndarray with shape (height, width, 3))
        - Metadata dict with keys: 'fps', 'width', 'height', 'total_frames', 'duration'

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is unsupported or file is corrupted
    """
    # Check file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file. Format may be unsupported: {video_path}")

    # Extract metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validate minimum resolution
    if width < MIN_VIDEO_RESOLUTION[0] or height < MIN_VIDEO_RESOLUTION[1]:
        cap.release()
        raise ValueError(
            f"Video resolution {width}x{height} is too low. "
            f"Minimum: {MIN_VIDEO_RESOLUTION[0]}x{MIN_VIDEO_RESOLUTION[1]}"
        )

    # Calculate duration
    duration = total_frames / fps if fps > 0 else 0

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Release video capture
    cap.release()

    # Create metadata dictionary
    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'duration': duration
    }

    return frames, metadata


def save_video(
    output_frames: List[np.ndarray],
    output_path: str,
    fps: float = 24.0,
    codec: str = 'mp4v'
) -> None:
    """
    Save list of frames as video file.

    Args:
        output_frames: List of frames (np.ndarray, shape (height, width, 3))
        output_path: Path where video will be saved (should end in .mp4 or .avi)
        fps: Frames per second for output video
        codec: FourCC codec code ('mp4v' for MP4, 'XVID' for AVI)

    Raises:
        ValueError: If output_frames is empty or frames have inconsistent dimensions
        IOError: If unable to write to output_path
    """
    # Validate input
    if len(output_frames) == 0:
        raise ValueError("Cannot save video: output_frames list is empty")

    # Check all frames have same dimensions
    first_shape = output_frames[0].shape
    for i, frame in enumerate(output_frames):
        if frame.shape != first_shape:
            raise ValueError(
                f"Frame {i} has shape {frame.shape}, expected {first_shape}. "
                f"All frames must have consistent dimensions."
            )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's a directory component
        os.makedirs(output_dir, exist_ok=True)

    # Get frame dimensions
    height, width = output_frames[0].shape[:2]

    # Create FourCC code
    fourcc = cv2.VideoWriter_fourcc(*codec)

    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise IOError(f"Unable to write to output path: {output_path}")

    # Write each frame (convert RGB back to BGR for OpenCV)
    for frame in output_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release video writer
    out.release()


def save_cache(data: Any, cache_path: str) -> None:
    """
    Save detection results to pickle file for faster reprocessing.

    Args:
        data: Data to cache (must be picklable)
        cache_path: Path to save cache file
    """
    # Create directory if needed
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Save data
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Cache saved: {cache_path}")


def load_cache(cache_path: str) -> Any:
    """
    Load cached detection results from pickle file.

    Args:
        cache_path: Path to cache file

    Returns:
        Cached data

    Raises:
        FileNotFoundError: If cache file doesn't exist
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    print(f"✓ Cache loaded: {cache_path}")
    return data


def cache_exists(cache_path: str) -> bool:
    """
    Check if cache file exists.

    Args:
        cache_path: Path to cache file

    Returns:
        True if cache file exists, False otherwise
    """
    return os.path.exists(cache_path)


def display_progress(
    current: int,
    total: int,
    prefix: str = "",
    start_time: float = None
) -> None:
    """
    Display progress bar in console.

    Args:
        current: Current iteration number (0 to total)
        total: Total number of iterations
        prefix: Text to display before progress bar
        start_time: Start time from time.time() for ETA calculation
    """
    from constants import PROGRESS_BAR_LENGTH

    bar_length = PROGRESS_BAR_LENGTH
    filled_length = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = 100 * (current / total) if total > 0 else 0

    # Calculate ETA
    eta_str = ""
    if start_time and current > 0:
        elapsed = time.time() - start_time
        eta_seconds = (elapsed / current) * (total - current)
        eta_str = f" | ETA: {int(eta_seconds)}s"

    sys.stdout.write(f'\r{prefix} [{bar}] {percent:.1f}%{eta_str}')
    sys.stdout.flush()

    if current == total:
        print()  # New line when complete


def draw_frame_number(
    frames: List[np.ndarray],
    start_frame: int = 0,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 1.0
) -> List[np.ndarray]:
    """
    Draw frame numbers on all frames.

    Args:
        frames: List of video frames
        start_frame: Starting frame number (default 0)
        position: (x, y) pixel position for text
        color: RGB color tuple (default white)
        font_scale: Text size multiplier

    Returns:
        List of frames with frame numbers drawn
    """
    output_frames = []

    for i, frame in enumerate(frames):
        # Create copy to avoid modifying original
        frame_copy = frame.copy()

        # Calculate frame number
        frame_num = start_frame + i
        text = f"Frame: {frame_num}"

        # Draw text (convert RGB to BGR for cv2.putText, then back)
        frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame_bgr,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color[::-1],  # Convert RGB to BGR
            2,  # Thickness
            cv2.LINE_AA  # Anti-aliasing
        )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        output_frames.append(frame_rgb)

    return output_frames
