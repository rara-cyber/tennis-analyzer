# Sub-PRD 1: Video Processing Pipeline

**Module Name:** `video_processor`  
**Claude Code Instruction:** "Implement this complete video processing pipeline module following all specifications below."

---

## Module Overview

Create a Python module that handles all video input/output operations, frame extraction, buffering, and result caching for the tennis analysis system. This module provides the foundation for all other components.

## What You're Building

A video processing system that:
- Reads video files and extracts individual frames
- Saves processed videos with annotations
- Caches intermediate results to speed up development
- Manages frame numbering and metadata

## File Structure to Create

```
utils/
├── __init__.py
├── video_utils.py         # Main video I/O functions

constants/
└── __init__.py            # Video processing constants

tests/
└── test_video_utils.py    # Unit tests
```

## Requirements

### REQ-1: Video File Reading

**Function Signature:**
```python
def read_video(video_path: str) -> tuple[list[np.ndarray], dict]:
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
```

**Implementation Details:**
- Use OpenCV (`cv2.VideoCapture`) to open video
- Read frames in a loop until `ret` is False
- Store each frame in a Python list
- Extract metadata from `VideoCapture.get()` properties:
  - `cv2.CAP_PROP_FPS` for frame rate
  - `cv2.CAP_PROP_FRAME_WIDTH` for width
  - `cv2.CAP_PROP_FRAME_HEIGHT` for height
  - `cv2.CAP_PROP_FRAME_COUNT` for total frames
- Calculate duration as `total_frames / fps`
- Always call `cap.release()` when done

**Acceptance Criteria:**
- Supports MP4, AVI, MOV formats (H.264/H.265 codecs)
- Successfully reads 1080p videos up to 60 minutes
- Returns frames in RGB format (convert from BGR using `cv2.cvtColor`)
- Raises clear error messages for missing or corrupted files
- Metadata accuracy: FPS within ±0.1, dimensions exact

**Error Handling:**
```python
# Check file exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

# Check video opened successfully
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Unable to open video file. Format may be unsupported: {video_path}")

# Validate minimum resolution
if width < 1280 or height < 720:
    raise ValueError(f"Video resolution {width}x{height} is too low. Minimum: 1280x720")
```

### REQ-2: Video File Writing

**Function Signature:**
```python
def save_video(
    output_frames: list[np.ndarray], 
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
```

**Implementation Details:**
- Use `cv2.VideoWriter` to create video file
- Get frame dimensions from first frame: `height, width = output_frames[0].shape[:2]`
- Create FourCC code: `fourcc = cv2.VideoWriter_fourcc(*codec)`
- Write each frame using `out.write(frame)`
- Always call `out.release()` when done
- Create output directory if it doesn't exist: `os.makedirs(os.path.dirname(output_path), exist_ok=True)`

**Acceptance Criteria:**
- Creates valid MP4 files playable in standard video players
- Output video has same resolution as input frames
- Output video matches specified FPS (±0.5 fps tolerance)
- Automatically creates output directory if missing
- Validates all frames have same dimensions before writing

**Validation Code:**
```python
if len(output_frames) == 0:
    raise ValueError("Cannot save video: output_frames list is empty")

# Check all frames have same dimensions
first_shape = output_frames[0].shape
for i, frame in enumerate(output_frames):
    if frame.shape != first_shape:
        raise ValueError(f"Frame {i} has shape {frame.shape}, expected {first_shape}")
```

### REQ-3: Frame Caching System

**Function Signatures:**
```python
def save_cache(data: Any, cache_path: str) -> None:
    """Save detection results to pickle file for faster reprocessing."""

def load_cache(cache_path: str) -> Any:
    """Load cached detection results from pickle file."""
    
def cache_exists(cache_path: str) -> bool:
    """Check if cache file exists."""
```

**Implementation Details:**
- Use Python's `pickle` module for serialization
- Cache files stored in `tracker_stubs/` directory
- Naming convention: `{module_name}_detections.pkl` (e.g., `player_detections.pkl`)

**Save Cache Implementation:**
```python
import pickle
import os

def save_cache(data: Any, cache_path: str) -> None:
    # Create directory if needed
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Save data
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Cache saved: {cache_path}")
```

**Load Cache Implementation:**
```python
def load_cache(cache_path: str) -> Any:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Cache loaded: {cache_path}")
    return data

def cache_exists(cache_path: str) -> bool:
    return os.path.exists(cache_path)
```

**Acceptance Criteria:**
- Successfully serializes/deserializes Python lists and dictionaries
- Cache files saved to `tracker_stubs/` directory
- Provides clear error if cache file is corrupted
- Loading cache is >50x faster than reprocessing

### REQ-4: Progress Display

**Function Signature:**
```python
def display_progress(current: int, total: int, prefix: str = "", start_time: float = None) -> None:
    """
    Display progress bar in console.
    
    Args:
        current: Current iteration number (0 to total)
        total: Total number of iterations
        prefix: Text to display before progress bar
        start_time: Start time from time.time() for ETA calculation
    """
```

**Implementation Details:**
- Use `\r` (carriage return) to update same line in console
- Progress bar format: `[████████████████████] 100%`
- Show percentage and ETA

**Example Implementation:**
```python
import sys
import time

def display_progress(current: int, total: int, prefix: str = "", start_time: float = None) -> None:
    """Display progress bar with percentage and ETA."""
    bar_length = 40
    filled_length = int(bar_length * current / total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = 100 * (current / total)
    
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
```

**Acceptance Criteria:**
- Updates in real-time (refreshes every frame)
- Shows percentage accurate to 1 decimal place
- Displays ETA based on average processing speed
- Prints newline when progress reaches 100%

### REQ-5: Draw Frame Number Overlay

**Function Signature:**
```python
def draw_frame_number(
    frames: list[np.ndarray], 
    start_frame: int = 0,
    position: tuple = (10, 30),
    color: tuple = (255, 255, 255),
    font_scale: float = 1.0
) -> list[np.ndarray]:
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
```

**Implementation Details:**
- Use `cv2.putText()` to draw text
- Font: `cv2.FONT_HERSHEY_SIMPLEX`
- Format: "Frame: 1234"
- Draw on copy of frame to avoid modifying original

**Example Implementation:**
```python
def draw_frame_number(
    frames: list[np.ndarray], 
    start_frame: int = 0,
    position: tuple = (10, 30),
    color: tuple = (255, 255, 255),
    font_scale: float = 1.0
) -> list[np.ndarray]:
    
    output_frames = []
    for i, frame in enumerate(frames):
        # Create copy to avoid modifying original
        frame_copy = frame.copy()
        
        # Calculate frame number
        frame_num = start_frame + i
        text = f"Frame: {frame_num}"
        
        # Draw text
        cv2.putText(
            frame_copy,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,  # Thickness
            cv2.LINE_AA  # Anti-aliasing
        )
        
        output_frames.append(frame_copy)
    
    return output_frames
```

**Acceptance Criteria:**
- Frame numbers visible in top-left corner
- Text is readable (white on dark backgrounds)
- Frame numbering starts at 0 and increments by 1
- Does not modify original frames (returns copies)

## Constants to Define

**File:** `constants/__init__.py`

```python
# Video processing constants
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov']
MIN_VIDEO_RESOLUTION = (1280, 720)  # (width, height)
DEFAULT_FPS = 24.0
DEFAULT_CODEC = 'mp4v'  # For MP4 output

# Cache settings
CACHE_DIR = 'tracker_stubs'
ENABLE_CACHING = True

# Progress display
PROGRESS_BAR_LENGTH = 40
PROGRESS_UPDATE_FREQUENCY = 1  # Update every N frames
```

## Testing Requirements

**File:** `tests/test_video_utils.py`

Create unit tests for:

1. **Test Read Video:**
```python
def test_read_video_success():
    """Test reading valid video file."""
    # Create a small test video (10 frames, 640x480)
    # Read it back
    # Assert: len(frames) == 10
    # Assert: frames[0].shape == (480, 640, 3)
    # Assert: metadata['fps'] == 24.0

def test_read_video_missing_file():
    """Test reading non-existent video raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_video("nonexistent.mp4")

def test_read_video_low_resolution():
    """Test reading video below minimum resolution raises ValueError."""
    # Create 640x480 video (below 1280x720 minimum)
    with pytest.raises(ValueError, match="resolution.*too low"):
        read_video("low_res.mp4")
```

2. **Test Save Video:**
```python
def test_save_video_success():
    """Test saving frames to video file."""
    # Create dummy frames (10 frames, 640x480, random colors)
    # Save to temporary file
    # Assert: file exists
    # Assert: saved video can be read back with same frame count

def test_save_video_empty_frames():
    """Test saving empty frame list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        save_video([], "output.mp4")

def test_save_video_inconsistent_dimensions():
    """Test frames with different sizes raise ValueError."""
    frames = [
        np.zeros((480, 640, 3)),  # First frame 640x480
        np.zeros((720, 1280, 3))  # Second frame 1280x720
    ]
    with pytest.raises(ValueError, match="inconsistent"):
        save_video(frames, "output.mp4")
```

3. **Test Caching:**
```python
def test_cache_save_and_load():
    """Test saving and loading cache."""
    test_data = {"player_1": [[100, 200, 150, 250]], "player_2": [[300, 400, 350, 450]]}
    cache_path = "tracker_stubs/test_cache.pkl"
    
    # Save cache
    save_cache(test_data, cache_path)
    assert cache_exists(cache_path)
    
    # Load cache
    loaded_data = load_cache(cache_path)
    assert loaded_data == test_data
    
    # Cleanup
    os.remove(cache_path)
```

## Usage Example

**File:** `examples/video_processing_example.py`

```python
from utils.video_utils import read_video, save_video, draw_frame_number

# Read input video
print("Reading video...")
frames, metadata = read_video("input_videos/sample.mp4")
print(f"Loaded {len(frames)} frames at {metadata['fps']} fps")
print(f"Resolution: {metadata['width']}x{metadata['height']}")

# Draw frame numbers
print("Adding frame numbers...")
frames_with_numbers = draw_frame_number(frames)

# Save output video
print("Saving output video...")
save_video(
    frames_with_numbers, 
    "output_videos/sample_with_frames.mp4",
    fps=metadata['fps']
)

print("Done!")
```

## Dependencies to Install

Add to `requirements.txt`:
```
opencv-python==4.8.1.78
numpy==1.24.3
```

Install with:
```bash
pip install opencv-python numpy
```

## Validation Checklist

Before marking this module complete, verify:

- [ ] `read_video()` successfully reads MP4, AVI, MOV files
- [ ] `read_video()` raises errors for missing/corrupted files
- [ ] `save_video()` creates playable MP4 files
- [ ] `save_video()` automatically creates output directories
- [ ] Caching saves and loads data correctly
- [ ] Progress bar displays and updates in real-time
- [ ] Frame numbers appear in top-left corner of videos
- [ ] All unit tests pass
- [ ] Example script runs without errors

## Common Issues & Solutions

**Issue 1: "Unable to open video file"**
- **Cause:** Unsupported codec or corrupted file
- **Solution:** Re-encode video with `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

**Issue 2: "Memory error" when reading long videos**
- **Cause:** Loading all frames into memory at once
- **Solution:** Process video in 5-minute chunks (implement frame chunking in future)

**Issue 3: Saved video plays too fast/slow**
- **Cause:** FPS mismatch between input and output
- **Solution:** Always pass `metadata['fps']` to `save_video()`

**Issue 4: Progress bar doesn't update smoothly**
- **Cause:** Console buffering
- **Solution:** Use `sys.stdout.flush()` after each update

---

**End of Sub-PRD 1**