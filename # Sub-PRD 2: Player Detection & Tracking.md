# Sub-PRD 2: Player Detection & Tracking

**Module Name:** `player_tracker`  
**Claude Code Instruction:** "Implement this complete player detection and tracking module following all specifications below."

---

## Module Overview

Create a Python module that detects tennis players in video frames using YOLO, tracks them with consistent IDs throughout the video, and filters to identify only the two competing players (excluding audience, umpires).

## What You're Building

A player tracking system that:
- Detects all people in each frame using YOLOv8
- Assigns persistent IDs to the two players
- Tracks players across frames without ID swaps
- Extracts player positions (foot locations)
- Draws bounding boxes with player IDs on output videos

## File Structure to Create

```
trackers/
├── __init__.py
├── player_tracker.py      # Main player detection/tracking class

utils/
├── bbox_utils.py          # Bounding box helper functions

tests/
└── test_player_tracker.py # Unit tests
```

## Requirements

### REQ-1: PlayerTracker Class

**Class Definition:**
```python
class PlayerTracker:
    """Detects and tracks tennis players across video frames."""
    
    def __init__(self, model_path: str = 'yolov8x.pt'):
        """
        Initialize player tracker with YOLO model.
        
        Args:
            model_path: Path to YOLO model weights (default uses pre-trained YOLOv8x)
        """
        
    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Detect and track players in a single frame.
        
        Args:
            frame: Video frame (np.ndarray, shape (height, width, 3))
            
        Returns:
            Dictionary mapping player_id (int) to bounding box (list of 4 floats)
            Example: {1: [100.5, 200.3, 150.2, 300.8], 2: [400.1, 210.5, 450.3, 310.2]}
            Bounding box format: [x_min, y_min, x_max, y_max]
        """
        
    def detect_frames(
        self, 
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str = None
    ) -> list[dict]:
        """
        Detect and track players across all video frames.
        
        Args:
            frames: List of video frames
            read_from_stub: If True, load cached results instead of running detection
            stub_path: Path to cache file (e.g., 'tracker_stubs/player_detections.pkl')
            
        Returns:
            List of dictionaries (one per frame), each mapping player_id to bbox
        """
        
    def choose_and_filter_players(
        self, 
        court_keypoints: list[float],
        player_detections: list[dict]
    ) -> list[dict]:
        """
        Filter detections to only include the two competing players.
        
        Args:
            court_keypoints: List of 28 floats (14 x/y coordinate pairs)
            player_detections: List of detection dicts (may include >2 people per frame)
            
        Returns:
            Filtered list of detection dicts (exactly 2 players per frame)
        """
        
    def draw_bounding_boxes(
        self, 
        frames: list[np.ndarray],
        player_detections: list[dict]
    ) -> list[np.ndarray]:
        """
        Draw bounding boxes and player IDs on frames.
        
        Args:
            frames: List of video frames
            player_detections: List of detection dicts
            
        Returns:
            List of frames with bounding boxes drawn
        """
```

### REQ-2: YOLO Model Integration

**Implementation Details:**

```python
from ultralytics import YOLO
import time

def __init__(self, model_path: str = 'yolov8x.pt'):
    """Initialize YOLO model."""
    self.model = YOLO(model_path)
    print(f"✓ Player detector ({model_path}) loaded")
```

**Model Download:**
- YOLOv8x automatically downloads on first use (from Ultralytics)
- File size: ~131 MB
- Location: Cached in `~/.cache/torch/hub/`

**Acceptance Criteria:**
- Model loads successfully within 5 seconds
- Supports CUDA if GPU available (auto-detects)
- Falls back to CPU if no GPU

### REQ-3: Player Detection in Single Frame

**Implementation:**

```python
def detect_frame(self, frame: np.ndarray) -> dict:
    """Detect and track players in single frame."""
    
    # Run YOLO tracking (persist=True maintains IDs across frames)
    results = self.model.track(frame, persist=True, verbose=False)
    
    # Extract detections
    player_dict = {}
    
    for box in results[0].boxes:
        # Get class ID and name
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        
        # Only process "person" class (ID 0 in COCO dataset)
        if class_name == 'person':
            # Get tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
                
                # Get bounding box in xyxy format
                bbox = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
                
                player_dict[track_id] = bbox
    
    return player_dict
```

**Key YOLO Methods:**
- `model.track()` - Runs detection + tracking (maintains IDs)
- `results[0].boxes` - List of detected objects
- `box.cls` - Class ID tensor
- `box.id` - Tracking ID tensor (None if track=False)
- `box.xyxy` - Bounding box coordinates
- `results[0].names` - Dictionary mapping class IDs to names

**Acceptance Criteria:**
- Detects players with confidence ≥0.5 (YOLO default)
- Returns empty dict if no players detected
- Bounding boxes are within frame dimensions
- Tracking IDs remain consistent across frames (≥98% accuracy)

### REQ-4: Player Detection Across All Frames

**Implementation:**

```python
def detect_frames(
    self, 
    frames: list[np.ndarray],
    read_from_stub: bool = False,
    stub_path: str = None
) -> list[dict]:
    """Detect players in all frames with caching support."""
    
    # Import cache utilities
    from utils.video_utils import load_cache, save_cache, cache_exists
    from utils.video_utils import display_progress
    
    # Load from cache if requested
    if read_from_stub and stub_path and cache_exists(stub_path):
        print(f"Loading cached player detections from {stub_path}")
        return load_cache(stub_path)
    
    # Run detection on all frames
    print("Detecting players in all frames...")
    player_detections = []
    start_time = time.time()
    
    for i, frame in enumerate(frames):
        # Detect players in this frame
        detections = self.detect_frame(frame)
        player_detections.append(detections)
        
        # Display progress
        display_progress(i + 1, len(frames), "Progress", start_time)
    
    # Save to cache if path provided
    if stub_path:
        save_cache(player_detections, stub_path)
    
    return player_detections
```

**Acceptance Criteria:**
- Processes all frames without crashing
- Shows progress bar during processing
- Saves results to cache automatically
- Loads from cache in <1 second (vs. minutes for detection)

### REQ-5: Filter to Two Players Only

**Implementation:**

```python
def choose_and_filter_players(
    self, 
    court_keypoints: list[float],
    player_detections: list[dict]
) -> list[dict]:
    """Filter to the two players closest to the court."""
    
    # Get first frame with detections
    first_frame_detections = player_detections[0]
    
    # Choose 2 players closest to any court keypoint
    chosen_players = self._choose_players(court_keypoints, first_frame_detections)
    
    # Filter all frames to only include chosen players
    filtered_detections = []
    for frame_detections in player_detections:
        filtered_dict = {
            player_id: bbox 
            for player_id, bbox in frame_detections.items() 
            if player_id in chosen_players
        }
        filtered_detections.append(filtered_dict)
    
    return filtered_detections

def _choose_players(self, court_keypoints: list[float], player_dict: dict) -> list[int]:
    """Choose 2 players closest to court."""
    
    from utils.bbox_utils import get_foot_position, measure_distance
    
    distances = []
    
    for track_id, bbox in player_dict.items():
        # Get player foot position (center-x, bottom-y of bbox)
        player_foot = get_foot_position(bbox)
        
        # Calculate minimum distance to any court keypoint
        min_distance = float('inf')
        for i in range(0, len(court_keypoints), 2):
            keypoint_x = court_keypoints[i]
            keypoint_y = court_keypoints[i + 1]
            distance = measure_distance(player_foot, (keypoint_x, keypoint_y))
            min_distance = min(min_distance, distance)
        
        distances.append((track_id, min_distance))
    
    # Sort by distance and take 2 closest
    distances.sort(key=lambda x: x[1])
    chosen_players = [distances[0][0], distances[1][0]]
    
    return chosen_players
```

**Acceptance Criteria:**
- Correctly identifies 2 players on court (not audience/umpires)
- Uses first frame to determine which players to track
- Maintains same player IDs throughout video
- Handles edge case: only 1 player visible in first frame (raises warning)

### REQ-6: Draw Bounding Boxes on Frames

**Implementation:**

```python
import cv2

def draw_bounding_boxes(
    self, 
    frames: list[np.ndarray],
    player_detections: list[dict]
) -> list[np.ndarray]:
    """Draw bounding boxes with player IDs on all frames."""
    
    output_frames = []
    
    for frame, player_dict in zip(frames, player_detections):
        # Create copy to avoid modifying original
        frame_copy = frame.copy()
        
        for track_id, bbox in player_dict.items():
            # Extract coordinates
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw rectangle (red color in BGR: 0, 0, 255)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw player ID label
            label = f"Player {track_id}"
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
        
        output_frames.append(frame_copy)
    
    return output_frames
```

**Acceptance Criteria:**
- Red bounding boxes visible on all players
- Player IDs displayed above bounding boxes
- Text is readable (appropriate size and color)
- Does not modify original frames

## Bounding Box Utilities

**File:** `utils/bbox_utils.py`

```python
import numpy as np

def get_center_of_bbox(bbox: list[float]) -> tuple[int, int]:
    """
    Calculate center point of bounding box.
    
    Args:
        bbox: [x_min, y_min, x_max, y_max]
        
    Returns:
        (center_x, center_y) as integers
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def get_foot_position(bbox: list[float]) -> tuple[int, int]:
    """
    Calculate foot position (ground contact point) of player.
    
    Args:
        bbox: [x_min, y_min, x_max, y_max]
        
    Returns:
        (center_x, bottom_y) as integers
    """
    x1, y1, x2, y2 = bbox
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)  # Bottom of bounding box
    return foot_x, foot_y

def get_bbox_width(bbox: list[float]) -> int:
    """Calculate width of bounding box."""
    x1, _, x2, _ = bbox
    return int(x2 - x1)

def get_bbox_height(bbox: list[float]) -> int:
    """Calculate height of bounding box."""
    _, y1, _, y2 = bbox
    return int(y2 - y1)

def measure_distance(point1: tuple, point2: tuple) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: (x1, y1)
        point2: (x2, y2)
        
    Returns:
        Distance as float
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def measure_xy_distance(point1: tuple, point2: tuple) -> tuple[float, float]:
    """
    Calculate x and y distances separately.
    
    Args:
        point1: (x1, y1)
        point2: (x2, y2)
        
    Returns:
        (distance_x, distance_y) as floats
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1), abs(y2 - y1)
```

**Update `utils/__init__.py`:**
```python
from .bbox_utils import (
    get_center_of_bbox,
    get_foot_position,
    get_bbox_width,
    get_bbox_height,
    measure_distance,
    measure_xy_distance
)
```

**Acceptance Criteria for Utils:**
- All functions return integer coordinates (not floats)
- Distance calculations use standard Euclidean formula
- Handle edge cases (e.g., zero-width bounding boxes)

## Expose PlayerTracker Class

**File:** `trackers/__init__.py`

```python
from .player_tracker import PlayerTracker

__all__ = ['PlayerTracker']
```

## Testing Requirements

**File:** `tests/test_player_tracker.py`

```python
import pytest
import numpy as np
from trackers.player_tracker import PlayerTracker
from utils.bbox_utils import get_center_of_bbox, measure_distance, get_foot_position

def test_player_tracker_initialization():
    """Test tracker initializes with YOLO model."""
    tracker = PlayerTracker()
    assert tracker.model is not None

def test_detect_frame_with_players():
    """Test detecting players in frame with 2 people."""
    tracker = PlayerTracker()
    
    # Create dummy frame (black image)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Run detection (may return empty dict on black image)
    detections = tracker.detect_frame(frame)
    
    # Check return type
    assert isinstance(detections, dict)

def test_bbox_center_calculation():
    """Test bounding box center calculation."""
    bbox = [100, 200, 200, 400]  # x1, y1, x2, y2
    center = get_center_of_bbox(bbox)
    assert center == (150, 300)

def test_bbox_foot_position():
    """Test foot position calculation."""
    bbox = [100, 200, 200, 400]
    foot = get_foot_position(bbox)
    assert foot == (150, 400)  # center-x, bottom-y

def test_distance_measurement():
    """Test Euclidean distance calculation."""
    point1 = (0, 0)
    point2 = (3, 4)
    distance = measure_distance(point1, point2)
    assert distance == 5.0  # 3-4-5 triangle

def test_filter_to_two_players():
    """Test filtering detections to 2 closest players."""
    tracker = PlayerTracker()
    
    # Mock detections: 4 people detected
    first_frame = {
        1: [100, 200, 150, 300],  # Player near court
        2: [900, 100, 950, 200],  # Player near court
        3: [50, 50, 100, 150],    # Audience (far from court)
        4: [1200, 50, 1250, 150]  # Audience (far from court)
    }
    
    # Mock court keypoints (center of court)
    court_keypoints = [640, 360] * 14  # 14 points at center
    
    # Choose 2 players
    chosen = tracker._choose_players(court_keypoints, first_frame)
    
    # Should choose IDs 1 and 2 (closest to center)
    assert len(chosen) == 2
    assert 1 in chosen
    assert 2 in chosen
```

## Usage Example

**File:** `examples/player_tracking_example.py`

```python
from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker

# Read video
print("Reading video...")
frames, metadata = read_video("input_videos/sample.mp4")

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
print(f"First frame detections: {player_detections[0]}")
# Output: {1: [123.4, 234.5, 178.9, 345.6], 2: [567.8, 234.1, 623.4, 345.2]}

# Draw bounding boxes
print("Drawing bounding boxes...")
output_frames = tracker.draw_bounding_boxes(frames, player_detections)

# Save output video
print("Saving video...")
save_video(output_frames, "output_videos/players_detected.mp4", fps=metadata['fps'])

print("✓ Done!")
```

## Dependencies

Add to `requirements.txt`:
```
ultralytics==8.0.196
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.1.78
numpy==1.24.3
```

Install with:
```bash
pip install ultralytics torch torchvision opencv-python numpy
```

## Validation Checklist

Before marking this module complete:

- [ ] YOLOv8 model downloads and loads automatically
- [ ] `detect_frame()` detects people in test images
- [ ] Tracking IDs persist across frames (no swaps)
- [ ] `choose_and_filter_players()` selects correct 2 players
- [ ] Bounding boxes drawn correctly on output videos
- [ ] Caching saves/loads detection results
- [ ] All utility functions work correctly
- [ ] Unit tests pass
- [ ] Example script runs end-to-end

## Common Issues & Solutions

**Issue 1: Tracking IDs swap between players**
- **Cause:** Players crossing paths or occlusions
- **Solution:** ByteTrack (YOLO's default tracker) handles this. Ensure `persist=True` in `model.track()`

**Issue 2: Too many people detected (audience, umpires)**
- **Cause:** YOLO detects all people, not just players
- **Solution:** Use `choose_and_filter_players()` to select 2 closest to court

**Issue 3: No players detected in some frames**
- **Cause:** Occlusion, player out of frame, low confidence
- **Solution:** Normal behavior. Dictionary will be empty for those frames

**Issue 4: YOLO model download fails**
- **Cause:** Network issues or cache directory permissions
- **Solution:** Manually download from Ultralytics GitHub and specify path

**Issue 5: GPU not being used**
- **Cause:** PyTorch not detecting CUDA
- **Solution:** Check with `torch.cuda.is_available()`. Reinstall torch with CUDA support

---

**End of Sub-PRD 2**