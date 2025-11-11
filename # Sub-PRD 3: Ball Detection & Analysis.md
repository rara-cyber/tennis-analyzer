# Sub-PRD 3: Ball Detection & Analysis

**Module Name:** `ball_tracker`  
**Claude Code Instruction:** "Implement this complete ball detection, interpolation, and shot analysis module following all specifications below."

---

## Module Overview

Create a Python module that detects tennis balls in video frames using a fine-tuned YOLO model, interpolates missing detections to create smooth trajectories, and identifies shot events (when players hit the ball) and bounce events (when ball hits ground).

## What You're Building

A ball analysis system that:
- Detects tennis balls using YOLOv5/YOLOv8 fine-tuned on tennis ball dataset
- Interpolates missing detections (ball moves too fast to detect every frame)
- Identifies when ball bounces (touches ground)
- Identifies when players hit the ball (shot events)
- Draws ball bounding boxes on output videos

## File Structure to Create

```
trackers/
├── ball_tracker.py        # Ball detection, interpolation, shot detection

tests/
└── test_ball_tracker.py   # Unit tests

models/
└── yolov5_tennis_ball.pt  # Fine-tuned model (manual download)
```

## Requirements

### REQ-1: BallTracker Class

**Class Definition:**

```python
class BallTracker:
    """Detects and tracks tennis ball across video frames."""
    
    def __init__(self, model_path: str):
        """
        Initialize ball tracker with fine-tuned YOLO model.
        
        Args:
            model_path: Path to fine-tuned YOLO model (e.g., 'models/yolov5_tennis_ball.pt')
        """
        
    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Detect ball in a single frame.
        
        Args:
            frame: Video frame (np.ndarray)
            
        Returns:
            Dictionary with single key-value pair: {1: [x_min, y_min, x_max, y_max]}
            Returns empty dict {} if no ball detected
        """
        
    def detect_frames(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str = None
    ) -> list[dict]:
        """
        Detect ball in all frames with caching.
        
        Args:
            frames: List of video frames
            read_from_stub: Load from cache if True
            stub_path: Path to cache file
            
        Returns:
            List of dictionaries (one per frame)
        """
        
    def interpolate_ball_positions(self, ball_detections: list[dict]) -> list[dict]:
        """
        Fill in missing ball detections using linear interpolation.
        
        Args:
            ball_detections: List of detection dicts (may have empty dicts for missing frames)
            
        Returns:
            List of detection dicts with gaps filled in
        """
        
    def get_ball_shot_frames(self, ball_detections: list[dict]) -> list[int]:
        """
        Detect frames where player hit the ball (shot events).
        
        Args:
            ball_detections: List of ball detection dicts (with interpolation)
            
        Returns:
            List of frame numbers where shots occurred
            Example: [11, 58, 95, 131, 182]
        """
        
    def draw_bounding_boxes(
        self,
        frames: list[np.ndarray],
        ball_detections: list[dict]
    ) -> list[np.ndarray]:
        """
        Draw ball bounding boxes on frames (yellow color).
        
        Args:
            frames: List of video frames
            ball_detections: List of detection dicts
            
        Returns:
            List of frames with ball bounding boxes drawn
        """
```

### REQ-2: Ball Detection in Single Frame

**Implementation:**

```python
from ultralytics import YOLO
import time

def __init__(self, model_path: str):
    """Initialize ball detector with fine-tuned YOLO model."""
    self.model = YOLO(model_path)
    print(f"✓ Ball detector ({model_path}) loaded")

def detect_frame(self, frame: np.ndarray) -> dict:
    """Detect ball in single frame."""
    
    # Run YOLO prediction (no tracking, just detection)
    # Lower confidence threshold because balls are small and fast
    results = self.model.predict(frame, conf=0.15, verbose=False)
    
    # Extract ball detection
    ball_dict = {}
    
    if len(results[0].boxes) > 0:
        # Take first detection (highest confidence)
        box = results[0].boxes[0]
        bbox = box.xyxy[0].tolist()
        
        # Always use ID 1 for ball (only one ball on court)
        ball_dict[1] = bbox
    
    return ball_dict
```

**Key Differences from Player Detection:**
- Uses `model.predict()` not `model.track()` (no need to track, only 1 ball)
- Lower confidence threshold (0.15 vs. 0.5) because balls are small and fast
- Always assigns ID 1 to detected ball

**Acceptance Criteria:**
- Detects ball in ≥75% of frames (raw, before interpolation)
- Confidence threshold = 0.15 (adjustable)
- Returns empty dict if no ball detected
- Takes only highest confidence detection (ignores false positives)

### REQ-3: Ball Detection Across All Frames

**Implementation:**

```python
def detect_frames(
    self,
    frames: list[np.ndarray],
    read_from_stub: bool = False,
    stub_path: str = None
) -> list[dict]:
    """Detect ball in all frames with caching."""
    
    # Import cache utilities
    from utils.video_utils import load_cache, save_cache, cache_exists
    from utils.video_utils import display_progress
    
    # Load from cache if requested
    if read_from_stub and stub_path and cache_exists(stub_path):
        print(f"Loading cached ball detections from {stub_path}")
        return load_cache(stub_path)
    
    # Run detection on all frames
    print("Detecting ball in all frames...")
    ball_detections = []
    start_time = time.time()
    
    for i, frame in enumerate(frames):
        detections = self.detect_frame(frame)
        ball_detections.append(detections)
        
        # Display progress
        display_progress(i + 1, len(frames), "Progress", start_time)
    
    # Save to cache
    if stub_path:
        save_cache(ball_detections, stub_path)
    
    # Calculate detection rate
    frames_with_ball = sum(1 for d in ball_detections if len(d) > 0)
    detection_rate = frames_with_ball / len(ball_detections) * 100
    print(f"\n✓ Ball detected in {frames_with_ball}/{len(ball_detections)} frames ({detection_rate:.1f}%)")
    
    return ball_detections
```

**Acceptance Criteria:**
- Processes all frames without crashing
- Shows progress bar
- Reports detection rate (percentage of frames with ball)
- Saves results to cache

### REQ-4: Ball Position Interpolation

**Implementation:**

```python
import pandas as pd

def interpolate_ball_positions(self, ball_detections: list[dict]) -> list[dict]:
    """Interpolate missing ball positions using pandas."""
    
    # Convert to list of bounding boxes (None for missing frames)
    ball_positions = [d.get(1, None) for d in ball_detections]
    
    # Create DataFrame
    df_ball_positions = pd.DataFrame(
        ball_positions,
        columns=['x1', 'y1', 'x2', 'y2']
    )
    
    # Interpolate missing values linearly
    df_ball_positions = df_ball_positions.interpolate()
    
    # Back-fill start (if first frames have no detection)
    df_ball_positions = df_ball_positions.bfill()
    
    # Convert back to list of dicts
    ball_detections_interpolated = [
        {1: row.tolist()} 
        for _, row in df_ball_positions.iterrows()
    ]
    
    # Report interpolation results
    original_detections = sum(1 for d in ball_detections if len(d) > 0)
    final_detections = sum(1 for d in ball_detections_interpolated if len(d) > 0)
    print(f"✓ Interpolated {final_detections - original_detections} missing ball positions")
    print(f"✓ Final ball coverage: {final_detections}/{len(ball_detections)} frames ({final_detections/len(ball_detections)*100:.1f}%)")
    
    return ball_detections_interpolated
```

**Interpolation Logic:**
1. Convert list of dicts to pandas DataFrame (easier to interpolate)
2. Use `.interpolate()` to fill gaps linearly
3. Use `.bfill()` (back-fill) to handle missing frames at start of video
4. Convert back to original dict format

**Acceptance Criteria:**
- Fills gaps ≤5 frames using linear interpolation
- Uses back-fill for missing detections at video start
- Final detection rate ≥95% (after interpolation)
- Prints statistics showing improvement

**Example:**
```
Frame 0: {1: [100, 200, 120, 220]}  ← Detected
Frame 1: {}                          ← Missing (will interpolate)
Frame 2: {}                          ← Missing (will interpolate)
Frame 3: {1: [140, 240, 160, 260]}  ← Detected

After interpolation:
Frame 1: {1: [110, 210, 130, 230]}  ← Interpolated
Frame 2: {1: [120, 220, 140, 240]}  ← Interpolated
```

### REQ-5: Shot Detection

**Implementation:**

```python
def get_ball_shot_frames(self, ball_detections: list[dict]) -> list[int]:
    """
    Detect frames where player hit the ball (shot events).
    
    Args:
        ball_detections: List of ball detection dicts (with interpolation)
        
    Returns:
        List of frame numbers where shots occurred
        Example: [11, 58, 95, 131, 182]
    """
    
    # Convert to DataFrame for easier analysis
    ball_positions = [d.get(1, [0, 0, 0, 0]) for d in ball_detections]
    df_ball = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    
    # Calculate ball center y-position
    df_ball['mid_y'] = (df_ball['y1'] + df_ball['y2']) / 2
    
    # Smooth trajectory with rolling mean (reduces noise)
    df_ball['mid_y_rolling'] = df_ball['mid_y'].rolling(
        window=5, 
        min_periods=1, 
        center=False
    ).mean()
    
    # Calculate change in y-position (velocity)
    df_ball['delta_y'] = df_ball['mid_y_rolling'].diff()
    
    # Initialize shot detection
    shot_frames = []
    minimum_change_frames_for_hit = 25  # Ball must change direction for 25 frames
    
    for i in range(len(df_ball) - int(minimum_change_frames_for_hit * 1.2)):
        # Detect negative to positive change (ball going down, then up = hit)
        negative_position_change = df_ball['delta_y'].iloc[i] > 0 and df_ball['delta_y'].iloc[i + 1] < 0
        positive_position_change = df_ball['delta_y'].iloc[i] < 0 and df_ball['delta_y'].iloc[i + 1] > 0
        
        if negative_position_change or positive_position_change:
            # Count how many frames the change persists
            change_count = 0
            for change_frame in range(i + 1, i + minimum_change_frames_for_hit):
                negative_change = df_ball['delta_y'].iloc[change_frame] < 0 and negative_position_change
                positive_change = df_ball['delta_y'].iloc[change_frame] > 0 and positive_position_change
                
                if negative_change or positive_change:
                    change_count += 1
            
            # If change persists for minimum frames, it's a shot
            if change_count >= minimum_change_frames_for_hit - 1:
                shot_frames.append(i)
    
    print(f"✓ Detected {len(shot_frames)} shots at frames: {shot_frames}")
    return shot_frames
```

**Shot Detection Logic:**
1. Calculate ball's y-position (vertical) over time
2. Smooth trajectory with rolling mean (reduces detection noise)
3. Calculate velocity (change in y-position between frames)
4. Look for velocity reversals:
   - Ball going down → Ball going up = Shot (player hit ball upward)
   - Ball going up → Ball going down = Shot (player hit ball downward)
5. Require change to persist for 25 frames (confirms real shot, not noise)

**Acceptance Criteria:**
- Detects shots with ≥90% accuracy
- Returns frame numbers in ascending order
- Does not detect false shots (noise, camera shake)
- Handles edge case: shot at very start or end of video

### REQ-6: Draw Ball Bounding Boxes

**Implementation:**

```python
import cv2

def draw_bounding_boxes(
    self,
    frames: list[np.ndarray],
    ball_detections: list[dict]
) -> list[np.ndarray]:
    """Draw ball bounding boxes (yellow) on all frames."""
    
    output_frames = []
    
    for frame, ball_dict in zip(frames, ball_detections):
        frame_copy = frame.copy()
        
        if 1 in ball_dict:  # Ball detected in this frame
            bbox = ball_dict[1]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle (yellow color in BGR: 0, 255, 255)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw label
            cv2.putText(
                frame_copy,
                "Ball",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )
        
        output_frames.append(frame_copy)
    
    return output_frames
```

**Acceptance Criteria:**
- Yellow bounding boxes visible on ball in all frames
- Label "Ball" displayed above bounding box
- Does not draw anything if ball not detected in frame

## Expose BallTracker Class

**File:** `trackers/__init__.py`

```python
from .player_tracker import PlayerTracker
from .ball_tracker import BallTracker

__all__ = ['PlayerTracker', 'BallTracker']
```

## Testing Requirements

**File:** `tests/test_ball_tracker.py`

```python
import pytest
import numpy as np
from trackers.ball_tracker import BallTracker

def test_ball_tracker_initialization():
    """Test tracker initializes with model."""
    tracker = BallTracker(model_path='models/yolov5_tennis_ball.pt')
    assert tracker.model is not None

def test_detect_frame_with_ball():
    """Test detecting ball in frame."""
    tracker = BallTracker(model_path='models/yolov5_tennis_ball.pt')
    
    # Create dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Run detection
    detections = tracker.detect_frame(frame)
    
    # Check return type
    assert isinstance(detections, dict)

def test_interpolation_fills_gaps():
    """Test interpolation fills missing detections."""
    tracker = BallTracker(model_path='models/yolov5_tennis_ball.pt')
    
    # Mock detections with gaps
    ball_detections = [
        {1: [100, 200, 120, 220]},  # Frame 0: detected
        {},                          # Frame 1: missing
        {},                          # Frame 2: missing
        {1: [140, 240, 160, 260]}   # Frame 3: detected
    ]
    
    # Interpolate
    interpolated = tracker.interpolate_ball_positions(ball_detections)
    
    # Check all frames now have detections
    assert all(len(d) > 0 for d in interpolated)
    
    # Check frame 1 is approximately halfway between frame 0 and 3
    frame1_bbox = interpolated[1][1]
    assert 100 < frame1_bbox[0] < 140  # x1 between 100 and 140

def test_shot_detection():
    """Test shot frame detection."""
    tracker = BallTracker(model_path='models/yolov5_tennis_ball.pt')
    
    # Create mock ball trajectory (goes down, then up = shot at frame 50)
    ball_detections = []
    
    # Frames 0-49: Ball descending (y increases)
    for i in range(50):
        y_pos = 100 + i * 2
        ball_detections.append({1: [640, y_pos, 660, y_pos + 20]})
    
    # Frames 50-99: Ball ascending (y decreases)
    for i in range(50):
        y_pos = 200 - i * 2
        ball_detections.append({1: [640, y_pos, 660, y_pos + 20]})
    
    # Detect shots
    shot_frames = tracker.get_ball_shot_frames(ball_detections)
    
    # Should detect shot around frame 50
    assert len(shot_frames) > 0
    assert 45 <= shot_frames[0] <= 55  # Within 5 frames of actual shot

def test_empty_detections():
    """Test handling of all empty detections."""
    tracker = BallTracker(model_path='models/yolov5_tennis_ball.pt')
    
    # All frames have no detections
    ball_detections = [{} for _ in range(100)]
    
    # Should not crash
    interpolated = tracker.interpolate_ball_positions(ball_detections)
    assert len(interpolated) == 100
```

## Usage Example

**File:** `examples/ball_tracking_example.py`

```python
from utils.video_utils import read_video, save_video
from trackers.ball_tracker import BallTracker

# Read video
print("Reading video...")
frames, metadata = read_video("input_videos/sample.mp4")

# Initialize tracker
tracker = BallTracker(model_path='models/yolov5_tennis_ball.pt')

# Detect ball in all frames
print("\nDetecting ball...")
ball_detections = tracker.detect_frames(
    frames,
    read_from_stub=False,  # Set to True after first run
    stub_path='tracker_stubs/ball_detections.pkl'
)

# Interpolate missing detections
print("\nInterpolating ball positions...")
ball_detections = tracker.interpolate_ball_positions(ball_detections)

# Detect shots
print("\nDetecting shots...")
shot_frames = tracker.get_ball_shot_frames(ball_detections)

# Draw bounding boxes
print("\nDrawing ball bounding boxes...")
output_frames = tracker.draw_bounding_boxes(frames, ball_detections)

# Save output
print("\nSaving video...")
save_video(output_frames, "output_videos/ball_tracked.mp4", fps=metadata['fps'])

print("\n✓ Done!")
```

## Dependencies

Add to `requirements.txt`:
```
ultralytics==8.0.196
pandas==2.0.3
numpy==1.24.3
opencv-python==4.8.1.78
torch==2.0.1
```

## Model Requirements

**Fine-Tuned Ball Detection Model:**

You need a YOLO model trained specifically on tennis balls. Options:

**Option 1: Download Pre-trained Model (Recommended)**

Create a download script: `download_models.py`

```python
import urllib.request
import os

def download_ball_model():
    """Download fine-tuned tennis ball detection model."""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Model URL (replace with actual URL when available)
    model_url = "https://github.com/your-repo/releases/download/v1.0/yolov5_tennis_ball.pt"
    model_path = "models/yolov5_tennis_ball.pt"
    
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return
    
    print(f"Downloading ball detection model...")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"✓ Model downloaded: {model_path}")

if __name__ == "__main__":
    download_ball_model()
```

**Option 2: Use YOLOv8n Pre-trained (Fallback)**

If fine-tuned model unavailable, use generic sports ball detection:

```python
# In ball_tracker.py __init__
def __init__(self, model_path: str = 'yolov8n.pt'):
    """Initialize with generic YOLO model as fallback."""
    self.model = YOLO(model_path)
    self.use_generic = (model_path == 'yolov8n.pt')
    
    if self.use_generic:
        print("⚠ Using generic YOLOv8n model. Detection accuracy may be lower.")
        print("  Download fine-tuned model for better results.")
```

**Option 3: Train Your Own (Advanced)**

See training guide in repository: `/training/README.md`

## Validation Checklist

Before marking this module complete:

- [ ] Ball detection model loads successfully
- [ ] `detect_frame()` detects balls in test images
- [ ] Detection rate ≥75% before interpolation
- [ ] Interpolation increases detection rate to ≥95%
- [ ] Shot detection finds correct frames (±5 frame tolerance)
- [ ] Yellow bounding boxes drawn correctly
- [ ] Caching works for detection results
- [ ] All unit tests pass
- [ ] Example script runs end-to-end

## Common Issues & Solutions

**Issue 1: Low ball detection rate (<70%)**
- **Cause:** Ball too small, moves too fast, or lighting poor
- **Solution:** Lower confidence threshold to 0.10 (more false positives, but catches more balls)

**Issue 2: Too many false positives (detecting white lines, player clothing)**
- **Cause:** Confidence threshold too low
- **Solution:** Increase to 0.20-0.25, or add size filter (ball bbox should be 15-30 pixels)

**Issue 3: Interpolation creates unrealistic trajectories**
- **Cause:** Large gaps (>5 frames) between detections
- **Solution:** Gaps >5 frames are treated as separate events (no interpolation across them)

**Issue 4: Shot detection misses obvious shots**
- **Cause:** `minimum_change_frames_for_hit` is too high (25)
- **Solution:** Reduce to 15-20 frames for faster shots

**Issue 5: Shot detection finds false shots (ball bounces)**
- **Cause:** Bounces also reverse direction
- **Solution:** Add player proximity check in future version (only count as shot if player within 2m)

**Issue 6: ModuleNotFoundError for pandas**
- **Cause:** Pandas not installed
- **Solution:** `pip install pandas==2.0.3`

**Issue 7: Model file not found**
- **Cause:** Model not downloaded to `models/` directory
- **Solution:** Run `python download_models.py` or manually download model

---

**End of Sub-PRD 3**