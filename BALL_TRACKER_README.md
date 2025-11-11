# Ball Tracker Module

Complete implementation of the tennis ball detection, interpolation, and shot analysis module.

## Overview

The Ball Tracker module provides comprehensive tennis ball tracking capabilities:
- **Ball Detection**: Uses fine-tuned YOLO model to detect tennis balls in video frames
- **Interpolation**: Fills gaps in detection using linear interpolation
- **Shot Detection**: Identifies frames where players hit the ball
- **Visualization**: Draws bounding boxes on output videos

## Module Structure

```
trackers/
├── __init__.py           # Module exports
└── ball_tracker.py       # BallTracker class

utils/
├── __init__.py           # Utility exports
└── video_utils.py        # Video I/O and caching utilities

tests/
├── __init__.py
└── test_ball_tracker.py  # Comprehensive unit tests

examples/
└── ball_tracking_example.py  # Usage example

models/
└── .gitkeep              # Place YOLO models here

tracker_stubs/
└── .gitkeep              # Cached detection results

input_videos/
└── .gitkeep              # Input tennis videos

output_videos/
└── .gitkeep              # Processed output videos
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

**Option 1: Fine-tuned Model (Recommended)**

```bash
python download_models.py
```

**Option 2: Use Generic YOLOv8n (Fallback)**

The module will automatically fall back to YOLOv8n if the fine-tuned model is not available.

## Usage

### Basic Example

```python
from utils.video_utils import read_video, save_video
from trackers.ball_tracker import BallTracker

# Read video
frames, metadata = read_video("input_videos/sample.mp4")

# Initialize tracker
tracker = BallTracker(model_path='models/yolov5_tennis_ball.pt')

# Detect ball
ball_detections = tracker.detect_frames(
    frames,
    read_from_stub=False,
    stub_path='tracker_stubs/ball_detections.pkl'
)

# Interpolate missing detections
ball_detections = tracker.interpolate_ball_positions(ball_detections)

# Detect shots
shot_frames = tracker.get_ball_shot_frames(ball_detections)

# Draw bounding boxes
output_frames = tracker.draw_bounding_boxes(frames, ball_detections)

# Save output
save_video(output_frames, "output_videos/ball_tracked.mp4", fps=metadata['fps'])
```

### Run Example Script

```bash
python examples/ball_tracking_example.py
```

## API Reference

### BallTracker Class

#### `__init__(model_path: str)`

Initialize ball tracker with YOLO model.

**Parameters:**
- `model_path`: Path to YOLO model file

**Example:**
```python
tracker = BallTracker(model_path='models/yolov5_tennis_ball.pt')
```

#### `detect_frame(frame: np.ndarray) -> dict`

Detect ball in a single frame.

**Parameters:**
- `frame`: Video frame (numpy array)

**Returns:**
- Dictionary: `{1: [x1, y1, x2, y2]}` or `{}` if no ball detected

**Example:**
```python
detection = tracker.detect_frame(frame)
# Returns: {1: [640.5, 360.2, 660.8, 380.5]}
```

#### `detect_frames(frames, read_from_stub=False, stub_path=None) -> List[dict]`

Detect ball in all frames with caching support.

**Parameters:**
- `frames`: List of video frames
- `read_from_stub`: Load from cache if True
- `stub_path`: Path to cache file

**Returns:**
- List of detection dictionaries (one per frame)

**Example:**
```python
detections = tracker.detect_frames(
    frames,
    read_from_stub=True,
    stub_path='tracker_stubs/ball_detections.pkl'
)
```

#### `interpolate_ball_positions(ball_detections: List[dict]) -> List[dict]`

Fill missing detections using linear interpolation.

**Parameters:**
- `ball_detections`: List of detection dictionaries

**Returns:**
- List of detection dictionaries with gaps filled

**Example:**
```python
interpolated = tracker.interpolate_ball_positions(ball_detections)
```

#### `get_ball_shot_frames(ball_detections: List[dict]) -> List[int]`

Detect frames where players hit the ball.

**Parameters:**
- `ball_detections`: List of detection dictionaries (interpolated)

**Returns:**
- List of frame numbers where shots occurred

**Example:**
```python
shot_frames = tracker.get_ball_shot_frames(ball_detections)
# Returns: [11, 58, 95, 131, 182]
```

#### `draw_bounding_boxes(frames, ball_detections) -> List[np.ndarray]`

Draw yellow bounding boxes on frames.

**Parameters:**
- `frames`: List of video frames
- `ball_detections`: List of detection dictionaries

**Returns:**
- List of frames with bounding boxes drawn

**Example:**
```python
output_frames = tracker.draw_bounding_boxes(frames, ball_detections)
```

## Testing

Run unit tests:

```bash
pytest tests/test_ball_tracker.py -v
```

Run with coverage:

```bash
pytest tests/test_ball_tracker.py --cov=trackers --cov-report=html
```

## Performance Characteristics

- **Detection Rate**: ≥75% before interpolation (depends on video quality)
- **Final Coverage**: ≥95% after interpolation
- **Shot Detection Accuracy**: ≥90% (±5 frame tolerance)
- **Processing Speed**: ~10-30 FPS on modern GPU

## Configuration

### Confidence Threshold

Adjust in `ball_tracker.py`:

```python
# Lower threshold = more detections (more false positives)
# Higher threshold = fewer detections (misses fast balls)
results = self.model.predict(frame, conf=0.15, verbose=False)
```

**Recommended values:**
- High quality video: 0.20-0.25
- Normal quality video: 0.15 (default)
- Low quality video: 0.10-0.12

### Shot Detection Sensitivity

Adjust in `ball_tracker.py`:

```python
# Lower value = more sensitive (may detect false shots)
# Higher value = less sensitive (may miss quick shots)
minimum_change_frames_for_hit = 25  # Default
```

**Recommended values:**
- Fast gameplay: 15-20
- Normal gameplay: 25 (default)
- Slow gameplay: 30-35

## Troubleshooting

### Low Detection Rate (<70%)

**Possible causes:**
- Ball too small or fast
- Poor lighting
- Low video quality

**Solutions:**
- Lower confidence threshold to 0.10
- Use higher resolution video
- Use fine-tuned model instead of generic

### Too Many False Positives

**Possible causes:**
- Confidence threshold too low
- Detecting white lines or clothing

**Solutions:**
- Increase confidence threshold to 0.20-0.25
- Add size filter (ball bbox should be 15-30 pixels)
- Use fine-tuned model

### Unrealistic Interpolation

**Possible causes:**
- Large gaps (>5 frames) between detections

**Solutions:**
- Improve detection rate before interpolation
- Gaps >5 frames should not be interpolated

### Missed Shot Detections

**Possible causes:**
- `minimum_change_frames_for_hit` too high

**Solutions:**
- Reduce to 15-20 for faster shots
- Check if interpolation is working correctly

### False Shot Detections (Bounces)

**Possible causes:**
- Bounces also cause direction changes

**Solutions:**
- Add player proximity check (future enhancement)
- Increase `minimum_change_frames_for_hit`

## Implementation Details

### Detection Algorithm

1. Uses YOLO object detection (YOLOv5/YOLOv8)
2. Confidence threshold: 0.15 (lower than player detection)
3. Takes highest confidence detection only
4. Assigns ID 1 to all ball detections

### Interpolation Algorithm

1. Converts detections to pandas DataFrame
2. Uses linear interpolation for gaps
3. Back-fills missing frames at start
4. Handles edge cases (all empty, single detection)

### Shot Detection Algorithm

1. Calculates ball vertical position over time
2. Applies rolling mean (window=5) to smooth trajectory
3. Calculates velocity (delta_y between frames)
4. Detects velocity reversals (up→down or down→up)
5. Confirms shot if reversal persists for 25 frames

## Future Enhancements

- [ ] Add player proximity check for shot detection
- [ ] Detect bounce events (ball touches ground)
- [ ] Add confidence scores to shot detections
- [ ] Support multiple ball tracking (practice sessions)
- [ ] Add ball trajectory prediction
- [ ] Export shot statistics (speed, angle, spin)

## License

TBD

## Contributing

Contributions welcome! Please follow the existing code style and add tests for new features.
