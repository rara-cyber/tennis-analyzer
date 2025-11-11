# Sub-PRD 4: Court Analysis & Coordinate Transformation

**Module Name:** `court_analysis`  
**Claude Code Instruction:** "Implement this complete court keypoint detection and coordinate transformation system following all specifications below."

---

## Module Overview

Create a Python module that detects tennis court keypoints (corners, lines, net intersections) using a CNN model, transforms pixel coordinates to real-world metric coordinates, and creates a mini-court visualization system for overlaying player and ball positions.

## What You're Building

A court analysis system that:
- Detects 14 court keypoints using ResNet50-based CNN
- Maps pixel coordinates to real-world court dimensions (meters)
- Creates a mini-court visualization (250x500 pixels)
- Transforms player/ball positions from video to mini-court coordinates
- Draws mini-court overlay on output videos

## File Structure to Create

```
court_line_detector/
├── __init__.py
├── court_line_detector.py    # Keypoint detection class

mini_court/
├── __init__.py
├── mini_court.py              # Mini-court visualization and transformation

utils/
├── conversions.py             # Pixel ↔ meter conversion utilities

constants/
└── __init__.py                # Court dimensions (update existing file)

tests/
├── test_court_detector.py
└── test_mini_court.py

models/
└── court_keypoint_model.pth   # Pre-trained model (manual download)
```

## Requirements

### REQ-1: CourtLineDetector Class

**Class Definition:**

```python
class CourtLineDetector:
    """Detects tennis court keypoints using CNN model."""
    
    def __init__(self, model_path: str):
        """
        Initialize court keypoint detector.
        
        Args:
            model_path: Path to trained model weights (e.g., 'models/court_keypoint_model.pth')
        """
        
    def predict(self, frame: np.ndarray) -> list[float]:
        """
        Detect 14 court keypoints in a frame.
        
        Args:
            frame: Video frame (np.ndarray, shape (height, width, 3))
            
        Returns:
            List of 28 floats representing 14 (x, y) coordinate pairs
            Format: [x0, y0, x1, y1, x2, y2, ..., x13, y13]
            
        Note: Only processes first frame (assumes static camera)
        """
        
    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: list[float]
    ) -> np.ndarray:
        """
        Draw keypoints on frame for visualization.
        
        Args:
            frame: Video frame
            keypoints: List of 28 floats (14 x/y pairs)
            
        Returns:
            Frame with keypoints drawn as red circles with numbers
        """
```

### REQ-2: Model Architecture and Loading

**Implementation:**

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

def __init__(self, model_path: str):
    """Initialize court keypoint detector with ResNet50 backbone."""
    
    # Create ResNet50 model
    self.model = models.resnet50(pretrained=False)
    
    # Replace final fully connected layer
    # Original ResNet50 outputs 1000 classes
    # We need 28 outputs (14 keypoints × 2 coordinates)
    num_features = self.model.fc.in_features
    self.model.fc = nn.Linear(num_features, 28)
    
    # Load trained weights
    self.model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
    )
    
    # Set to evaluation mode
    self.model.eval()
    
    # Define image preprocessing transforms
    self.transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"✓ Court detector ({model_path}) loaded")
```

**Acceptance Criteria:**
- Model loads successfully within 3 seconds
- Works on CPU (no GPU required for single frame)
- Handles missing model file with clear error message

### REQ-3: Keypoint Detection

**Implementation:**

```python
def predict(self, frame: np.ndarray) -> list[float]:
    """Detect 14 court keypoints in frame."""
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions for denormalization later
    original_height, original_width = frame.shape[:2]
    
    # Apply transforms
    image_tensor = self.transforms(frame_rgb)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        outputs = self.model(image_tensor)
    
    # Convert to numpy and remove batch dimension
    keypoints = outputs.squeeze().cpu().numpy()
    
    # Denormalize coordinates from 224x224 to original resolution
    # Model outputs are for 224x224 image, need to scale to original
    for i in range(0, len(keypoints), 2):
        # Scale x coordinate
        keypoints[i] = keypoints[i] * original_width / 224
        # Scale y coordinate
        keypoints[i + 1] = keypoints[i + 1] * original_height / 224
    
    print(f"✓ Detected 14 court keypoints")
    return keypoints.tolist()
```

**Keypoint Indices:**
```
Keypoint layout on tennis court:
0: Top-left outer corner
1: Top-right outer corner
2: Top-left service box corner
3: Top-right service box corner
4: Left net post
5: Right net post
6: Bottom-left service box corner
7: Bottom-right service box corner
8: Bottom-left outer corner
9: Bottom-right outer corner
10: Top center (baseline)
11: Bottom center (baseline)
12: Left center (service line)
13: Right center (service line)
```

**Acceptance Criteria:**
- Returns list of exactly 28 floats
- Keypoints are within frame boundaries
- Average error <10 pixels on test images
- Processes single frame in <200ms

### REQ-4: Draw Keypoints for Visualization

**Implementation:**

```python
def draw_keypoints(
    self,
    frame: np.ndarray,
    keypoints: list[float]
) -> np.ndarray:
    """Draw keypoints on frame."""
    
    frame_copy = frame.copy()
    
    # Draw each keypoint
    for i in range(0, len(keypoints), 2):
        x = int(keypoints[i])
        y = int(keypoints[i + 1])
        
        # Draw red circle
        cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        
        # Draw keypoint number
        keypoint_num = i // 2
        cv2.putText(
            frame_copy,
            str(keypoint_num),
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )
    
    return frame_copy
```

**Acceptance Criteria:**
- Red circles visible at each keypoint
- Numbers displayed near each keypoint
- Does not modify original frame

## Court Constants

**File:** `constants/__init__.py` (update existing file)

```python
# Video processing constants (existing)
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov']
MIN_VIDEO_RESOLUTION = (1280, 720)
DEFAULT_FPS = 24.0
DEFAULT_CODEC = 'mp4v'

# Cache settings (existing)
CACHE_DIR = 'tracker_stubs'
ENABLE_CACHING = True

# Tennis court dimensions (in meters) - NEW
SINGLE_LINE_WIDTH = 8.23      # Width of singles court
DOUBLE_LINE_WIDTH = 10.97     # Width of doubles court
HALF_COURT_HEIGHT = 11.88     # Half court length
SERVICE_LINE_WIDTH = 6.4      # Distance from net to service line
DOUBLE_ALLEY_DIFFERENCE = 1.37  # Difference between singles and doubles width
NO_MANS_LAND_HEIGHT = 5.48    # Distance from service line to baseline

# Player heights (in meters) - for coordinate transformation
PLAYER_1_HEIGHT = 1.88
PLAYER_2_HEIGHT = 1.91
```

## Coordinate Conversion Utilities

**File:** `utils/conversions.py`

```python
def convert_meters_to_pixel_distance(
    meters: float,
    reference_height_in_meters: float,
    reference_height_in_pixels: float
) -> float:
    """
    Convert distance in meters to pixels using reference.
    
    Args:
        meters: Distance in meters to convert
        reference_height_in_meters: Known height in meters (e.g., court width = 10.97m)
        reference_height_in_pixels: Same height in pixels
        
    Returns:
        Distance in pixels
        
    Example:
        Court width is 10.97m and measures 500 pixels.
        How many pixels is 5 meters?
        convert_meters_to_pixel_distance(5, 10.97, 500) = 228.0
    """
    return (meters * reference_height_in_pixels) / reference_height_in_meters

def convert_pixel_distance_to_meters(
    pixels: float,
    reference_height_in_meters: float,
    reference_height_in_pixels: float
) -> float:
    """
    Convert distance in pixels to meters using reference.
    
    Args:
        pixels: Distance in pixels to convert
        reference_height_in_meters: Known height in meters
        reference_height_in_pixels: Same height in pixels
        
    Returns:
        Distance in meters
        
    Example:
        Court width is 10.97m and measures 500 pixels.
        How many meters is 250 pixels?
        convert_pixel_distance_to_meters(250, 10.97, 500) = 5.485
    """
    return (pixels * reference_height_in_meters) / reference_height_in_pixels
```

**Update `utils/__init__.py`:**
```python
from .video_utils import read_video, save_video, display_progress, save_cache, load_cache, cache_exists, draw_frame_number
from .bbox_utils import get_center_of_bbox, get_foot_position, get_bbox_width, get_bbox_height, measure_distance, measure_xy_distance
from .conversions import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters

__all__ = [
    'read_video', 'save_video', 'display_progress', 'save_cache', 'load_cache', 'cache_exists', 'draw_frame_number',
    'get_center_of_bbox', 'get_foot_position', 'get_bbox_width', 'get_bbox_height', 'measure_distance', 'measure_xy_distance',
    'convert_meters_to_pixel_distance', 'convert_pixel_distance_to_meters'
]
```

## REQ-5: MiniCourt Class

**File:** `mini_court/mini_court.py`

**Class Definition:**

```python
class MiniCourt:
    """
    Creates mini-court visualization and transforms coordinates
    from video pixels to mini-court pixels.
    """
    
    def __init__(self, frame: np.ndarray):
        """
        Initialize mini-court with frame dimensions.
        
        Args:
            frame: First video frame (to get dimensions)
        """
        
    def convert_position_to_mini_court(
        self,
        position: tuple[int, int],
        closest_keypoint: tuple[int, int],
        closest_keypoint_index: int,
        player_height_pixels: int,
        player_height_meters: float
    ) -> tuple[int, int]:
        """
        Convert position from video coordinates to mini-court coordinates.
        
        Args:
            position: (x, y) position in video frame
            closest_keypoint: (x, y) of nearest court keypoint in video
            closest_keypoint_index: Index of nearest keypoint (0-13)
            player_height_pixels: Player height in video (pixels)
            player_height_meters: Player actual height (meters)
            
        Returns:
            (x, y) position in mini-court coordinates
        """
        
    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw mini-court with lines on frame.
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with mini-court drawn in top-right corner
        """
        
    def draw_points_on_mini_court(
        self,
        frame: np.ndarray,
        positions: dict,
        color: tuple = (0, 0, 255)
    ) -> np.ndarray:
        """
        Draw player/ball positions on mini-court.
        
        Args:
            frame: Video frame with mini-court already drawn
            positions: Dict mapping ID to (x, y) mini-court position
            color: BGR color tuple
            
        Returns:
            Frame with positions drawn on mini-court
        """
```

### REQ-6: MiniCourt Implementation

**Implementation:**

```python
import cv2
import numpy as np
import sys
sys.path.append('..')
from constants import *
from utils.conversions import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters

class MiniCourt:
    """Mini-court visualization and coordinate transformation."""
    
    def __init__(self, frame: np.ndarray):
        """Initialize mini-court dimensions and position."""
        
        # Mini-court dimensions (pixels)
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50  # Distance from frame edges
        self.padding_court = 20  # Padding inside rectangle
        
        # Calculate position on frame (top-right corner)
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.buffer
        
        # Calculate actual court area (inside padding)
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
        # Set up court keypoints for mini-court
        self._set_court_drawing_keypoints()
        
        # Set up court lines
        self._set_court_lines()
        
        print(f"✓ Mini-court initialized ({self.drawing_rectangle_width}x{self.drawing_rectangle_height})")
    
    def _set_court_drawing_keypoints(self):
        """Calculate mini-court keypoint positions."""
        
        self.drawing_keypoints = [0] * 28
        
        # Helper function to convert meters to mini-court pixels
        def meters_to_pixels(meters):
            return convert_meters_to_pixel_distance(
                meters,
                DOUBLE_LINE_WIDTH,
                self.court_drawing_width
            )
        
        # Keypoint 0: Top-left corner
        self.drawing_keypoints[0] = int(self.court_start_x)
        self.drawing_keypoints[1] = int(self.court_start_y)
        
        # Keypoint 1: Top-right corner
        self.drawing_keypoints[2] = int(self.court_end_x)
        self.drawing_keypoints[3] = int(self.court_start_y)
        
        # Keypoint 2: Top-left (after one half court)
        self.drawing_keypoints[4] = int(self.court_start_x)
        self.drawing_keypoints[5] = int(self.court_start_y + meters_to_pixels(HALF_COURT_HEIGHT * 2))
        
        # Keypoint 3: Top-right (after one half court)
        self.drawing_keypoints[6] = int(self.court_end_x)
        self.drawing_keypoints[7] = int(self.court_start_y + meters_to_pixels(HALF_COURT_HEIGHT * 2))
        
        # Continue for all 14 keypoints...
        # (Full implementation would include all 14 keypoints)
        # For brevity, showing pattern for first 4
        
    def _set_court_lines(self):
        """Define which keypoints connect to form court lines."""
        
        self.lines = [
            (0, 2),   # Top baseline
            (0, 4),   # Left sideline
            (1, 3),   # Right sideline
            (2, 6),   # Bottom baseline
            (4, 5),   # Net line
            # Add more lines for service boxes, center lines, etc.
        ]
    
    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        """Draw mini-court on frame."""
        
        frame_copy = frame.copy()
        
        # Draw white semi-transparent background
        shapes = np.zeros_like(frame_copy, dtype=np.uint8)
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            -1
        )
        
        # Blend with original frame
        alpha = 0.5
        mask = shapes.astype(bool)
        frame_copy[mask] = cv2.addWeighted(
            frame_copy, alpha,
            shapes, 1 - alpha, 0
        )[mask]
        
        # Draw court lines
        for line in self.lines:
            start_point = (
                int(self.drawing_keypoints[line[0] * 2]),
                int(self.drawing_keypoints[line[0] * 2 + 1])
            )
            end_point = (
                int(self.drawing_keypoints[line[1] * 2]),
                int(self.drawing_keypoints[line[1] * 2 + 1])
            )
            cv2.line(frame_copy, start_point, end_point, (0, 0, 0), 2)
        
        # Draw net (thicker line)
        net_start = (
            int(self.court_start_x),
            int(self.court_start_y + self.court_drawing_width)
        )
        net_end = (
            int(self.court_end_x),
            int(self.court_start_y + self.court_drawing_width)
        )
        cv2.line(frame_copy, net_start, net_end, (0, 0, 0), 3)
        
        return frame_copy
    
    def convert_position_to_mini_court(
        self,
        position: tuple[int, int],
        closest_keypoint: tuple[int, int],
        closest_keypoint_index: int,
        player_height_pixels: int,
        player_height_meters: float
    ) -> tuple[int, int]:
        """Transform video position to mini-court position."""
        
        # Calculate distance from position to closest keypoint (in pixels)
        distance_x_pixels = abs(position[0] - closest_keypoint[0])
        distance_y_pixels = abs(position[1] - closest_keypoint[1])
        
        # Convert pixel distance to meters using player height as reference
        distance_x_meters = convert_pixel_distance_to_meters(
            distance_x_pixels,
            player_height_meters,
            player_height_pixels
        )
        distance_y_meters = convert_pixel_distance_to_meters(
            distance_y_pixels,
            player_height_meters,
            player_height_pixels
        )
        
        # Convert meters to mini-court pixels
        distance_x_mini = convert_meters_to_pixel_distance(
            distance_x_meters,
            DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )
        distance_y_mini = convert_meters_to_pixel_distance(
            distance_y_meters,
            DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )
        
        # Get closest keypoint position in mini-court
        closest_mini_x = self.drawing_keypoints[closest_keypoint_index * 2]
        closest_mini_y = self.drawing_keypoints[closest_keypoint_index * 2 + 1]
        
        # Calculate final mini-court position
        mini_x = int(closest_mini_x + distance_x_mini)
        mini_y = int(closest_mini_y + distance_y_mini)
        
        return (mini_x, mini_y)
    
    def draw_points_on_mini_court(
        self,
        frame: np.ndarray,
        positions: dict,
        color: tuple = (0, 0, 255)
    ) -> np.ndarray:
        """Draw positions on mini-court."""
        
        frame_copy = frame.copy()
        
        for player_id, position in positions.items():
            x, y = position
            cv2.circle(frame_copy, (int(x), int(y)), 5, color, -1)
        
        return frame_copy
```

**Acceptance Criteria:**
- Mini-court appears in top-right corner
- White semi-transparent background
- Black court lines visible
- Positions plot correctly relative to court layout

## Expose Classes

**File:** `court_line_detector/__init__.py`

```python
from .court_line_detector import CourtLineDetector

__all__ = ['CourtLineDetector']
```

**File:** `mini_court/__init__.py`

```python
from .mini_court import MiniCourt

__all__ = ['MiniCourt']
```

## Testing Requirements

**File:** `tests/test_court_detector.py`

```python
import pytest
import numpy as np
from court_line_detector import CourtLineDetector

def test_court_detector_initialization():
    """Test detector initializes with model."""
    detector = CourtLineDetector(model_path='models/court_keypoint_model.pth')
    assert detector.model is not None

def test_predict_returns_28_values():
    """Test prediction returns 28 floats."""
    detector = CourtLineDetector(model_path='models/court_keypoint_model.pth')
    
    # Create dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Predict
    keypoints = detector.predict(frame)
    
    # Check return type and length
    assert isinstance(keypoints, list)
    assert len(keypoints) == 28

def test_keypoints_within_frame():
    """Test keypoints are within frame boundaries."""
    detector = CourtLineDetector(model_path='models/court_keypoint_model.pth')
    
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    keypoints = detector.predict(frame)
    
    # Check all x coordinates within width
    for i in range(0, 28, 2):
        assert 0 <= keypoints[i] <= 1280
    
    # Check all y coordinates within height
    for i in range(1, 28, 2):
        assert 0 <= keypoints[i] <= 720
```

**File:** `tests/test_mini_court.py`

```python
import pytest
import numpy as np
from mini_court import MiniCourt

def test_mini_court_initialization():
    """Test mini-court initializes with frame."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mini_court = MiniCourt(frame)
    
    assert mini_court.drawing_rectangle_width == 250
    assert mini_court.drawing_rectangle_height == 500

def test_coordinate_conversion():
    """Test converting video position to mini-court position."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mini_court = MiniCourt(frame)
    
    # Mock conversion
    position = (640, 360)  # Center of frame
    closest_keypoint = (640, 360)
    closest_keypoint_index = 0
    player_height_pixels = 200
    player_height_meters = 1.88
    
    mini_pos = mini_court.convert_position_to_mini_court(
        position,
        closest_keypoint,
        closest_keypoint_index,
        player_height_pixels,
        player_height_meters
    )
    
    # Check return type
    assert isinstance(mini_pos, tuple)
    assert len(mini_pos) == 2
```

## Usage Example

**File:** `examples/court_analysis_example.py`

```python
from utils.video_utils import read_video, save_video
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

# Read video
print("Reading video...")
frames, metadata = read_video("input_videos/sample.mp4")

# Initialize court detector
detector = CourtLineDetector(model_path='models/court_keypoint_model.pth')

# Detect court keypoints (only first frame)
print("\nDetecting court keypoints...")
court_keypoints = detector.predict(frames[0])
print(f"Keypoints: {court_keypoints[:4]}...")  # Print first 2 points

# Draw keypoints on first frame for verification
frame_with_keypoints = detector.draw_keypoints(frames[0], court_keypoints)
cv2.imwrite("output_videos/court_keypoints.jpg", frame_with_keypoints)
print("✓ Saved keypoints visualization")

# Initialize mini-court
mini_court = MiniCourt(frames[0])

# Draw mini-court on all frames
print("\nDrawing mini-court...")
output_frames = []
for frame in frames:
    frame_with_court = mini_court.draw_court(frame)
    output_frames.append(frame_with_court)

# Save output
print("\nSaving video...")
save_video(output_frames, "output_videos/with_mini_court.mp4", fps=metadata['fps'])

print("\n✓ Done!")
```

## Dependencies

Add to `requirements.txt`:
```
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.1.78
numpy==1.24.3
```

## Model Download

**Create download script:** `download_models.py` (update existing)

```python
import urllib.request
import os

def download_court_model():
    """Download court keypoint detection model."""
    
    os.makedirs('models', exist_ok=True)
    
    # Model URL (replace with actual URL)
    model_url = "https://github.com/your-repo/releases/download/v1.0/court_keypoint_model.pth"
    model_path = "models/court_keypoint_model.pth"
    
    if os.path.exists(model_path):
        print(f"✓ Court model already exists: {model_path}")
        return
    
    print(f"Downloading court keypoint model...")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"✓ Court model downloaded: {model_path}")

def download_ball_model():
    """Download ball detection model."""
    # (existing code from Sub-PRD 3)
    pass

if __name__ == "__main__":
    download_court_model()
    download_ball_model()
```

## Validation Checklist

Before marking this module complete:

- [ ] Court detector loads model successfully
- [ ] `predict()` returns 28 keypoint coordinates
- [ ] Keypoints are within frame boundaries
- [ ] Keypoint visualization shows red circles with numbers
- [ ] Mini-court draws in top-right corner with white background
- [ ] Court lines visible on mini-court
- [ ] Coordinate conversion produces reasonable positions
- [ ] All unit tests pass
- [ ] Example script runs end-to-end

## Common Issues & Solutions

**Issue 1: Model file not found**
- **Cause:** Model not downloaded
- **Solution:** Run `python download_models.py`

**Issue 2: RuntimeError: Expected tensor on CPU**
- **Cause:** Model loaded on wrong device
- **Solution:** Use `map_location=torch.device('cpu')` in `torch.load()`

**Issue 3: Keypoints appear in wrong positions**
- **Cause:** Coordinate denormalization incorrect
- **Solution:** Verify scaling from 224x224 to original frame size

**Issue 4: Mini-court not visible**
- **Cause:** Position calculation error or transparency too high
- **Solution:** Check `start_x`, `start_y` calculations and alpha blending

**Issue 5: Lines don't connect properly**
- **Cause:** Incorrect keypoint indices in `self.lines`
- **Solution:** Verify line definitions match keypoint layout

---

**End of Sub-PRD 4**