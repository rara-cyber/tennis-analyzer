# Tennis Analyzer

AI Tennis Analysis System - A comprehensive solution for analyzing tennis matches and player performance.

## Features

- ✅ **Court Keypoint Detection**: Detect 14 tennis court keypoints using ResNet50-based CNN
- ✅ **Coordinate Transformation**: Convert pixel coordinates to real-world meters
- ✅ **Mini-Court Visualization**: Overlay mini-court with player/ball positions on videos
- 🚧 **Player Tracking**: Track player movements (coming soon)
- 🚧 **Ball Tracking**: Detect and track tennis ball (coming soon)
- 🚧 **Shot Analysis**: Analyze shot types and patterns (coming soon)
- 🚧 **Statistics**: Generate match statistics and insights (coming soon)

## Project Structure

```
tennis-analyzer/
├── court_line_detector/    # Court keypoint detection using CNN
│   ├── __init__.py
│   └── court_line_detector.py
├── mini_court/             # Mini-court visualization
│   ├── __init__.py
│   └── mini_court.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── conversions.py      # Coordinate conversions
│   ├── video_utils.py      # Video I/O
│   └── bbox_utils.py       # Bounding box operations
├── constants/              # System constants
│   └── __init__.py
├── tests/                  # Unit tests
│   ├── test_court_detector.py
│   └── test_mini_court.py
├── examples/               # Usage examples
│   └── court_analysis_example.py
├── models/                 # Pre-trained models
├── input_videos/          # Input video files
├── output_videos/         # Processed output
├── requirements.txt       # Python dependencies
└── download_models.py     # Model download script
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/tennis-analyzer.git
   cd tennis-analyzer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models:**

   For testing with placeholder models:
   ```bash
   python download_models.py --placeholder
   ```

   For production (when models are available):
   ```bash
   python download_models.py
   ```

## Quick Start

### Basic Usage

```python
from utils.video_utils import read_video, save_video
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

# Read video
frames, metadata = read_video("input_videos/tennis_match.mp4")

# Detect court keypoints
detector = CourtLineDetector(model_path='models/court_keypoint_model.pth')
keypoints = detector.predict(frames[0])

# Create mini-court visualization
mini_court = MiniCourt(frames[0])
output_frames = []
for frame in frames:
    frame_with_court = mini_court.draw_court(frame)
    output_frames.append(frame_with_court)

# Save output
save_video(output_frames, "output_videos/with_mini_court.mp4", fps=metadata['fps'])
```

### Run Example

```bash
cd examples
python court_analysis_example.py
```

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_court_detector.py -v
pytest tests/test_mini_court.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Module Documentation

### Court Line Detector

Detects 14 court keypoints using a ResNet50-based CNN:

```python
from court_line_detector import CourtLineDetector

detector = CourtLineDetector(model_path='models/court_keypoint_model.pth')
keypoints = detector.predict(frame)  # Returns 28 floats (14 x/y pairs)

# Visualize keypoints
frame_with_keypoints = detector.draw_keypoints(frame, keypoints)
```

**Keypoint Layout:**
- 0-1: Top baseline corners (left, right)
- 2-3: Top service line corners
- 4-5: Net posts (left, right)
- 6-7: Bottom service line corners
- 8-9: Bottom baseline corners
- 10-11: Center baseline marks
- 12-13: Center service line marks

### Mini Court

Creates mini-court overlay with coordinate transformation:

```python
from mini_court import MiniCourt

mini_court = MiniCourt(frame)

# Draw court
frame_with_court = mini_court.draw_court(frame)

# Convert real position to mini-court coordinates
mini_pos = mini_court.convert_position_to_mini_court(
    position=(640, 360),
    closest_keypoint=(640, 350),
    closest_keypoint_index=0,
    player_height_pixels=200,
    player_height_meters=1.88
)

# Draw positions
positions = {1: mini_pos}
frame_with_points = mini_court.draw_points_on_mini_court(
    frame_with_court,
    positions,
    color=(0, 0, 255)
)
```

### Coordinate Conversions

Convert between pixels and meters:

```python
from utils.conversions import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters

# Meters to pixels
pixels = convert_meters_to_pixel_distance(
    meters=5.0,
    reference_height_in_meters=10.97,  # Court width
    reference_height_in_pixels=500
)

# Pixels to meters
meters = convert_pixel_distance_to_meters(
    pixels=250,
    reference_height_in_meters=10.97,
    reference_height_in_pixels=500
)
```

## Configuration

Court dimensions and system parameters are defined in `constants/__init__.py`:

```python
from constants import DOUBLE_LINE_WIDTH, HALF_COURT_HEIGHT, PLAYER_1_HEIGHT

# Tennis court dimensions (meters)
DOUBLE_LINE_WIDTH = 10.97  # Full court width
HALF_COURT_HEIGHT = 11.88  # Baseline to net

# Player heights (for coordinate transformation)
PLAYER_1_HEIGHT = 1.88
PLAYER_2_HEIGHT = 1.91
```

## Troubleshooting

### Model File Not Found

**Error:** `FileNotFoundError: models/court_keypoint_model.pth`

**Solution:**
```bash
python download_models.py --placeholder
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Coordinate Conversion Issues

Ensure you're using correct reference dimensions:
- Use court width (10.97m) as reference
- Use player height in both pixels and meters
- Verify keypoint detection accuracy first

## Performance

- **Court Detection**: ~200ms per frame (CPU)
- **Mini-Court Drawing**: ~10ms per frame
- **Memory Usage**: ~500MB for 1000 frames (1280x720)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development

### Code Style

- Follow PEP 8
- Use type hints
- Document all public functions

### Running Linters

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

## Roadmap

- [x] Court keypoint detection
- [x] Mini-court visualization
- [x] Coordinate transformation
- [ ] Player tracking and identification
- [ ] Ball detection and tracking
- [ ] Shot classification
- [ ] Match statistics
- [ ] Real-time analysis
- [ ] Web interface

## License

TBD

## Acknowledgments

- ResNet50 architecture from torchvision
- Tennis court dimensions from ITF regulations
- Inspired by professional tennis analysis systems
