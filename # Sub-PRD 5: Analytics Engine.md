# Sub-PRD 5: Analytics Engine

**Module Name:** `analytics`
**Claude Code Instruction:** "Implement this complete analytics engine module following all specifications below."

---

## Module Overview

Create a Python module that calculates performance metrics from tracked player and ball positions. This module transforms raw detection data into actionable insights including shot speeds, player movement speeds, distance covered, and shot statistics.

## What You're Building

An analytics system that:
- Detects when players hit the ball (shot events)
- Calculates ball speed for each shot in km/h
- Tracks player movement speed during rallies
- Computes total distance covered by each player
- Generates comprehensive match statistics

## File Structure to Create

```
analytics/
├── __init__.py
├── shot_detector.py         # Shot and bounce detection
├── speed_calculator.py       # Speed calculations
├── distance_tracker.py       # Distance tracking
└── statistics.py             # Statistics aggregation

tests/
├── test_shot_detection.py
├── test_speed_calculator.py
└── test_distance_tracker.py
```

## Requirements

### REQ-1: Shot Detection

**Function Signature:**
```python
def detect_shots(
    ball_positions: list[dict],
    player_positions: dict[int, list[dict]],
    fps: float
) -> list[dict]:
    """
    Detect frames where players hit the ball.

    Args:
        ball_positions: List of dicts with keys: {'frame': int, 'x': float, 'y': float, 'bbox': list}
        player_positions: Dict mapping player_id to list of position dicts
                         {1: [{'frame': int, 'x': float, 'y': float, 'bbox': list}, ...], 2: [...]}
        fps: Video frame rate

    Returns:
        List of shot event dicts with keys:
        {
            'frame': int,              # Frame number when shot occurred
            'player_id': int,          # Which player hit the ball (1 or 2)
            'ball_position': (float, float),  # Ball position at shot (x, y)
            'player_position': (float, float), # Player position at shot (x, y)
            'ball_velocity_y': float   # Ball's vertical velocity change
        }
    """
```

**Implementation Details:**

**Step 1: Calculate Ball Velocity**
```python
def calculate_velocity(positions: list[dict], fps: float) -> list[float]:
    """Calculate frame-to-frame velocity in pixels per second."""
    velocities_y = []

    for i in range(len(positions) - 1):
        current_y = positions[i]['y']
        next_y = positions[i + 1]['y']

        # Velocity = change in position / time
        velocity = (next_y - current_y) * fps  # pixels/second
        velocities_y.append(velocity)

    return velocities_y
```

**Step 2: Detect Velocity Direction Changes**
```python
def detect_direction_changes(velocities: list[float], min_sustained_frames: int = 25) -> list[int]:
    """
    Detect when velocity changes direction (positive to negative or vice versa).

    Args:
        velocities: List of velocity values
        min_sustained_frames: Minimum frames the new direction must be sustained

    Returns:
        List of frame indices where sustained direction changes occurred
    """
    direction_changes = []

    for i in range(len(velocities) - min_sustained_frames):
        current_vel = velocities[i]

        # Check if next min_sustained_frames all have opposite sign
        if current_vel > 0:  # Currently moving down (positive y)
            # Check if sustained upward movement follows
            if all(v < 0 for v in velocities[i+1:i+min_sustained_frames+1]):
                direction_changes.append(i)
        elif current_vel < 0:  # Currently moving up (negative y)
            # Check if sustained downward movement follows
            if all(v > 0 for v in velocities[i+1:i+min_sustained_frames+1]):
                direction_changes.append(i)

    return direction_changes
```

**Step 3: Identify Which Player Hit the Ball**
```python
def find_closest_player(
    ball_pos: tuple[float, float],
    player_positions: dict[int, list[dict]],
    frame: int,
    max_distance: float = 200.0  # pixels
) -> int | None:
    """
    Find which player is closest to ball at given frame.

    Args:
        ball_pos: (x, y) ball position
        player_positions: Dict of player positions
        frame: Frame number
        max_distance: Maximum distance (pixels) to consider a player

    Returns:
        Player ID (1 or 2) or None if no player within max_distance
    """
    import math

    closest_player = None
    min_distance = max_distance

    for player_id, positions in player_positions.items():
        # Find position at this frame
        player_pos = next((p for p in positions if p['frame'] == frame), None)
        if not player_pos:
            continue

        # Calculate Euclidean distance
        distance = math.sqrt(
            (ball_pos[0] - player_pos['x'])**2 +
            (ball_pos[1] - player_pos['y'])**2
        )

        if distance < min_distance:
            min_distance = distance
            closest_player = player_id

    return closest_player
```

**Main Shot Detection Logic:**
```python
def detect_shots(ball_positions, player_positions, fps):
    """Detect shot events."""
    shots = []

    # Step 1: Calculate ball velocities
    velocities_y = calculate_velocity(ball_positions, fps)

    # Step 2: Find velocity direction changes (potential shots)
    shot_frames = detect_direction_changes(velocities_y, min_sustained_frames=25)

    # Step 3: For each potential shot, identify which player hit it
    for frame_idx in shot_frames:
        frame = ball_positions[frame_idx]['frame']
        ball_pos = (ball_positions[frame_idx]['x'], ball_positions[frame_idx]['y'])

        # Find closest player
        player_id = find_closest_player(ball_pos, player_positions, frame)

        if player_id is not None:
            # Get player position
            player_pos_data = next(
                p for p in player_positions[player_id] if p['frame'] == frame
            )
            player_pos = (player_pos_data['x'], player_pos_data['y'])

            shots.append({
                'frame': frame,
                'player_id': player_id,
                'ball_position': ball_pos,
                'player_position': player_pos,
                'ball_velocity_y': velocities_y[frame_idx]
            })

    return shots
```

**Acceptance Criteria:**
- Detects ≥90% of actual shots in test videos
- False positive rate <10% (doesn't detect non-shots)
- Correctly identifies which player made each shot
- Handles edge cases: serves, volleys, smashes
- Min sustained frames = 25 (configurable)

### REQ-2: Bounce Detection

**Function Signature:**
```python
def detect_bounces(
    ball_positions: list[dict],
    fps: float,
    min_sustained_frames: int = 25
) -> list[dict]:
    """
    Detect frames where ball bounces on the ground.

    Args:
        ball_positions: List of ball position dicts
        fps: Video frame rate
        min_sustained_frames: Minimum frames for sustained velocity change

    Returns:
        List of bounce event dicts with keys:
        {
            'frame': int,
            'position': (float, float),  # (x, y) where ball bounced
            'velocity_change': float     # Magnitude of velocity change
        }
    """
```

**Implementation Details:**
- Use same velocity calculation and direction change detection as shot detection
- Bounces are detected when ball changes from downward (positive y velocity) to upward (negative y velocity)
- Store bounce positions for heatmap generation

```python
def detect_bounces(ball_positions, fps, min_sustained_frames=25):
    """Detect ball bounce events."""
    bounces = []

    # Calculate velocities
    velocities_y = calculate_velocity(ball_positions, fps)

    # Find direction changes (downward to upward = bounce)
    bounce_frames = []
    for i in range(len(velocities_y) - min_sustained_frames):
        if velocities_y[i] > 0:  # Moving down
            # Check sustained upward movement
            if all(v < 0 for v in velocities_y[i+1:i+min_sustained_frames+1]):
                bounce_frames.append(i)

    # Create bounce events
    for frame_idx in bounce_frames:
        frame = ball_positions[frame_idx]['frame']
        position = (ball_positions[frame_idx]['x'], ball_positions[frame_idx]['y'])
        velocity_change = abs(velocities_y[frame_idx+1] - velocities_y[frame_idx])

        bounces.append({
            'frame': frame,
            'position': position,
            'velocity_change': velocity_change
        })

    return bounces
```

**Acceptance Criteria:**
- Detects ≥85% of actual bounces
- Bounce positions accurate to ±10 pixels
- Stores positions in format compatible with heatmap module
- Filters out false positives from camera shake

### REQ-3: Shot Speed Calculation

**Function Signature:**
```python
def calculate_shot_speed(
    shot_event: dict,
    ball_positions: list[dict],
    fps: float,
    pixel_to_meter_ratio: float
) -> float:
    """
    Calculate ball speed for a shot in km/h.

    Args:
        shot_event: Dict from detect_shots() containing shot details
        ball_positions: Full list of ball positions
        fps: Video frame rate
        pixel_to_meter_ratio: Conversion factor from pixels to meters

    Returns:
        Shot speed in kilometers per hour

    Raises:
        ValueError: If speed is unrealistic (>200 km/h or <0)
    """
```

**Implementation Details:**

```python
def calculate_shot_speed(shot_event, ball_positions, fps, pixel_to_meter_ratio):
    """Calculate shot speed in km/h."""
    shot_frame = shot_event['frame']

    # Find shot in ball_positions list
    shot_idx = next(i for i, pos in enumerate(ball_positions) if pos['frame'] == shot_frame)

    # Get position 10 frames after shot (for speed calculation)
    frame_offset = 10
    if shot_idx + frame_offset >= len(ball_positions):
        frame_offset = len(ball_positions) - shot_idx - 1

    if frame_offset < 2:
        # Not enough data to calculate speed
        return 0.0

    # Calculate distance traveled
    start_pos = ball_positions[shot_idx]
    end_pos = ball_positions[shot_idx + frame_offset]

    distance_pixels = math.sqrt(
        (end_pos['x'] - start_pos['x'])**2 +
        (end_pos['y'] - start_pos['y'])**2
    )

    # Convert to meters
    distance_meters = distance_pixels * pixel_to_meter_ratio

    # Calculate time elapsed
    time_seconds = frame_offset / fps

    # Calculate speed
    speed_mps = distance_meters / time_seconds  # meters per second
    speed_kmh = speed_mps * 3.6  # convert to km/h

    # Validate speed
    if speed_kmh < 0 or speed_kmh > 200:
        print(f"Warning: Unrealistic speed detected: {speed_kmh:.1f} km/h")
        return 0.0

    return speed_kmh
```

**Acceptance Criteria:**
- Speed calculation error <5% compared to ground truth
- Returns speed in km/h (not m/s)
- Filters unrealistic speeds (>200 km/h flagged as errors)
- Handles edge cases: ball leaving frame after shot
- Uses mini-court coordinates for accurate distance

### REQ-4: Player Movement Speed Calculation

**Function Signature:**
```python
def calculate_player_speed(
    player_id: int,
    start_frame: int,
    end_frame: int,
    player_positions: dict[int, list[dict]],
    fps: float,
    pixel_to_meter_ratio: float
) -> float:
    """
    Calculate player movement speed between two frames.

    Args:
        player_id: Player ID (1 or 2)
        start_frame: Starting frame number
        end_frame: Ending frame number
        player_positions: Dict of player positions
        fps: Video frame rate
        pixel_to_meter_ratio: Conversion factor

    Returns:
        Average speed in km/h
    """
```

**Implementation Details:**

```python
def calculate_player_speed(player_id, start_frame, end_frame, player_positions, fps, pixel_to_meter_ratio):
    """Calculate player movement speed."""
    positions = player_positions[player_id]

    # Get start and end positions
    start_pos = next((p for p in positions if p['frame'] == start_frame), None)
    end_pos = next((p for p in positions if p['frame'] == end_frame), None)

    if not start_pos or not end_pos:
        return 0.0

    # Calculate distance traveled
    distance_pixels = math.sqrt(
        (end_pos['x'] - start_pos['x'])**2 +
        (end_pos['y'] - start_pos['y'])**2
    )

    # Convert to meters
    distance_meters = distance_pixels * pixel_to_meter_ratio

    # Calculate time elapsed
    time_seconds = (end_frame - start_frame) / fps

    if time_seconds == 0:
        return 0.0

    # Calculate speed
    speed_mps = distance_meters / time_seconds
    speed_kmh = speed_mps * 3.6

    return speed_kmh
```

**Acceptance Criteria:**
- Calculates speed between any two frames
- Returns speed in km/h
- Handles cases where player not visible in some frames
- Speed calculation error <10% compared to GPS tracking

### REQ-5: Distance Tracking

**Function Signature:**
```python
def calculate_total_distance(
    player_id: int,
    player_positions: dict[int, list[dict]],
    pixel_to_meter_ratio: float
) -> float:
    """
    Calculate total distance covered by a player.

    Args:
        player_id: Player ID (1 or 2)
        player_positions: Dict of player positions
        pixel_to_meter_ratio: Conversion factor

    Returns:
        Total distance in meters
    """
```

**Implementation Details:**

```python
def calculate_total_distance(player_id, player_positions, pixel_to_meter_ratio):
    """Calculate total distance covered."""
    positions = player_positions[player_id]

    if len(positions) < 2:
        return 0.0

    total_distance_meters = 0.0

    # Sum frame-to-frame distances
    for i in range(len(positions) - 1):
        pos1 = positions[i]
        pos2 = positions[i + 1]

        # Calculate distance in pixels
        distance_pixels = math.sqrt(
            (pos2['x'] - pos1['x'])**2 +
            (pos2['y'] - pos1['y'])**2
        )

        # Convert to meters and accumulate
        distance_meters = distance_pixels * pixel_to_meter_ratio
        total_distance_meters += distance_meters

    return total_distance_meters
```

**Acceptance Criteria:**
- Accumulates distance across entire match
- Uses mini-court coordinates for accuracy
- Handles gaps in tracking (player temporarily not visible)
- Accuracy: ±0.5 meters per rally

### REQ-6: Match Statistics Generation

**Function Signature:**
```python
def generate_match_statistics(
    shots: list[dict],
    bounces: list[dict],
    player_positions: dict[int, list[dict]],
    fps: float,
    pixel_to_meter_ratio: float,
    video_metadata: dict
) -> dict:
    """
    Generate comprehensive match statistics.

    Args:
        shots: List of shot events from detect_shots()
        bounces: List of bounce events from detect_bounces()
        player_positions: Dict of player positions
        fps: Video frame rate
        pixel_to_meter_ratio: Conversion factor
        video_metadata: Dict with video info (duration, filename, etc.)

    Returns:
        Statistics dict with structure defined in master PRD (MatchStatistics)
    """
```

**Implementation Details:**

```python
def generate_match_statistics(shots, bounces, player_positions, fps, pixel_to_meter_ratio, video_metadata):
    """Generate match statistics."""
    stats = {
        'video_metadata': {
            'filename': video_metadata.get('filename', 'unknown'),
            'duration_seconds': video_metadata.get('duration', 0),
            'total_frames': video_metadata.get('total_frames', 0),
            'processed_date': datetime.datetime.now().isoformat()
        },
        'player_1': {},
        'player_2': {},
        'match_summary': {}
    }

    # Process each player
    for player_id in [1, 2]:
        player_key = f'player_{player_id}'

        # Get all shots by this player
        player_shots = [s for s in shots if s['player_id'] == player_id]

        # Calculate shot statistics
        shot_speeds = []
        for shot in player_shots:
            speed = calculate_shot_speed(shot, [], fps, pixel_to_meter_ratio)  # Simplified
            if speed > 0:
                shot_speeds.append(speed)

        stats[player_key]['total_shots'] = len(player_shots)
        stats[player_key]['average_shot_speed_kmh'] = sum(shot_speeds) / len(shot_speeds) if shot_speeds else 0
        stats[player_key]['max_shot_speed_kmh'] = max(shot_speeds) if shot_speeds else 0

        # Calculate distance covered
        total_distance = calculate_total_distance(player_id, player_positions, pixel_to_meter_ratio)
        stats[player_key]['total_distance_covered_meters'] = total_distance

        # Calculate average movement speed
        duration_seconds = video_metadata.get('duration', 1)
        avg_speed_mps = total_distance / duration_seconds if duration_seconds > 0 else 0
        stats[player_key]['average_movement_speed_kmh'] = avg_speed_mps * 3.6

        # Placeholder for other stats (will be filled by court analysis module)
        stats[player_key]['court_coverage_percentage'] = 0.0
        stats[player_key]['zone_distribution'] = {}
        stats[player_key]['max_movement_speed_kmh'] = 0.0

    # Match summary
    stats['match_summary']['total_rallies'] = len(shots) // 2  # Estimate
    stats['match_summary']['total_ball_bounces'] = len(bounces)
    stats['match_summary']['average_rally_duration_seconds'] = 0.0  # Placeholder
    stats['match_summary']['longest_rally_duration_seconds'] = 0.0  # Placeholder

    return stats
```

**Acceptance Criteria:**
- Returns statistics in JSON-compatible format
- All numeric values formatted to 1 decimal place
- Includes metadata: filename, duration, processed date
- Calculates per-player and match-level statistics
- Output matches MatchStatistics data model from master PRD

### REQ-7: Statistics Export

**Function Signature:**
```python
def export_statistics_json(
    statistics: dict,
    output_path: str
) -> None:
    """
    Export statistics to JSON file.

    Args:
        statistics: Statistics dict from generate_match_statistics()
        output_path: Path where JSON file will be saved
    """
```

**Implementation Details:**

```python
import json
import os

def export_statistics_json(statistics, output_path):
    """Export statistics to JSON file."""
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write JSON with pretty formatting
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)

    print(f"✓ Statistics saved: {output_path}")
```

**Acceptance Criteria:**
- Creates valid JSON file
- Pretty-printed with 2-space indentation
- Creates output directory if needed
- File can be loaded back with `json.load()`

## Constants to Define

**File:** `analytics/__init__.py`

```python
import math

# Shot detection parameters
MIN_SUSTAINED_FRAMES = 25  # Minimum frames for velocity change to count as shot
MAX_PLAYER_DISTANCE_PIXELS = 200  # Max distance from ball to player for shot attribution
FRAME_OFFSET_SPEED_CALC = 10  # Frames after shot to use for speed calculation

# Speed validation
MIN_REALISTIC_SPEED_KMH = 10.0   # Minimum realistic shot speed
MAX_REALISTIC_SPEED_KMH = 200.0  # Maximum realistic shot speed
MIN_REALISTIC_PLAYER_SPEED = 0.0
MAX_REALISTIC_PLAYER_SPEED = 30.0  # Max human running speed ~28 km/h

# Distance tracking
DISTANCE_SMOOTHING_WINDOW = 5  # Frames to average for smoothing

# Statistics formatting
DECIMAL_PLACES = 1  # Round statistics to 1 decimal place
```

## Testing Requirements

**File:** `tests/test_shot_detection.py`

```python
import pytest
from analytics.shot_detector import detect_shots, detect_bounces

def test_detect_shots_basic():
    """Test basic shot detection."""
    # Create mock ball positions with velocity change
    ball_positions = []
    for i in range(100):
        y = 500 + i * 2 if i < 50 else 600 - (i - 50) * 2  # Downward then upward
        ball_positions.append({'frame': i, 'x': 500, 'y': y})

    player_positions = {
        1: [{'frame': i, 'x': 480, 'y': 600} for i in range(100)],
        2: [{'frame': i, 'x': 520, 'y': 400} for i in range(100)]
    }

    shots = detect_shots(ball_positions, player_positions, fps=30.0)

    # Should detect at least one shot around frame 50
    assert len(shots) > 0
    assert 45 <= shots[0]['frame'] <= 55

def test_detect_bounces():
    """Test bounce detection."""
    # Create ball trajectory with bounce
    ball_positions = []
    for i in range(100):
        y = 300 + i * 3 if i < 40 else 420 - (i - 40) * 3  # Down then up
        ball_positions.append({'frame': i, 'x': 500, 'y': y})

    bounces = detect_bounces(ball_positions, fps=30.0)

    # Should detect bounce around frame 40
    assert len(bounces) > 0
    assert 35 <= bounces[0]['frame'] <= 45
```

**File:** `tests/test_speed_calculator.py`

```python
def test_calculate_shot_speed():
    """Test shot speed calculation."""
    from analytics.speed_calculator import calculate_shot_speed

    shot_event = {
        'frame': 50,
        'player_id': 1,
        'ball_position': (500, 400),
        'player_position': (480, 600)
    }

    # Create ball positions (moving 100 pixels in 10 frames)
    ball_positions = [
        {'frame': 50, 'x': 500, 'y': 400},
        *[{'frame': 50+i, 'x': 500+i*10, 'y': 400} for i in range(1, 11)]
    ]

    speed = calculate_shot_speed(shot_event, ball_positions, fps=30.0, pixel_to_meter_ratio=0.01)

    # Speed should be positive and realistic
    assert 0 < speed < 200
```

**File:** `tests/test_distance_tracker.py`

```python
def test_calculate_total_distance():
    """Test distance tracking."""
    from analytics.distance_tracker import calculate_total_distance

    # Player moves 100 pixels each frame for 10 frames
    player_positions = {
        1: [{'frame': i, 'x': 500 + i*100, 'y': 400} for i in range(10)]
    }

    distance = calculate_total_distance(1, player_positions, pixel_to_meter_ratio=0.01)

    # 10 frames * 100 pixels * 0.01 m/pixel = 10 meters
    assert 9.0 < distance < 11.0  # Allow small tolerance
```

## Usage Example

**File:** `examples/analytics_example.py`

```python
from analytics.shot_detector import detect_shots, detect_bounces
from analytics.speed_calculator import calculate_shot_speed
from analytics.distance_tracker import calculate_total_distance
from analytics.statistics import generate_match_statistics, export_statistics_json

# Mock data (in real use, comes from tracker modules)
ball_positions = [...]  # From ball tracker
player_positions = {1: [...], 2: [...]}  # From player tracker
fps = 30.0
pixel_to_meter_ratio = 0.02  # From court analysis module

# Detect shots and bounces
print("Detecting shots...")
shots = detect_shots(ball_positions, player_positions, fps)
print(f"Found {len(shots)} shots")

print("Detecting bounces...")
bounces = detect_bounces(ball_positions, fps)
print(f"Found {len(bounces)} bounces")

# Calculate statistics
print("Generating statistics...")
video_metadata = {
    'filename': 'match.mp4',
    'duration': 1800.0,
    'total_frames': 54000
}

stats = generate_match_statistics(
    shots,
    bounces,
    player_positions,
    fps,
    pixel_to_meter_ratio,
    video_metadata
)

# Export to JSON
export_statistics_json(stats, 'output_stats/match_statistics.json')

print(f"Player 1: {stats['player_1']['total_shots']} shots, "
      f"{stats['player_1']['average_shot_speed_kmh']:.1f} km/h avg speed")
print(f"Player 2: {stats['player_2']['total_shots']} shots, "
      f"{stats['player_2']['average_shot_speed_kmh']:.1f} km/h avg speed")
```

## Dependencies to Install

Add to `requirements.txt`:
```
numpy==1.24.3
```

(No additional dependencies beyond what's already installed for other modules)

## Validation Checklist

Before marking this module complete, verify:

- [ ] `detect_shots()` correctly identifies shot events
- [ ] `detect_shots()` assigns shots to correct players
- [ ] `detect_bounces()` identifies ball bounce positions
- [ ] `calculate_shot_speed()` returns speeds in km/h
- [ ] Shot speeds are realistic (10-200 km/h range)
- [ ] `calculate_player_speed()` works for any frame range
- [ ] `calculate_total_distance()` accumulates correctly
- [ ] `generate_match_statistics()` creates complete stats dict
- [ ] `export_statistics_json()` creates valid JSON files
- [ ] All unit tests pass
- [ ] Example script runs without errors

## Integration Points

**Inputs Required from Other Modules:**
- Ball positions from `trackers/ball_tracker.py`
- Player positions from `trackers/player_tracker.py`
- Pixel-to-meter ratio from `mini_court/mini_court.py`
- FPS from video metadata (`utils/video_utils.py`)

**Outputs Provided to Other Modules:**
- Shot events for visualization overlay
- Bounce positions for heatmap generation (`heatmaps/ball_bounce_heatmap.py`)
- Statistics for output rendering
- JSON file for external analysis

## Common Issues & Solutions

**Issue 1: Too many false positive shots detected**
- **Cause:** MIN_SUSTAINED_FRAMES too low
- **Solution:** Increase to 30-35 frames for stricter detection

**Issue 2: Unrealistic shot speeds (>200 km/h)**
- **Cause:** Incorrect pixel-to-meter ratio or ball detection errors
- **Solution:** Validate court analysis module output and ball interpolation

**Issue 3: Player not assigned to shots**
- **Cause:** MAX_PLAYER_DISTANCE_PIXELS too small
- **Solution:** Increase to 250-300 pixels for wider camera angles

**Issue 4: Distance calculation seems too high**
- **Cause:** Accumulating small camera movements/jitter
- **Solution:** Apply smoothing filter or increase minimum movement threshold

**Issue 5: Missing shots at start/end of video**
- **Cause:** Not enough frames after shot for speed calculation
- **Solution:** Reduce FRAME_OFFSET_SPEED_CALC for edge cases

---

**End of Sub-PRD 5**
