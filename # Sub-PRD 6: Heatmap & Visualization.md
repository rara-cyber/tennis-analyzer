# Sub-PRD 6: Heatmap & Visualization

**Module Name:** `heatmaps` and `utils/visualization`
**Claude Code Instruction:** "Implement this complete heatmap generation and visualization module following all specifications below."

---

## Module Overview

Create Python modules that generate heatmap visualizations from position data and render annotated output videos with all overlays. This module transforms raw analytics data into visual insights including density heatmaps, annotated videos with bounding boxes, mini-court visualizations, and statistics overlays.

## What You're Building

A visualization system that:
- Generates ball bounce heatmaps showing shot distribution
- Creates player position heatmaps showing court coverage
- Renders annotated videos with bounding boxes and labels
- Draws mini-court overlays with real-time positions
- Displays statistics panels on output videos
- Exports high-resolution heatmap images

## File Structure to Create

```
heatmaps/
├── __init__.py
├── ball_bounce_heatmap.py    # Ball bounce density heatmap
├── player_position_heatmap.py # Player position density heatmap
└── heatmap_stats.py           # Heatmap statistical analysis

utils/
├── visualization.py           # Drawing functions for video annotation
└── bbox_utils.py              # Bounding box helper functions

tests/
├── test_heatmap_generation.py
└── test_visualization.py
```

## Requirements

### REQ-1: Ball Bounce Heatmap Generation

**Function Signature:**
```python
def generate_ball_bounce_heatmap(
    bounce_positions: list[tuple[float, float]],
    mini_court_dimensions: tuple[int, int] = (500, 1000),
    output_path: str = None,
    colormap: str = 'hot'
) -> np.ndarray:
    """
    Generate heatmap showing ball bounce distribution on court.

    Args:
        bounce_positions: List of (x, y) positions in mini-court coordinates
        mini_court_dimensions: (width, height) of mini-court in pixels
        output_path: If provided, saves heatmap as PNG to this path
        colormap: Matplotlib colormap name ('hot', 'viridis', 'jet', etc.)

    Returns:
        Heatmap image as numpy array (height, width, 3) in RGB format
    """
```

**Implementation Details:**

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

def generate_ball_bounce_heatmap(
    bounce_positions,
    mini_court_dimensions=(500, 1000),
    output_path=None,
    colormap='hot'
):
    """Generate ball bounce heatmap."""
    width, height = mini_court_dimensions

    # Create empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Add each bounce position with Gaussian kernel
    sigma = 15  # Standard deviation for Gaussian (controls blur radius)

    for x, y in bounce_positions:
        # Ensure position is within bounds
        x_int = int(np.clip(x, 0, width - 1))
        y_int = int(np.clip(y, 0, height - 1))

        # Add point to heatmap
        heatmap[y_int, x_int] += 1

    # Apply Gaussian blur for smooth density
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Normalize to 0-1 range
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)  # Returns RGBA

    # Convert to RGB (0-255)
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    # Draw court outline
    heatmap_with_court = draw_court_outline(heatmap_rgb, width, height)

    # Save if output path provided
    if output_path:
        save_heatmap_image(heatmap_with_court, output_path, title="Ball Bounce Distribution")

    return heatmap_with_court
```

**Helper Function: Draw Court Outline**
```python
def draw_court_outline(heatmap_img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Draw tennis court lines on heatmap."""
    # Create copy to avoid modifying original
    img = heatmap_img.copy()

    # Court line color (white)
    color = (255, 255, 255)
    thickness = 2

    # Outer boundary
    cv2.rectangle(img, (20, 20), (width-20, height-20), color, thickness)

    # Service lines (horizontal)
    service_line_y = height // 4
    cv2.line(img, (20, service_line_y), (width-20, service_line_y), color, thickness)
    cv2.line(img, (20, height - service_line_y), (width-20, height - service_line_y), color, thickness)

    # Center line (horizontal)
    center_y = height // 2
    cv2.line(img, (20, center_y), (width-20, center_y), color, thickness)

    # Center service line (vertical)
    center_x = width // 2
    cv2.line(img, (center_x, service_line_y), (center_x, height - service_line_y), color, thickness)

    return img
```

**Acceptance Criteria:**
- Generates heatmap using Gaussian kernel density estimation
- Color scheme: blue (low) → green → yellow → red (high)
- Court lines overlaid in white with 2px thickness
- Output resolution: 500x1000 pixels (scalable)
- Normalizes density to match video duration
- Exports as PNG with transparent background option

### REQ-2: Player Position Heatmap Generation

**Function Signature:**
```python
def generate_player_position_heatmap(
    player_positions: list[tuple[float, float]],
    player_id: int,
    mini_court_dimensions: tuple[int, int] = (500, 1000),
    output_path: str = None,
    color_scheme: str = 'red'
) -> np.ndarray:
    """
    Generate heatmap showing player court coverage.

    Args:
        player_positions: List of (x, y) positions in mini-court coordinates
        player_id: Player ID (1 or 2) for labeling
        mini_court_dimensions: (width, height) of mini-court
        output_path: If provided, saves heatmap as PNG
        color_scheme: 'red' for player 1, 'blue' for player 2, or colormap name

    Returns:
        Heatmap image as numpy array
    """
```

**Implementation Details:**

```python
def generate_player_position_heatmap(
    player_positions,
    player_id,
    mini_court_dimensions=(500, 1000),
    output_path=None,
    color_scheme='red'
):
    """Generate player position heatmap."""
    width, height = mini_court_dimensions

    # Create empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Sample positions every 5 frames to reduce computation
    sampled_positions = player_positions[::5]

    # Add each position with Gaussian kernel
    sigma = 25  # Larger sigma for player positions (they cover more area)

    for x, y in sampled_positions:
        x_int = int(np.clip(x, 0, width - 1))
        y_int = int(np.clip(y, 0, height - 1))
        heatmap[y_int, x_int] += 1

    # Apply Gaussian blur
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Apply color scheme
    if color_scheme == 'red':
        # Red gradient: white -> yellow -> red -> dark red
        colors = [(0, 0, 0, 0), (1, 1, 0, 0.4), (1, 0.5, 0, 0.7), (0.8, 0, 0, 1)]
        cmap = LinearSegmentedColormap.from_list('red_gradient', colors)
    elif color_scheme == 'blue':
        # Blue gradient: white -> cyan -> blue -> dark blue
        colors = [(0, 0, 0, 0), (0, 1, 1, 0.4), (0, 0.5, 1, 0.7), (0, 0, 0.8, 1)]
        cmap = LinearSegmentedColormap.from_list('blue_gradient', colors)
    else:
        cmap = plt.get_cmap(color_scheme)

    heatmap_colored = cmap(heatmap)
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    # Draw court outline
    heatmap_with_court = draw_court_outline(heatmap_rgb, width, height)

    # Save if requested
    if output_path:
        title = f"Player {player_id} Court Coverage"
        save_heatmap_image(heatmap_with_court, output_path, title=title)

    return heatmap_with_court
```

**Acceptance Criteria:**
- Samples positions every 5 frames for performance
- Uses larger Gaussian kernel (sigma=25) vs ball bounces
- Color schemes: Player 1 = red gradient, Player 2 = blue gradient
- Overlays court lines in white
- Exports individual and combined heatmaps

### REQ-3: Combined Player Heatmap

**Function Signature:**
```python
def generate_combined_player_heatmap(
    player1_positions: list[tuple[float, float]],
    player2_positions: list[tuple[float, float]],
    mini_court_dimensions: tuple[int, int] = (500, 1000),
    output_path: str = None
) -> np.ndarray:
    """
    Generate combined heatmap showing both players' coverage.

    Args:
        player1_positions: Player 1 positions list
        player2_positions: Player 2 positions list
        mini_court_dimensions: (width, height)
        output_path: Save path

    Returns:
        Combined heatmap with red (P1), blue (P2), purple (overlap)
    """
```

**Implementation Details:**

```python
def generate_combined_player_heatmap(
    player1_positions,
    player2_positions,
    mini_court_dimensions=(500, 1000),
    output_path=None
):
    """Generate combined player heatmap."""
    # Generate individual heatmaps
    heatmap1 = generate_player_position_heatmap(
        player1_positions, 1, mini_court_dimensions, output_path=None, color_scheme='red'
    )
    heatmap2 = generate_player_position_heatmap(
        player2_positions, 2, mini_court_dimensions, output_path=None, color_scheme='blue'
    )

    # Blend heatmaps with transparency
    alpha = 0.5  # 50% transparency
    combined = cv2.addWeighted(heatmap1, alpha, heatmap2, alpha, 0)

    # Draw court outline on combined
    width, height = mini_court_dimensions
    combined = draw_court_outline(combined, width, height)

    if output_path:
        save_heatmap_image(combined, output_path, title="Combined Court Coverage")

    return combined
```

**Acceptance Criteria:**
- Blends both player heatmaps with 50% opacity
- Purple areas indicate overlap (both players present)
- Court lines visible on combined image
- Legend indicates Player 1 (red) and Player 2 (blue)

### REQ-4: Heatmap Statistical Summary

**Function Signature:**
```python
def calculate_heatmap_statistics(
    positions: list[tuple[float, float]],
    mini_court_dimensions: tuple[int, int] = (500, 1000)
) -> dict:
    """
    Calculate quantitative statistics from position data.

    Args:
        positions: List of (x, y) positions
        mini_court_dimensions: (width, height)

    Returns:
        Dict with keys:
        - 'court_coverage_percentage': float (0-100)
        - 'most_frequented_zone': str (e.g., 'baseline_center')
        - 'zone_distribution': dict mapping zone names to percentages
    """
```

**Implementation Details:**

```python
def calculate_heatmap_statistics(positions, mini_court_dimensions=(500, 1000)):
    """Calculate heatmap statistics."""
    width, height = mini_court_dimensions

    # Define court zones
    zones = {
        'baseline_left': (0, 0, width//3, height//3),
        'baseline_center': (width//3, 0, 2*width//3, height//3),
        'baseline_right': (2*width//3, 0, width, height//3),
        'midcourt_left': (0, height//3, width//3, 2*height//3),
        'midcourt_center': (width//3, height//3, 2*width//3, 2*height//3),
        'midcourt_right': (2*width//3, height//3, width, 2*height//3),
        'net_left': (0, 2*height//3, width//3, height),
        'net_center': (width//3, 2*height//3, 2*width//3, height),
        'net_right': (2*width//3, 2*height//3, width, height)
    }

    # Count positions in each zone
    zone_counts = {zone: 0 for zone in zones.keys()}

    for x, y in positions:
        for zone_name, (x1, y1, x2, y2) in zones.items():
            if x1 <= x < x2 and y1 <= y < y2:
                zone_counts[zone_name] += 1
                break

    total_positions = len(positions)

    # Calculate percentages
    zone_distribution = {
        zone: (count / total_positions * 100) if total_positions > 0 else 0
        for zone, count in zone_counts.items()
    }

    # Find most frequented zone
    most_frequented_zone = max(zone_counts, key=zone_counts.get)

    # Calculate court coverage (percentage of zones with >1% presence)
    zones_covered = sum(1 for pct in zone_distribution.values() if pct > 1)
    court_coverage_percentage = (zones_covered / len(zones)) * 100

    return {
        'court_coverage_percentage': round(court_coverage_percentage, 1),
        'most_frequented_zone': most_frequented_zone,
        'zone_distribution': {k: round(v, 1) for k, v in zone_distribution.items()}
    }
```

**Acceptance Criteria:**
- Divides court into 9 zones (3x3 grid)
- Calculates time spent in each zone as percentage
- Identifies most frequented zone
- Court coverage = percentage of zones with >1% presence
- All percentages sum to 100%

### REQ-5: Save Heatmap Image

**Function Signature:**
```python
def save_heatmap_image(
    heatmap_img: np.ndarray,
    output_path: str,
    title: str = "Heatmap",
    dpi: int = 150
) -> None:
    """
    Save heatmap as high-resolution PNG with title.

    Args:
        heatmap_img: Heatmap image array
        output_path: File path to save (should end in .png)
        title: Title text to display above heatmap
        dpi: Resolution in dots per inch
    """
```

**Implementation Details:**

```python
import os
import matplotlib.pyplot as plt

def save_heatmap_image(heatmap_img, output_path, title="Heatmap", dpi=150):
    """Save heatmap with title as PNG."""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 12))  # 1:2 aspect ratio for tennis court

    # Display heatmap
    ax.imshow(heatmap_img)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    # Save with high resolution
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"✓ Heatmap saved: {output_path}")
```

**Acceptance Criteria:**
- Saves as PNG format
- Resolution: 150 DPI (default), configurable
- Includes title above heatmap
- No axis labels or ticks
- Creates output directory if needed

### REQ-6: Draw Bounding Boxes

**Function Signature:**
```python
def draw_bounding_boxes(
    frame: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    label: str = None
) -> np.ndarray:
    """
    Draw bounding boxes on frame.

    Args:
        frame: Video frame (np.ndarray)
        detections: List of dicts with 'bbox' key: [x_min, y_min, x_max, y_max]
        color: RGB color tuple
        thickness: Line thickness in pixels
        label: Optional label text to display

    Returns:
        Frame with bounding boxes drawn
    """
```

**Implementation Details:**

```python
def draw_bounding_boxes(frame, detections, color=(255, 0, 0), thickness=2, label=None):
    """Draw bounding boxes on frame."""
    frame_copy = frame.copy()

    for detection in detections:
        bbox = detection['bbox']
        x_min, y_min, x_max, y_max = bbox

        # Convert to integers
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)

        # Draw rectangle
        cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), color, thickness)

        # Draw label if provided
        if label:
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

            # Draw background rectangle for text
            cv2.rectangle(
                frame_copy,
                (x_min, y_min - text_height - 10),
                (x_min + text_width, y_min),
                color,
                -1  # Filled rectangle
            )

            # Draw text
            cv2.putText(
                frame_copy,
                label,
                (x_min, y_min - 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                text_thickness,
                cv2.LINE_AA
            )

    return frame_copy
```

**Acceptance Criteria:**
- Draws rectangles with specified color and thickness
- Labels appear above bounding box with background
- Text is white on colored background
- Does not modify original frame (returns copy)

### REQ-7: Draw Mini-Court Overlay

**Function Signature:**
```python
def draw_mini_court_overlay(
    frame: np.ndarray,
    player_positions: dict[int, tuple[float, float]],
    ball_position: tuple[float, float],
    position: tuple[int, int] = (50, 50),
    size: tuple[int, int] = (250, 500)
) -> np.ndarray:
    """
    Draw mini-court visualization on video frame.

    Args:
        frame: Video frame
        player_positions: Dict mapping player_id to (x, y) in mini-court coords
        ball_position: (x, y) ball position in mini-court coords
        position: (x, y) position on frame where mini-court appears
        size: (width, height) of mini-court overlay

    Returns:
        Frame with mini-court overlay
    """
```

**Implementation Details:**

```python
def draw_mini_court_overlay(frame, player_positions, ball_position, position=(50, 50), size=(250, 500)):
    """Draw mini-court overlay on frame."""
    frame_copy = frame.copy()
    x_offset, y_offset = position
    width, height = size

    # Create mini-court background (white rectangle)
    cv2.rectangle(
        frame_copy,
        (x_offset, y_offset),
        (x_offset + width, y_offset + height),
        (255, 255, 255),
        -1  # Filled
    )

    # Draw court lines (black)
    line_color = (0, 0, 0)
    line_thickness = 1

    # Outer boundary
    cv2.rectangle(
        frame_copy,
        (x_offset + 10, y_offset + 10),
        (x_offset + width - 10, y_offset + height - 10),
        line_color,
        line_thickness
    )

    # Service lines
    service_y1 = y_offset + height // 4
    service_y2 = y_offset + 3 * height // 4
    cv2.line(frame_copy, (x_offset + 10, service_y1), (x_offset + width - 10, service_y1), line_color, line_thickness)
    cv2.line(frame_copy, (x_offset + 10, service_y2), (x_offset + width - 10, service_y2), line_color, line_thickness)

    # Center line
    center_y = y_offset + height // 2
    cv2.line(frame_copy, (x_offset + 10, center_y), (x_offset + width - 10, center_y), line_color, line_thickness)

    # Center service line
    center_x = x_offset + width // 2
    cv2.line(frame_copy, (center_x, service_y1), (center_x, service_y2), line_color, line_thickness)

    # Draw player positions
    for player_id, (px, py) in player_positions.items():
        # Scale position to mini-court size
        scaled_x = int(x_offset + 10 + (px / 500) * (width - 20))
        scaled_y = int(y_offset + 10 + (py / 1000) * (height - 20))

        # Player 1 = red, Player 2 = blue
        color = (0, 0, 255) if player_id == 1 else (255, 0, 0)
        cv2.circle(frame_copy, (scaled_x, scaled_y), 5, color, -1)

    # Draw ball position (yellow)
    if ball_position:
        bx, by = ball_position
        scaled_bx = int(x_offset + 10 + (bx / 500) * (width - 20))
        scaled_by = int(y_offset + 10 + (by / 1000) * (height - 20))
        cv2.circle(frame_copy, (scaled_bx, scaled_by), 4, (0, 255, 255), -1)

    return frame_copy
```

**Acceptance Criteria:**
- Mini-court positioned in top-right corner (default)
- White background with black court lines
- Player 1 = red dot, Player 2 = blue dot
- Ball = yellow dot
- Size: 250x500 pixels (configurable)

### REQ-8: Draw Statistics Overlay

**Function Signature:**
```python
def draw_statistics_overlay(
    frame: np.ndarray,
    statistics: dict,
    position: tuple[int, int] = None,
    size: tuple[int, int] = (350, 230)
) -> np.ndarray:
    """
    Draw statistics panel on video frame.

    Args:
        frame: Video frame
        statistics: Dict with player stats (shot speeds, distances, etc.)
        position: (x, y) position (default: bottom-right)
        size: (width, height) of stats panel

    Returns:
        Frame with statistics overlay
    """
```

**Implementation Details:**

```python
def draw_statistics_overlay(frame, statistics, position=None, size=(350, 230)):
    """Draw statistics overlay on frame."""
    frame_copy = frame.copy()
    height, width = frame.shape[:2]

    # Default position: bottom-right
    if position is None:
        x_offset = width - size[0] - 50
        y_offset = height - size[1] - 50
    else:
        x_offset, y_offset = position

    # Draw semi-transparent background
    overlay = frame_copy.copy()
    cv2.rectangle(
        overlay,
        (x_offset, y_offset),
        (x_offset + size[0], y_offset + size[1]),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.5, frame_copy, 0.5, 0, frame_copy)

    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    color = (255, 255, 255)
    line_height = 25

    # Title
    cv2.putText(
        frame_copy,
        "Match Statistics",
        (x_offset + 10, y_offset + 25),
        font,
        0.6,
        color,
        2,
        cv2.LINE_AA
    )

    # Player statistics
    y = y_offset + 55

    # Headers
    cv2.putText(frame_copy, "Player 1", (x_offset + 10, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame_copy, "Player 2", (x_offset + 190, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    y += line_height

    # Shot speed
    p1_shot_speed = statistics.get('player_1', {}).get('last_shot_speed', 0)
    p2_shot_speed = statistics.get('player_2', {}).get('last_shot_speed', 0)
    cv2.putText(frame_copy, f"Shot: {p1_shot_speed:.1f} km/h", (x_offset + 10, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Shot: {p2_shot_speed:.1f} km/h", (x_offset + 190, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    y += line_height

    # Average shot speed
    p1_avg = statistics.get('player_1', {}).get('average_shot_speed_kmh', 0)
    p2_avg = statistics.get('player_2', {}).get('average_shot_speed_kmh', 0)
    cv2.putText(frame_copy, f"Avg: {p1_avg:.1f} km/h", (x_offset + 10, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Avg: {p2_avg:.1f} km/h", (x_offset + 190, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    y += line_height

    # Shot count
    p1_shots = statistics.get('player_1', {}).get('total_shots', 0)
    p2_shots = statistics.get('player_2', {}).get('total_shots', 0)
    cv2.putText(frame_copy, f"Shots: {p1_shots}", (x_offset + 10, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Shots: {p2_shots}", (x_offset + 190, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    y += line_height

    # Distance covered
    p1_dist = statistics.get('player_1', {}).get('total_distance_covered_meters', 0)
    p2_dist = statistics.get('player_2', {}).get('total_distance_covered_meters', 0)
    cv2.putText(frame_copy, f"Dist: {p1_dist:.0f} m", (x_offset + 10, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Dist: {p2_dist:.0f} m", (x_offset + 190, y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    return frame_copy
```

**Acceptance Criteria:**
- Semi-transparent black background (50% opacity)
- White text with anti-aliasing
- Position: bottom-right corner (default)
- Size: 350x230 pixels
- Displays: shot speeds, average speeds, shot counts, distance

### REQ-9: Render Annotated Video

**Function Signature:**
```python
def render_annotated_video(
    frames: list[np.ndarray],
    player_detections: dict[int, list[dict]],
    ball_detections: list[dict],
    statistics: dict,
    output_path: str,
    fps: float = 24.0
) -> None:
    """
    Render complete annotated video with all overlays.

    Args:
        frames: List of video frames
        player_detections: Dict mapping player_id to detection list
        ball_detections: List of ball detections
        statistics: Match statistics dict
        output_path: Path to save annotated video
        fps: Frame rate
    """
```

**Implementation Details:**

```python
from utils.video_utils import save_video, display_progress
import time

def render_annotated_video(frames, player_detections, ball_detections, statistics, output_path, fps=24.0):
    """Render annotated video with all overlays."""
    print("\n[7/7] Rendering output video...")

    annotated_frames = []
    start_time = time.time()

    for frame_idx, frame in enumerate(frames):
        # Start with original frame
        annotated = frame.copy()

        # Draw player bounding boxes
        for player_id, detections in player_detections.items():
            # Find detection for this frame
            frame_detections = [d for d in detections if d['frame'] == frame_idx]
            if frame_detections:
                color = (0, 0, 255) if player_id == 1 else (255, 0, 0)
                annotated = draw_bounding_boxes(
                    annotated,
                    frame_detections,
                    color=color,
                    label=f"Player {player_id}"
                )

        # Draw ball bounding box
        ball_frame_detections = [d for d in ball_detections if d['frame'] == frame_idx]
        if ball_frame_detections:
            annotated = draw_bounding_boxes(
                annotated,
                ball_frame_detections,
                color=(0, 255, 255),
                label="Ball"
            )

        # Draw mini-court overlay
        player_positions = {}
        for player_id, detections in player_detections.items():
            det = next((d for d in detections if d['frame'] == frame_idx), None)
            if det:
                player_positions[player_id] = (det['x'], det['y'])

        ball_position = None
        if ball_frame_detections:
            ball_position = (ball_frame_detections[0]['x'], ball_frame_detections[0]['y'])

        h, w = annotated.shape[:2]
        annotated = draw_mini_court_overlay(
            annotated,
            player_positions,
            ball_position,
            position=(w - 300, 50)
        )

        # Draw statistics overlay
        annotated = draw_statistics_overlay(annotated, statistics)

        # Draw frame number
        cv2.putText(
            annotated,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        annotated_frames.append(annotated)

        # Display progress
        display_progress(frame_idx + 1, len(frames), prefix="  Rendering", start_time=start_time)

    # Save video
    print(f"\n  Saving to {output_path}...")
    save_video(annotated_frames, output_path, fps=fps)
    print(f"✓ Video saved: {output_path}")
```

**Acceptance Criteria:**
- Draws all overlays on each frame in order
- Progress bar shows rendering progress
- Output video playable in standard players
- Same resolution and frame rate as input
- All visual elements visible and not overlapping

## Constants to Define

**File:** `heatmaps/__init__.py`

```python
# Heatmap generation
HEATMAP_RESOLUTION = (500, 1000)  # (width, height) in pixels
BALL_BOUNCE_SIGMA = 15  # Gaussian blur radius for ball bounces
PLAYER_POSITION_SIGMA = 25  # Gaussian blur radius for player positions
POSITION_SAMPLING_RATE = 5  # Sample every N frames for player heatmaps

# Color schemes
BALL_HEATMAP_COLORMAP = 'hot'  # Matplotlib colormap
PLAYER1_COLOR_SCHEME = 'red'
PLAYER2_COLOR_SCHEME = 'blue'

# Export settings
HEATMAP_DPI = 150  # Resolution for saved images
HEATMAP_OUTPUT_FORMAT = 'png'

# Overlay settings
MINI_COURT_SIZE = (250, 500)  # (width, height)
MINI_COURT_POSITION = (50, 50)  # (x, y) default position
STATS_PANEL_SIZE = (350, 230)
STATS_PANEL_OPACITY = 0.5  # 50% transparent

# Bounding box settings
BBOX_THICKNESS = 2  # pixels
PLAYER1_BBOX_COLOR = (0, 0, 255)  # Red (BGR)
PLAYER2_BBOX_COLOR = (255, 0, 0)  # Blue (BGR)
BALL_BBOX_COLOR = (0, 255, 255)  # Yellow (BGR)
```

## Testing Requirements

**File:** `tests/test_heatmap_generation.py`

```python
import pytest
import numpy as np
from heatmaps.ball_bounce_heatmap import generate_ball_bounce_heatmap
from heatmaps.player_position_heatmap import generate_player_position_heatmap

def test_generate_ball_bounce_heatmap():
    """Test ball bounce heatmap generation."""
    # Create mock bounce positions
    bounce_positions = [
        (250, 500),  # Center of court
        (250, 750),  # Below center
        (250, 250),  # Above center
    ]

    heatmap = generate_ball_bounce_heatmap(bounce_positions, output_path=None)

    # Check output shape
    assert heatmap.shape == (1000, 500, 3)  # (height, width, RGB)

    # Check center has higher intensity than corners
    center_intensity = heatmap[500, 250].sum()
    corner_intensity = heatmap[50, 50].sum()
    assert center_intensity > corner_intensity

def test_generate_player_position_heatmap():
    """Test player position heatmap generation."""
    # Create mock player positions (baseline play)
    player_positions = [(250, 100 + i) for i in range(100)]

    heatmap = generate_player_position_heatmap(
        player_positions,
        player_id=1,
        output_path=None
    )

    assert heatmap.shape == (1000, 500, 3)

    # Check baseline area has high intensity
    baseline_intensity = heatmap[150, 250].sum()
    net_intensity = heatmap[900, 250].sum()
    assert baseline_intensity > net_intensity
```

**File:** `tests/test_visualization.py`

```python
def test_draw_bounding_boxes():
    """Test bounding box drawing."""
    from utils.visualization import draw_bounding_boxes

    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray frame
    detections = [{'bbox': [100, 100, 200, 200]}]

    result = draw_bounding_boxes(frame, detections, color=(255, 0, 0))

    # Check box was drawn (pixels should be red)
    assert result[100, 100].tolist() == [255, 0, 0]

def test_draw_mini_court_overlay():
    """Test mini-court overlay."""
    from utils.visualization import draw_mini_court_overlay

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    player_positions = {1: (250, 500), 2: (250, 100)}
    ball_position = (250, 300)

    result = draw_mini_court_overlay(frame, player_positions, ball_position)

    # Check overlay was added (frame no longer all black)
    assert result.sum() > 0
```

## Usage Example

**File:** `examples/visualization_example.py`

```python
from heatmaps.ball_bounce_heatmap import generate_ball_bounce_heatmap
from heatmaps.player_position_heatmap import generate_player_position_heatmap, generate_combined_player_heatmap
from utils.visualization import render_annotated_video

# Mock data (in real use, comes from trackers and analytics)
bounce_positions = [...]  # From analytics module
player1_positions = [...]  # From player tracker
player2_positions = [...]  # From player tracker

# Generate heatmaps
print("Generating ball bounce heatmap...")
ball_heatmap = generate_ball_bounce_heatmap(
    bounce_positions,
    output_path='output_heatmaps/ball_bounce_heatmap.png'
)

print("Generating player position heatmaps...")
p1_heatmap = generate_player_position_heatmap(
    player1_positions,
    player_id=1,
    output_path='output_heatmaps/player1_position_heatmap.png'
)

p2_heatmap = generate_player_position_heatmap(
    player2_positions,
    player_id=2,
    output_path='output_heatmaps/player2_position_heatmap.png'
)

combined_heatmap = generate_combined_player_heatmap(
    player1_positions,
    player2_positions,
    output_path='output_heatmaps/combined_player_heatmap.png'
)

print("All heatmaps generated!")

# Render annotated video
print("Rendering annotated video...")
frames = [...]  # From video_utils
player_detections = {...}
ball_detections = [...]
statistics = {...}

render_annotated_video(
    frames,
    player_detections,
    ball_detections,
    statistics,
    output_path='output_videos/match_annotated.mp4',
    fps=24.0
)

print("Done!")
```

## Dependencies to Install

Add to `requirements.txt`:
```
matplotlib==3.7.1
scipy==1.10.1
opencv-python==4.8.1.78
numpy==1.24.3
```

Install with:
```bash
pip install matplotlib scipy opencv-python numpy
```

## Validation Checklist

Before marking this module complete, verify:

- [ ] Ball bounce heatmap generates correctly
- [ ] Player position heatmaps use correct color schemes
- [ ] Combined heatmap shows both players with blending
- [ ] Court lines visible on all heatmaps
- [ ] Heatmaps saved as PNG with titles
- [ ] Bounding boxes drawn with correct colors
- [ ] Mini-court overlay positioned correctly
- [ ] Statistics panel displays all metrics
- [ ] Annotated video renders with all overlays
- [ ] All unit tests pass
- [ ] Example script runs without errors

## Integration Points

**Inputs Required from Other Modules:**
- Bounce positions from `analytics/shot_detector.py`
- Player positions from `trackers/player_tracker.py`
- Ball positions from `trackers/ball_tracker.py`
- Match statistics from `analytics/statistics.py`
- Video frames from `utils/video_utils.py`
- Mini-court coordinates from `mini_court/mini_court.py`

**Outputs Provided:**
- Heatmap PNG images (5 files: ball, player1, player2, combined, stats)
- Annotated video with all overlays
- Visual analytics for user reports

## Common Issues & Solutions

**Issue 1: Heatmap appears too blurry**
- **Cause:** Gaussian sigma too large
- **Solution:** Reduce BALL_BOUNCE_SIGMA or PLAYER_POSITION_SIGMA

**Issue 2: Heatmap shows no intensity**
- **Cause:** No positions recorded or positions out of bounds
- **Solution:** Validate position data and check clipping logic

**Issue 3: Video rendering too slow**
- **Cause:** Drawing operations on every frame
- **Solution:** Reduce position sampling rate or use GPU acceleration

**Issue 4: Overlays blocking important video content**
- **Cause:** Overlay positions not optimized for camera angle
- **Solution:** Make positions configurable or adjust transparency

**Issue 5: Colors not matching expectations**
- **Cause:** OpenCV uses BGR, matplotlib uses RGB
- **Solution:** Convert color spaces: `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)`

**Issue 6: Heatmap statistics don't sum to 100%**
- **Cause:** Rounding errors or positions outside court
- **Solution:** Validate all positions within bounds before calculation

---

**End of Sub-PRD 6**
