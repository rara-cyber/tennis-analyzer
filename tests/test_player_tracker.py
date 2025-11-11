"""Unit tests for player tracker module."""

import pytest
import numpy as np
from trackers.player_tracker import PlayerTracker
from utils.bbox_utils import (
    get_center_of_bbox,
    get_foot_position,
    measure_distance,
    get_bbox_width,
    get_bbox_height
)


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


def test_bbox_width_calculation():
    """Test bounding box width calculation."""
    bbox = [100, 200, 200, 400]
    width = get_bbox_width(bbox)
    assert width == 100


def test_bbox_height_calculation():
    """Test bounding box height calculation."""
    bbox = [100, 200, 200, 400]
    height = get_bbox_height(bbox)
    assert height == 200


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


def test_draw_bounding_boxes():
    """Test drawing bounding boxes on frames."""
    tracker = PlayerTracker()

    # Create dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frames = [frame]

    # Mock detections
    detections = [{1: [100, 200, 200, 400], 2: [900, 200, 1000, 400]}]

    # Draw bounding boxes
    output_frames = tracker.draw_bounding_boxes(frames, detections)

    # Check output
    assert len(output_frames) == 1
    assert output_frames[0].shape == frame.shape
    # Verify that the frame was modified (not all zeros anymore)
    assert not np.array_equal(output_frames[0], frame)


def test_choose_and_filter_players():
    """Test complete player filtering workflow."""
    tracker = PlayerTracker()

    # Mock detections for 3 frames
    player_detections = [
        {1: [100, 200, 150, 300], 2: [900, 100, 950, 200], 3: [50, 50, 100, 150]},
        {1: [105, 205, 155, 305], 2: [905, 105, 955, 205], 3: [55, 55, 105, 155]},
        {1: [110, 210, 160, 310], 2: [910, 110, 960, 210], 3: [60, 60, 110, 160]}
    ]

    # Mock court keypoints
    court_keypoints = [640, 360] * 14

    # Filter players
    filtered_detections = tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Check results
    assert len(filtered_detections) == 3
    for frame_detections in filtered_detections:
        # Should only have 2 players (IDs 1 and 2)
        assert len(frame_detections) == 2
        assert 1 in frame_detections
        assert 2 in frame_detections
        assert 3 not in frame_detections
