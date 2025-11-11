"""Unit tests for BallTracker class."""

import pytest
import numpy as np
import os
from trackers.ball_tracker import BallTracker


@pytest.fixture
def ball_tracker():
    """Fixture to create BallTracker instance."""
    # Use generic YOLOv8n model for testing
    # In production, use fine-tuned model
    model_path = 'yolov8n.pt'
    return BallTracker(model_path=model_path)


def test_ball_tracker_initialization(ball_tracker):
    """Test tracker initializes with model."""
    assert ball_tracker.model is not None
    assert hasattr(ball_tracker, 'detect_frame')
    assert hasattr(ball_tracker, 'detect_frames')
    assert hasattr(ball_tracker, 'interpolate_ball_positions')
    assert hasattr(ball_tracker, 'get_ball_shot_frames')
    assert hasattr(ball_tracker, 'draw_bounding_boxes')


def test_detect_frame_with_ball(ball_tracker):
    """Test detecting ball in frame."""
    # Create dummy frame (black image)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Run detection
    detections = ball_tracker.detect_frame(frame)

    # Check return type
    assert isinstance(detections, dict)
    # Note: Empty frame unlikely to have detections, but test should not crash


def test_detect_frame_returns_correct_format(ball_tracker):
    """Test detection returns correct dictionary format."""
    # Create dummy frame with some content
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Run detection
    detections = ball_tracker.detect_frame(frame)

    # Check return type
    assert isinstance(detections, dict)

    # If detection found, check format
    if len(detections) > 0:
        assert 1 in detections
        assert len(detections[1]) == 4  # [x1, y1, x2, y2]


def test_interpolation_fills_gaps(ball_tracker):
    """Test interpolation fills missing detections."""
    # Mock detections with gaps
    ball_detections = [
        {1: [100.0, 200.0, 120.0, 220.0]},  # Frame 0: detected
        {},                                   # Frame 1: missing
        {},                                   # Frame 2: missing
        {1: [140.0, 240.0, 160.0, 260.0]}    # Frame 3: detected
    ]

    # Interpolate
    interpolated = ball_tracker.interpolate_ball_positions(ball_detections)

    # Check all frames now have detections
    assert all(len(d) > 0 for d in interpolated)

    # Check frame 1 is approximately halfway between frame 0 and 3
    frame1_bbox = interpolated[1][1]
    assert 100 < frame1_bbox[0] < 140  # x1 between 100 and 140
    assert 200 < frame1_bbox[1] < 240  # y1 between 200 and 240


def test_interpolation_handles_all_empty(ball_tracker):
    """Test interpolation handles all empty detections."""
    # All frames have no detections
    ball_detections = [{} for _ in range(10)]

    # Should not crash
    interpolated = ball_tracker.interpolate_ball_positions(ball_detections)
    assert len(interpolated) == 10


def test_interpolation_handles_single_detection(ball_tracker):
    """Test interpolation with only one detection."""
    # Only first frame has detection
    ball_detections = [
        {1: [100.0, 200.0, 120.0, 220.0]},  # Frame 0: detected
        {},                                   # Frame 1: missing
        {},                                   # Frame 2: missing
    ]

    # Interpolate (should back-fill)
    interpolated = ball_tracker.interpolate_ball_positions(ball_detections)

    # All frames should have detections (filled)
    assert all(len(d) > 0 for d in interpolated)


def test_shot_detection(ball_tracker):
    """Test shot frame detection."""
    # Create mock ball trajectory (goes down, then up = shot at frame 50)
    ball_detections = []

    # Frames 0-49: Ball descending (y increases)
    for i in range(50):
        y_pos = 100.0 + i * 2.0
        ball_detections.append({1: [640.0, y_pos, 660.0, y_pos + 20.0]})

    # Frames 50-99: Ball ascending (y decreases)
    for i in range(50):
        y_pos = 200.0 - i * 2.0
        ball_detections.append({1: [640.0, y_pos, 660.0, y_pos + 20.0]})

    # Detect shots
    shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Should detect shot around frame 50
    assert len(shot_frames) > 0
    assert any(45 <= frame <= 55 for frame in shot_frames)  # Within 5 frames of actual shot


def test_shot_detection_no_shots(ball_tracker):
    """Test shot detection with no shots (monotonic trajectory)."""
    # Create monotonically increasing trajectory (no direction changes)
    ball_detections = []

    for i in range(100):
        y_pos = 100.0 + i * 2.0
        ball_detections.append({1: [640.0, y_pos, 660.0, y_pos + 20.0]})

    # Detect shots
    shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Should detect no shots
    assert len(shot_frames) == 0


def test_empty_detections(ball_tracker):
    """Test handling of all empty detections."""
    # All frames have no detections
    ball_detections = [{} for _ in range(100)]

    # Should not crash
    interpolated = ball_tracker.interpolate_ball_positions(ball_detections)
    assert len(interpolated) == 100

    # Shot detection should also handle gracefully
    shot_frames = ball_tracker.get_ball_shot_frames(interpolated)
    assert isinstance(shot_frames, list)


def test_draw_bounding_boxes(ball_tracker):
    """Test drawing bounding boxes on frames."""
    # Create dummy frames
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

    # Create detections
    ball_detections = [
        {1: [100.0, 200.0, 120.0, 220.0]},  # Frame 0: detected
        {},                                   # Frame 1: missing
        {1: [140.0, 240.0, 160.0, 260.0]}    # Frame 2: detected
    ]

    # Draw bounding boxes
    output_frames = ball_tracker.draw_bounding_boxes(frames, ball_detections)

    # Check output
    assert len(output_frames) == len(frames)
    assert all(isinstance(f, np.ndarray) for f in output_frames)
    assert all(f.shape == frames[0].shape for f in output_frames)


def test_draw_bounding_boxes_empty_detections(ball_tracker):
    """Test drawing with no detections."""
    # Create dummy frames
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

    # All empty detections
    ball_detections = [{} for _ in range(3)]

    # Draw bounding boxes (should not crash)
    output_frames = ball_tracker.draw_bounding_boxes(frames, ball_detections)

    # Check output
    assert len(output_frames) == len(frames)


def test_detect_frames_with_cache(ball_tracker, tmp_path):
    """Test detect_frames with caching."""
    # Create dummy frames
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]

    # Create cache path
    cache_path = os.path.join(tmp_path, "test_cache.pkl")

    # First run: detect and cache
    detections1 = ball_tracker.detect_frames(
        frames,
        read_from_stub=False,
        stub_path=cache_path
    )

    # Check cache exists
    assert os.path.exists(cache_path)

    # Second run: load from cache
    detections2 = ball_tracker.detect_frames(
        frames,
        read_from_stub=True,
        stub_path=cache_path
    )

    # Should be identical
    assert len(detections1) == len(detections2)


def test_multiple_shots_detection(ball_tracker):
    """Test detection of multiple shots in sequence."""
    ball_detections = []

    # First shot: down then up
    for i in range(40):
        y_pos = 100.0 + i * 2.0
        ball_detections.append({1: [640.0, y_pos, 660.0, y_pos + 20.0]})

    for i in range(40):
        y_pos = 180.0 - i * 2.0
        ball_detections.append({1: [640.0, y_pos, 660.0, y_pos + 20.0]})

    # Second shot: up then down
    for i in range(40):
        y_pos = 100.0 - i * 2.0
        ball_detections.append({1: [640.0, y_pos, 660.0, y_pos + 20.0]})

    for i in range(40):
        y_pos = 20.0 + i * 2.0
        ball_detections.append({1: [640.0, y_pos, 660.0, y_pos + 20.0]})

    # Detect shots
    shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Should detect at least 2 shots
    assert len(shot_frames) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
