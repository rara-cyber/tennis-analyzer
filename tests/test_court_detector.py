"""
Unit Tests for Court Line Detector

Tests for court keypoint detection functionality.
"""
import pytest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append('..')

from court_line_detector import CourtLineDetector


# Model path for testing
MODEL_PATH = 'models/court_keypoint_model.pth'


@pytest.fixture
def dummy_frame():
    """Create a dummy frame for testing."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def detector():
    """Create detector instance if model exists."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}. Run download_models.py first.")
    return CourtLineDetector(model_path=MODEL_PATH)


def test_court_detector_initialization(detector):
    """Test detector initializes with model."""
    assert detector.model is not None
    assert detector.transforms is not None


def test_predict_returns_28_values(detector, dummy_frame):
    """Test prediction returns 28 floats."""
    # Predict
    keypoints = detector.predict(dummy_frame)

    # Check return type and length
    assert isinstance(keypoints, list)
    assert len(keypoints) == 28


def test_predict_returns_floats(detector, dummy_frame):
    """Test all keypoint values are floats."""
    keypoints = detector.predict(dummy_frame)

    for kp in keypoints:
        assert isinstance(kp, (float, int))


def test_keypoints_within_frame(detector, dummy_frame):
    """Test keypoints are within frame boundaries."""
    frame = dummy_frame
    keypoints = detector.predict(frame)

    height, width = frame.shape[:2]

    # Check all x coordinates within width
    for i in range(0, 28, 2):
        # Allow some margin for edge cases
        assert -100 <= keypoints[i] <= width + 100, \
            f"X coordinate {keypoints[i]} out of bounds [0, {width}]"

    # Check all y coordinates within height
    for i in range(1, 28, 2):
        assert -100 <= keypoints[i] <= height + 100, \
            f"Y coordinate {keypoints[i]} out of bounds [0, {height}]"


def test_draw_keypoints_returns_frame(detector, dummy_frame):
    """Test draw_keypoints returns a frame."""
    keypoints = detector.predict(dummy_frame)
    frame_with_keypoints = detector.draw_keypoints(dummy_frame, keypoints)

    assert isinstance(frame_with_keypoints, np.ndarray)
    assert frame_with_keypoints.shape == dummy_frame.shape


def test_draw_keypoints_does_not_modify_original(detector, dummy_frame):
    """Test draw_keypoints doesn't modify original frame."""
    keypoints = detector.predict(dummy_frame)
    original_copy = dummy_frame.copy()

    detector.draw_keypoints(dummy_frame, keypoints)

    # Check original frame is unchanged
    assert np.array_equal(dummy_frame, original_copy)


def test_draw_keypoints_on_video(detector):
    """Test drawing keypoints on multiple frames."""
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    keypoints = detector.predict(frames[0])

    output_frames = detector.draw_keypoints_on_video(frames, keypoints)

    assert len(output_frames) == len(frames)
    assert all(isinstance(f, np.ndarray) for f in output_frames)


def test_model_in_eval_mode(detector):
    """Test model is in evaluation mode."""
    assert not detector.model.training


def test_different_frame_sizes(detector):
    """Test detector works with different frame sizes."""
    sizes = [(480, 640, 3), (1080, 1920, 3), (720, 1280, 3)]

    for size in sizes:
        frame = np.zeros(size, dtype=np.uint8)
        keypoints = detector.predict(frame)

        assert len(keypoints) == 28

        # Check coordinates are scaled to frame size
        for i in range(0, 28, 2):
            assert -100 <= keypoints[i] <= size[1] + 100
        for i in range(1, 28, 2):
            assert -100 <= keypoints[i] <= size[0] + 100


# Test for missing model file
def test_missing_model_file():
    """Test handling of missing model file."""
    if os.path.exists(MODEL_PATH):
        pytest.skip("Model exists, skipping missing model test")

    with pytest.raises(Exception):
        CourtLineDetector(model_path='nonexistent_model.pth')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
