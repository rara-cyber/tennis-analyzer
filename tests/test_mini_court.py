"""
Unit Tests for Mini Court

Tests for mini-court visualization and coordinate transformation.
"""
import pytest
import numpy as np
import sys

# Add parent directory to path
sys.path.append('..')

from mini_court import MiniCourt
from constants import MINI_COURT_WIDTH, MINI_COURT_HEIGHT


@pytest.fixture
def dummy_frame():
    """Create a dummy frame for testing."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def mini_court(dummy_frame):
    """Create MiniCourt instance."""
    return MiniCourt(dummy_frame)


def test_mini_court_initialization(mini_court):
    """Test mini-court initializes with frame."""
    assert mini_court.drawing_rectangle_width == MINI_COURT_WIDTH
    assert mini_court.drawing_rectangle_height == MINI_COURT_HEIGHT
    assert mini_court.buffer == 50
    assert mini_court.padding_court == 20


def test_drawing_keypoints_count(mini_court):
    """Test that 28 keypoints are initialized (14 points × 2 coordinates)."""
    assert len(mini_court.drawing_keypoints) == 28


def test_court_lines_defined(mini_court):
    """Test that court lines are defined."""
    assert len(mini_court.lines) > 0
    # Should have at least the major lines
    assert len(mini_court.lines) >= 10


def test_keypoint_coordinates_valid(mini_court):
    """Test that all keypoint coordinates are valid numbers."""
    for kp in mini_court.drawing_keypoints:
        assert isinstance(kp, (int, float))
        assert not np.isnan(kp)
        assert not np.isinf(kp)


def test_get_keypoint_position(mini_court):
    """Test getting keypoint positions."""
    for i in range(14):  # 14 keypoints
        position = mini_court.get_keypoint_position(i)
        assert isinstance(position, tuple)
        assert len(position) == 2
        assert isinstance(position[0], int)
        assert isinstance(position[1], int)


def test_draw_court_returns_frame(mini_court, dummy_frame):
    """Test draw_court returns a frame."""
    frame_with_court = mini_court.draw_court(dummy_frame)

    assert isinstance(frame_with_court, np.ndarray)
    assert frame_with_court.shape == dummy_frame.shape


def test_draw_court_modifies_frame(mini_court, dummy_frame):
    """Test draw_court actually modifies the frame."""
    frame_with_court = mini_court.draw_court(dummy_frame)

    # Should not be identical to original (court was drawn)
    assert not np.array_equal(frame_with_court, dummy_frame)


def test_coordinate_conversion_basic(mini_court):
    """Test basic coordinate conversion."""
    # Mock conversion - position at a keypoint should return close to keypoint position
    position = (640, 360)
    closest_keypoint = (640, 360)  # Same as position
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
    assert isinstance(mini_pos[0], int)
    assert isinstance(mini_pos[1], int)


def test_coordinate_conversion_offset(mini_court):
    """Test coordinate conversion with offset position."""
    # Position offset from keypoint
    position = (700, 400)
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

    # Should return valid coordinates
    assert isinstance(mini_pos, tuple)
    assert len(mini_pos) == 2


def test_draw_points_on_mini_court(mini_court, dummy_frame):
    """Test drawing points on mini-court."""
    # First draw the court
    frame_with_court = mini_court.draw_court(dummy_frame)

    # Create some positions
    positions = {
        1: (mini_court.court_start_x + 50, mini_court.court_start_y + 100),
        2: (mini_court.court_start_x + 100, mini_court.court_start_y + 200)
    }

    # Draw points
    frame_with_points = mini_court.draw_points_on_mini_court(
        frame_with_court,
        positions,
        color=(0, 0, 255)
    )

    assert isinstance(frame_with_points, np.ndarray)
    assert frame_with_points.shape == dummy_frame.shape


def test_draw_points_empty_dict(mini_court, dummy_frame):
    """Test drawing with empty positions dict."""
    frame_with_court = mini_court.draw_court(dummy_frame)
    frame_with_points = mini_court.draw_points_on_mini_court(
        frame_with_court,
        {},
        color=(0, 0, 255)
    )

    # Should return frame unchanged
    assert np.array_equal(frame_with_points, frame_with_court)


def test_mini_court_position_on_frame(mini_court, dummy_frame):
    """Test that mini-court is positioned correctly on frame."""
    height, width = dummy_frame.shape[:2]

    # Check mini-court is within frame bounds
    assert 0 <= mini_court.start_x < width
    assert 0 <= mini_court.start_y < height
    assert 0 < mini_court.end_x <= width
    assert 0 < mini_court.end_y <= height

    # Check it's in top-right area
    assert mini_court.end_x == width - mini_court.buffer


def test_court_boundaries(mini_court):
    """Test court boundaries are properly calculated."""
    # Court should be inside the drawing rectangle
    assert mini_court.court_start_x == mini_court.start_x + mini_court.padding_court
    assert mini_court.court_start_y == mini_court.start_y + mini_court.padding_court
    assert mini_court.court_end_x == mini_court.end_x - mini_court.padding_court
    assert mini_court.court_end_y == mini_court.end_y - mini_court.padding_court


def test_background_rectangle(mini_court, dummy_frame):
    """Test drawing background rectangle."""
    frame_with_bg = mini_court.draw_background_rectangle(dummy_frame)

    assert isinstance(frame_with_bg, np.ndarray)
    assert frame_with_bg.shape == dummy_frame.shape
    assert not np.array_equal(frame_with_bg, dummy_frame)


def test_different_frame_sizes():
    """Test mini-court works with different frame sizes."""
    sizes = [(480, 640, 3), (1080, 1920, 3), (720, 1280, 3)]

    for size in sizes:
        frame = np.zeros(size, dtype=np.uint8)
        mc = MiniCourt(frame)

        assert mc.drawing_rectangle_width == MINI_COURT_WIDTH
        assert mc.drawing_rectangle_height == MINI_COURT_HEIGHT

        # Should be able to draw court
        frame_with_court = mc.draw_court(frame)
        assert frame_with_court.shape == frame.shape


def test_keypoint_positions_within_court(mini_court):
    """Test that all keypoints are within the court drawing area."""
    for i in range(0, 28, 2):
        x = mini_court.drawing_keypoints[i]
        y = mini_court.drawing_keypoints[i + 1]

        # Should be within court boundaries (with small tolerance)
        assert mini_court.court_start_x - 1 <= x <= mini_court.court_end_x + 1
        assert mini_court.court_start_y - 1 <= y <= mini_court.court_end_y + 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
