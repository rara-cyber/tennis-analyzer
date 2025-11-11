"""
Unit tests for video processing utilities.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import cv2

from utils.video_utils import (
    read_video,
    save_video,
    save_cache,
    load_cache,
    cache_exists,
    display_progress,
    draw_frame_number
)


class TestReadVideo:
    """Tests for read_video function."""

    @pytest.fixture
    def test_video_path(self):
        """Create a temporary test video."""
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "test_video.mp4")

        # Create test video (10 frames, 1280x720)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 24.0, (1280, 720))

        for i in range(10):
            # Create frame with different colors
            frame = np.full((720, 1280, 3), i * 25, dtype=np.uint8)
            out.write(frame)

        out.release()

        yield video_path

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def low_res_video_path(self):
        """Create a low resolution test video (below minimum)."""
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "low_res_video.mp4")

        # Create test video (640x480 - below 1280x720 minimum)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 24.0, (640, 480))

        for i in range(5):
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            out.write(frame)

        out.release()

        yield video_path

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_read_video_success(self, test_video_path):
        """Test reading valid video file."""
        frames, metadata = read_video(test_video_path)

        # Assert frame count
        assert len(frames) == 10, f"Expected 10 frames, got {len(frames)}"

        # Assert frame shape (RGB format)
        assert frames[0].shape == (720, 1280, 3), f"Expected shape (720, 1280, 3), got {frames[0].shape}"

        # Assert metadata
        assert abs(metadata['fps'] - 24.0) < 0.1, f"Expected FPS ~24.0, got {metadata['fps']}"
        assert metadata['width'] == 1280
        assert metadata['height'] == 720
        assert metadata['total_frames'] == 10

    def test_read_video_missing_file(self):
        """Test reading non-existent video raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            read_video("nonexistent_video.mp4")

    def test_read_video_low_resolution(self, low_res_video_path):
        """Test reading video below minimum resolution raises ValueError."""
        with pytest.raises(ValueError, match="resolution.*too low"):
            read_video(low_res_video_path)

    def test_read_video_corrupted_file(self):
        """Test reading corrupted video file."""
        temp_dir = tempfile.mkdtemp()
        corrupted_path = os.path.join(temp_dir, "corrupted.mp4")

        # Create a corrupted file (just write random bytes)
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a video file')

        try:
            with pytest.raises(ValueError, match="Unable to open video file"):
                read_video(corrupted_path)
        finally:
            shutil.rmtree(temp_dir)


class TestSaveVideo:
    """Tests for save_video function."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for output videos."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dummy_frames(self):
        """Create dummy frames for testing."""
        frames = []
        for i in range(10):
            # Create 1280x720 frames with different colors
            frame = np.full((720, 1280, 3), i * 25, dtype=np.uint8)
            frames.append(frame)
        return frames

    def test_save_video_success(self, dummy_frames, temp_output_dir):
        """Test saving frames to video file."""
        output_path = os.path.join(temp_output_dir, "output.mp4")

        # Save video
        save_video(dummy_frames, output_path, fps=24.0)

        # Assert file exists
        assert os.path.exists(output_path), "Output video file was not created"

        # Read back and verify
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened(), "Saved video cannot be opened"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        assert frame_count == 10, f"Expected 10 frames in saved video, got {frame_count}"

    def test_save_video_creates_directory(self, dummy_frames, temp_output_dir):
        """Test that save_video creates output directory if it doesn't exist."""
        output_path = os.path.join(temp_output_dir, "subdir", "output.mp4")

        # Save video (directory doesn't exist yet)
        save_video(dummy_frames, output_path)

        # Assert file exists
        assert os.path.exists(output_path), "Output video file was not created"

    def test_save_video_empty_frames(self, temp_output_dir):
        """Test saving empty frame list raises ValueError."""
        output_path = os.path.join(temp_output_dir, "output.mp4")

        with pytest.raises(ValueError, match="empty"):
            save_video([], output_path)

    def test_save_video_inconsistent_dimensions(self, temp_output_dir):
        """Test frames with different sizes raise ValueError."""
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),   # First frame 640x480
            np.zeros((720, 1280, 3), dtype=np.uint8)   # Second frame 1280x720
        ]
        output_path = os.path.join(temp_output_dir, "output.mp4")

        with pytest.raises(ValueError, match="consistent"):
            save_video(frames, output_path)


class TestCaching:
    """Tests for caching functions."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_cache_save_and_load(self, temp_cache_dir):
        """Test saving and loading cache."""
        test_data = {
            "player_1": [[100, 200, 150, 250]],
            "player_2": [[300, 400, 350, 450]]
        }
        cache_path = os.path.join(temp_cache_dir, "test_cache.pkl")

        # Save cache
        save_cache(test_data, cache_path)
        assert cache_exists(cache_path), "Cache file was not created"

        # Load cache
        loaded_data = load_cache(cache_path)
        assert loaded_data == test_data, "Loaded data doesn't match saved data"

    def test_cache_load_nonexistent(self):
        """Test loading non-existent cache raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Cache file not found"):
            load_cache("nonexistent_cache.pkl")

    def test_cache_exists(self, temp_cache_dir):
        """Test cache_exists function."""
        cache_path = os.path.join(temp_cache_dir, "test_cache.pkl")

        # Should not exist initially
        assert not cache_exists(cache_path)

        # Save cache
        save_cache({"test": "data"}, cache_path)

        # Should exist now
        assert cache_exists(cache_path)

    def test_cache_complex_data(self, temp_cache_dir):
        """Test caching complex nested data structures."""
        complex_data = {
            "frames": [1, 2, 3, 4, 5],
            "detections": [
                {"bbox": [10, 20, 30, 40], "confidence": 0.95},
                {"bbox": [50, 60, 70, 80], "confidence": 0.87}
            ],
            "metadata": {
                "fps": 24.0,
                "total_frames": 100
            }
        }
        cache_path = os.path.join(temp_cache_dir, "complex_cache.pkl")

        # Save and load
        save_cache(complex_data, cache_path)
        loaded_data = load_cache(cache_path)

        assert loaded_data == complex_data


class TestDisplayProgress:
    """Tests for display_progress function."""

    def test_display_progress_basic(self, capsys):
        """Test basic progress display."""
        # Test progress at different points
        display_progress(0, 100, prefix="Processing")
        display_progress(50, 100, prefix="Processing")
        display_progress(100, 100, prefix="Processing")

        # Just verify it doesn't crash - actual output is hard to test
        captured = capsys.readouterr()
        assert "Processing" in captured.out

    def test_display_progress_with_eta(self, capsys):
        """Test progress display with ETA calculation."""
        import time
        start_time = time.time()

        display_progress(50, 100, prefix="Testing", start_time=start_time)

        captured = capsys.readouterr()
        assert "Testing" in captured.out
        assert "ETA" in captured.out


class TestDrawFrameNumber:
    """Tests for draw_frame_number function."""

    @pytest.fixture
    def sample_frames(self):
        """Create sample frames for testing."""
        frames = []
        for i in range(5):
            frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
            frames.append(frame)
        return frames

    def test_draw_frame_number_basic(self, sample_frames):
        """Test drawing frame numbers on frames."""
        frames_with_numbers = draw_frame_number(sample_frames)

        # Should return same number of frames
        assert len(frames_with_numbers) == len(sample_frames)

        # Should return frames with same shape
        for i, frame in enumerate(frames_with_numbers):
            assert frame.shape == sample_frames[i].shape

    def test_draw_frame_number_does_not_modify_original(self, sample_frames):
        """Test that original frames are not modified."""
        original_frame_copy = sample_frames[0].copy()

        frames_with_numbers = draw_frame_number(sample_frames)

        # Original frame should be unchanged
        assert np.array_equal(sample_frames[0], original_frame_copy)

        # Output frame should be different (has text drawn)
        assert not np.array_equal(frames_with_numbers[0], original_frame_copy)

    def test_draw_frame_number_with_start_frame(self, sample_frames):
        """Test drawing frame numbers with custom start frame."""
        frames_with_numbers = draw_frame_number(sample_frames, start_frame=100)

        # Should still work (hard to verify text content without OCR)
        assert len(frames_with_numbers) == len(sample_frames)

    def test_draw_frame_number_custom_position(self, sample_frames):
        """Test drawing frame numbers at custom position."""
        frames_with_numbers = draw_frame_number(
            sample_frames,
            position=(100, 100),
            color=(255, 0, 0),
            font_scale=1.5
        )

        assert len(frames_with_numbers) == len(sample_frames)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
