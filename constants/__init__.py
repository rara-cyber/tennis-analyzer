"""
Video processing constants for tennis analysis system.
"""

# Video processing constants
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov']
MIN_VIDEO_RESOLUTION = (1280, 720)  # (width, height)
DEFAULT_FPS = 24.0
DEFAULT_CODEC = 'mp4v'  # For MP4 output

# Cache settings
CACHE_DIR = 'tracker_stubs'
ENABLE_CACHING = True

# Progress display
PROGRESS_BAR_LENGTH = 40
PROGRESS_UPDATE_FREQUENCY = 1  # Update every N frames
