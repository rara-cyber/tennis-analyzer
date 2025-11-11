"""
Constants for Tennis Analysis System

Contains video processing settings, court dimensions, and player parameters.
"""

# Video processing constants
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov']
MIN_VIDEO_RESOLUTION = (1280, 720)
DEFAULT_FPS = 24.0
DEFAULT_CODEC = 'mp4v'

# Cache settings
CACHE_DIR = 'tracker_stubs'
ENABLE_CACHING = True

# Tennis court dimensions (in meters)
SINGLE_LINE_WIDTH = 8.23      # Width of singles court
DOUBLE_LINE_WIDTH = 10.97     # Width of doubles court (full width)
HALF_COURT_HEIGHT = 11.88     # Half court length (baseline to net)
FULL_COURT_HEIGHT = 23.77     # Full court length (baseline to baseline)
SERVICE_LINE_WIDTH = 6.4      # Distance from net to service line
DOUBLE_ALLEY_DIFFERENCE = 1.37  # Difference between singles and doubles width
NO_MANS_LAND_HEIGHT = 5.48    # Distance from service line to baseline

# Net dimensions
NET_HEIGHT = 0.914            # Net height at center (meters)
NET_POST_HEIGHT = 1.07        # Net height at posts (meters)

# Player heights (in meters) - for coordinate transformation
PLAYER_1_HEIGHT = 1.88        # Default player 1 height
PLAYER_2_HEIGHT = 1.91        # Default player 2 height

# Ball specifications
TENNIS_BALL_DIAMETER = 0.067  # Tennis ball diameter in meters (6.7 cm)

# Court colors (BGR format for OpenCV)
COURT_LINE_COLOR = (255, 255, 255)  # White
MINI_COURT_BG_COLOR = (255, 255, 255)  # White
MINI_COURT_LINE_COLOR = (0, 0, 0)  # Black

# Mini-court visualization settings
MINI_COURT_WIDTH = 250        # Mini-court display width in pixels
MINI_COURT_HEIGHT = 500       # Mini-court display height in pixels
MINI_COURT_BUFFER = 50        # Distance from frame edge
MINI_COURT_PADDING = 20       # Padding inside mini-court rectangle
