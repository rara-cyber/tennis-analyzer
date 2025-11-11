"""
Mini Court Visualization and Coordinate Transformation

Creates a mini-court overlay on video frames and transforms coordinates
from video pixels to mini-court pixels for visualizing player/ball positions.
"""
import cv2
import numpy as np
import sys
sys.path.append('..')
from constants import (
    DOUBLE_LINE_WIDTH,
    HALF_COURT_HEIGHT,
    SINGLE_LINE_WIDTH,
    SERVICE_LINE_WIDTH,
    DOUBLE_ALLEY_DIFFERENCE,
    MINI_COURT_WIDTH,
    MINI_COURT_HEIGHT,
    MINI_COURT_BUFFER,
    MINI_COURT_PADDING
)
from utils.conversions import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters


class MiniCourt:
    """
    Creates mini-court visualization and transforms coordinates
    from video pixels to mini-court pixels.
    """

    def __init__(self, frame: np.ndarray):
        """
        Initialize mini-court with frame dimensions.

        Args:
            frame: First video frame (to get dimensions)
        """
        # Mini-court dimensions (pixels)
        self.drawing_rectangle_width = MINI_COURT_WIDTH
        self.drawing_rectangle_height = MINI_COURT_HEIGHT
        self.buffer = MINI_COURT_BUFFER  # Distance from frame edges
        self.padding_court = MINI_COURT_PADDING  # Padding inside rectangle

        # Calculate position on frame (top-right corner)
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.buffer

        # Calculate actual court area (inside padding)
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

        # Set up court keypoints for mini-court
        self._set_court_drawing_keypoints()

        # Set up court lines
        self._set_court_lines()

        print(f"✓ Mini-court initialized ({self.drawing_rectangle_width}x{self.drawing_rectangle_height})")

    def _convert_meters_to_pixels(self, meters: float) -> float:
        """
        Convert meters to mini-court pixels using court width as reference.

        Args:
            meters: Distance in meters

        Returns:
            Distance in mini-court pixels
        """
        return convert_meters_to_pixel_distance(
            meters,
            DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )

    def _set_court_drawing_keypoints(self):
        """Calculate mini-court keypoint positions."""

        self.drawing_keypoints = [0] * 28

        # Calculate key distances in mini-court pixels
        full_court_height = self._convert_meters_to_pixels(HALF_COURT_HEIGHT * 2)
        half_court_height = self._convert_meters_to_pixels(HALF_COURT_HEIGHT)
        service_line_distance = self._convert_meters_to_pixels(SERVICE_LINE_WIDTH)
        single_width = self._convert_meters_to_pixels(SINGLE_LINE_WIDTH)
        double_width = self._convert_meters_to_pixels(DOUBLE_LINE_WIDTH)

        # Calculate horizontal positions
        court_left = self.court_start_x
        court_right = self.court_end_x
        court_center_x = (court_left + court_right) / 2

        # Calculate single court boundaries (for service boxes)
        single_left = court_center_x - single_width / 2
        single_right = court_center_x + single_width / 2

        # Calculate vertical positions
        court_top = self.court_start_y
        court_bottom = self.court_start_y + full_court_height
        court_center_y = court_top + half_court_height
        service_line_top = court_top + service_line_distance
        service_line_bottom = court_bottom - service_line_distance

        # Keypoint 0: Top-left outer corner
        self.drawing_keypoints[0] = int(court_left)
        self.drawing_keypoints[1] = int(court_top)

        # Keypoint 1: Top-right outer corner
        self.drawing_keypoints[2] = int(court_right)
        self.drawing_keypoints[3] = int(court_top)

        # Keypoint 2: Top-left service box corner (singles line, service line)
        self.drawing_keypoints[4] = int(single_left)
        self.drawing_keypoints[5] = int(service_line_top)

        # Keypoint 3: Top-right service box corner (singles line, service line)
        self.drawing_keypoints[6] = int(single_right)
        self.drawing_keypoints[7] = int(service_line_top)

        # Keypoint 4: Left net post (doubles line, center)
        self.drawing_keypoints[8] = int(court_left)
        self.drawing_keypoints[9] = int(court_center_y)

        # Keypoint 5: Right net post (doubles line, center)
        self.drawing_keypoints[10] = int(court_right)
        self.drawing_keypoints[11] = int(court_center_y)

        # Keypoint 6: Bottom-left service box corner (singles line, service line)
        self.drawing_keypoints[12] = int(single_left)
        self.drawing_keypoints[13] = int(service_line_bottom)

        # Keypoint 7: Bottom-right service box corner (singles line, service line)
        self.drawing_keypoints[14] = int(single_right)
        self.drawing_keypoints[15] = int(service_line_bottom)

        # Keypoint 8: Bottom-left outer corner
        self.drawing_keypoints[16] = int(court_left)
        self.drawing_keypoints[17] = int(court_bottom)

        # Keypoint 9: Bottom-right outer corner
        self.drawing_keypoints[18] = int(court_right)
        self.drawing_keypoints[19] = int(court_bottom)

        # Keypoint 10: Top center (baseline)
        self.drawing_keypoints[20] = int(court_center_x)
        self.drawing_keypoints[21] = int(court_top)

        # Keypoint 11: Bottom center (baseline)
        self.drawing_keypoints[22] = int(court_center_x)
        self.drawing_keypoints[23] = int(court_bottom)

        # Keypoint 12: Left center service line (singles line, net)
        self.drawing_keypoints[24] = int(single_left)
        self.drawing_keypoints[25] = int(court_center_y)

        # Keypoint 13: Right center service line (singles line, net)
        self.drawing_keypoints[26] = int(single_right)
        self.drawing_keypoints[27] = int(court_center_y)

    def _set_court_lines(self):
        """Define which keypoints connect to form court lines."""

        self.lines = [
            # Outer boundaries (doubles court)
            (0, 1),   # Top baseline (left to right)
            (0, 4),   # Left sideline (top to net)
            (4, 8),   # Left sideline (net to bottom)
            (1, 5),   # Right sideline (top to net)
            (5, 9),   # Right sideline (net to bottom)
            (8, 9),   # Bottom baseline (left to right)

            # Singles sidelines
            (2, 12),  # Left singles line (top service to net)
            (12, 6),  # Left singles line (net to bottom service)
            (3, 13),  # Right singles line (top service to net)
            (13, 7),  # Right singles line (net to bottom service)

            # Service lines
            (2, 3),   # Top service line
            (6, 7),   # Bottom service line

            # Center service line (splits service boxes)
            (10, 11), # Center line from top baseline to bottom baseline

            # Net line
            (4, 5),   # Net line (left post to right post)
        ]

    def get_keypoint_position(self, keypoint_index: int) -> tuple[int, int]:
        """
        Get mini-court position of a keypoint.

        Args:
            keypoint_index: Index of keypoint (0-13)

        Returns:
            (x, y) position in mini-court coordinates
        """
        x = int(self.drawing_keypoints[keypoint_index * 2])
        y = int(self.drawing_keypoints[keypoint_index * 2 + 1])
        return (x, y)

    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw mini-court on frame.

        Args:
            frame: Video frame

        Returns:
            Frame with mini-court drawn in top-right corner
        """
        frame_copy = frame.copy()

        # Draw white semi-transparent background
        shapes = np.zeros_like(frame_copy, dtype=np.uint8)
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            -1
        )

        # Blend with original frame
        alpha = 0.5
        mask = shapes.astype(bool)
        frame_copy[mask] = cv2.addWeighted(
            frame_copy, alpha,
            shapes, 1 - alpha, 0
        )[mask]

        # Draw court lines
        for line in self.lines:
            start_point = self.get_keypoint_position(line[0])
            end_point = self.get_keypoint_position(line[1])
            cv2.line(frame_copy, start_point, end_point, (0, 0, 0), 2)

        # Draw net line (thicker)
        net_start = self.get_keypoint_position(4)
        net_end = self.get_keypoint_position(5)
        cv2.line(frame_copy, net_start, net_end, (0, 0, 0), 3)

        return frame_copy

    def convert_position_to_mini_court(
        self,
        position: tuple[int, int],
        closest_keypoint: tuple[int, int],
        closest_keypoint_index: int,
        player_height_pixels: int,
        player_height_meters: float
    ) -> tuple[int, int]:
        """
        Convert position from video coordinates to mini-court coordinates.

        Args:
            position: (x, y) position in video frame
            closest_keypoint: (x, y) of nearest court keypoint in video
            closest_keypoint_index: Index of nearest keypoint (0-13)
            player_height_pixels: Player height in video (pixels)
            player_height_meters: Player actual height (meters)

        Returns:
            (x, y) position in mini-court coordinates
        """
        # Calculate distance from position to closest keypoint (in pixels)
        distance_x_pixels = position[0] - closest_keypoint[0]
        distance_y_pixels = position[1] - closest_keypoint[1]

        # Convert pixel distance to meters using player height as reference
        distance_x_meters = convert_pixel_distance_to_meters(
            abs(distance_x_pixels),
            player_height_meters,
            player_height_pixels
        )
        distance_y_meters = convert_pixel_distance_to_meters(
            abs(distance_y_pixels),
            player_height_meters,
            player_height_pixels
        )

        # Preserve direction (sign)
        if distance_x_pixels < 0:
            distance_x_meters = -distance_x_meters
        if distance_y_pixels < 0:
            distance_y_meters = -distance_y_meters

        # Convert meters to mini-court pixels
        distance_x_mini = convert_meters_to_pixel_distance(
            abs(distance_x_meters),
            DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )
        distance_y_mini = convert_meters_to_pixel_distance(
            abs(distance_y_meters),
            DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )

        # Preserve direction
        if distance_x_meters < 0:
            distance_x_mini = -distance_x_mini
        if distance_y_meters < 0:
            distance_y_mini = -distance_y_mini

        # Get closest keypoint position in mini-court
        closest_mini_x = self.drawing_keypoints[closest_keypoint_index * 2]
        closest_mini_y = self.drawing_keypoints[closest_keypoint_index * 2 + 1]

        # Calculate final mini-court position
        mini_x = int(closest_mini_x + distance_x_mini)
        mini_y = int(closest_mini_y + distance_y_mini)

        return (mini_x, mini_y)

    def draw_points_on_mini_court(
        self,
        frame: np.ndarray,
        positions: dict,
        color: tuple = (0, 0, 255)
    ) -> np.ndarray:
        """
        Draw player/ball positions on mini-court.

        Args:
            frame: Video frame with mini-court already drawn
            positions: Dict mapping ID to (x, y) mini-court position
            color: BGR color tuple (default: red)

        Returns:
            Frame with positions drawn on mini-court
        """
        frame_copy = frame.copy()

        for player_id, position in positions.items():
            x, y = position

            # Only draw if position is within mini-court bounds
            if (self.court_start_x <= x <= self.court_end_x and
                self.court_start_y <= y <= self.court_end_y):

                # Draw circle at position
                cv2.circle(frame_copy, (int(x), int(y)), 5, color, -1)

                # Optionally draw ID label
                cv2.putText(
                    frame_copy,
                    str(player_id),
                    (int(x) + 7, int(y) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )

        return frame_copy

    def draw_background_rectangle(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw just the white background rectangle (without court lines).

        Args:
            frame: Video frame

        Returns:
            Frame with background rectangle drawn
        """
        frame_copy = frame.copy()

        # Draw white semi-transparent background
        overlay = frame_copy.copy()
        cv2.rectangle(
            overlay,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            -1
        )

        # Blend
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

        return frame_copy
