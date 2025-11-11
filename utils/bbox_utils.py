"""Bounding box utility functions for player tracking."""

import numpy as np


def get_center_of_bbox(bbox: list[float]) -> tuple[int, int]:
    """
    Calculate center point of bounding box.

    Args:
        bbox: [x_min, y_min, x_max, y_max]

    Returns:
        (center_x, center_y) as integers
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def get_foot_position(bbox: list[float]) -> tuple[int, int]:
    """
    Calculate foot position (ground contact point) of player.

    Args:
        bbox: [x_min, y_min, x_max, y_max]

    Returns:
        (center_x, bottom_y) as integers
    """
    x1, y1, x2, y2 = bbox
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)  # Bottom of bounding box
    return foot_x, foot_y


def get_bbox_width(bbox: list[float]) -> int:
    """
    Calculate width of bounding box.

    Args:
        bbox: [x_min, y_min, x_max, y_max]

    Returns:
        Width as integer
    """
    x1, _, x2, _ = bbox
    return int(x2 - x1)


def get_bbox_height(bbox: list[float]) -> int:
    """
    Calculate height of bounding box.

    Args:
        bbox: [x_min, y_min, x_max, y_max]

    Returns:
        Height as integer
    """
    _, y1, _, y2 = bbox
    return int(y2 - y1)


def measure_distance(point1: tuple, point2: tuple) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: (x1, y1)
        point2: (x2, y2)

    Returns:
        Distance as float
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def measure_xy_distance(point1: tuple, point2: tuple) -> tuple[float, float]:
    """
    Calculate x and y distances separately.

    Args:
        point1: (x1, y1)
        point2: (x2, y2)

    Returns:
        (distance_x, distance_y) as floats
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1), abs(y2 - y1)
