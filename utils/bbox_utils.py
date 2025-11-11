"""
Bounding Box Utilities

Functions for working with bounding boxes and spatial measurements.
"""
import math


def get_center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """
    Get center point of bounding box.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Center point as (x, y)
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def get_foot_position(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """
    Get foot position (bottom center) of bounding box.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Foot position as (x, y)
    """
    x1, y1, x2, y2 = bbox
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)  # Bottom of bbox
    return (foot_x, foot_y)


def get_bbox_width(bbox: tuple[int, int, int, int]) -> int:
    """
    Get width of bounding box.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Width in pixels
    """
    x1, y1, x2, y2 = bbox
    return x2 - x1


def get_bbox_height(bbox: tuple[int, int, int, int]) -> int:
    """
    Get height of bounding box.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Height in pixels
    """
    x1, y1, x2, y2 = bbox
    return y2 - y1


def measure_distance(point1: tuple[int, int], point2: tuple[int, int]) -> float:
    """
    Measure Euclidean distance between two points.

    Args:
        point1: First point as (x, y)
        point2: Second point as (x, y)

    Returns:
        Distance in pixels
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def measure_xy_distance(point1: tuple[int, int], point2: tuple[int, int]) -> tuple[float, float]:
    """
    Measure distance in X and Y directions separately.

    Args:
        point1: First point as (x, y)
        point2: Second point as (x, y)

    Returns:
        Tuple of (x_distance, y_distance)
    """
    x1, y1 = point1
    x2, y2 = point2
    return (abs(x2 - x1), abs(y2 - y1))
