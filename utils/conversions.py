"""
Coordinate Conversion Utilities

Functions to convert between pixel coordinates and real-world meters.
"""


def convert_meters_to_pixel_distance(
    meters: float,
    reference_height_in_meters: float,
    reference_height_in_pixels: float
) -> float:
    """
    Convert distance in meters to pixels using reference.

    Args:
        meters: Distance in meters to convert
        reference_height_in_meters: Known height in meters (e.g., court width = 10.97m)
        reference_height_in_pixels: Same height in pixels

    Returns:
        Distance in pixels

    Example:
        Court width is 10.97m and measures 500 pixels.
        How many pixels is 5 meters?
        >>> convert_meters_to_pixel_distance(5, 10.97, 500)
        228.0
    """
    return (meters * reference_height_in_pixels) / reference_height_in_meters


def convert_pixel_distance_to_meters(
    pixels: float,
    reference_height_in_meters: float,
    reference_height_in_pixels: float
) -> float:
    """
    Convert distance in pixels to meters using reference.

    Args:
        pixels: Distance in pixels to convert
        reference_height_in_meters: Known height in meters
        reference_height_in_pixels: Same height in pixels

    Returns:
        Distance in meters

    Example:
        Court width is 10.97m and measures 500 pixels.
        How many meters is 250 pixels?
        >>> convert_pixel_distance_to_meters(250, 10.97, 500)
        5.485
    """
    return (pixels * reference_height_in_meters) / reference_height_in_pixels
