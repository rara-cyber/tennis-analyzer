"""
Utilities Module

Provides utility functions for video processing, bounding boxes, and coordinate conversions.
"""
from .video_utils import (
    read_video,
    save_video,
    display_progress,
    save_cache,
    load_cache,
    cache_exists,
    draw_frame_number
)
from .bbox_utils import (
    get_center_of_bbox,
    get_foot_position,
    get_bbox_width,
    get_bbox_height,
    measure_distance,
    measure_xy_distance
)
from .conversions import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters
)

__all__ = [
    # Video utilities
    'read_video',
    'save_video',
    'display_progress',
    'save_cache',
    'load_cache',
    'cache_exists',
    'draw_frame_number',
    # Bounding box utilities
    'get_center_of_bbox',
    'get_foot_position',
    'get_bbox_width',
    'get_bbox_height',
    'measure_distance',
    'measure_xy_distance',
    # Conversion utilities
    'convert_meters_to_pixel_distance',
    'convert_pixel_distance_to_meters'
]
