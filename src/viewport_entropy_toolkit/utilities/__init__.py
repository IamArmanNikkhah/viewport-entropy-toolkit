"""Utilities module for spatial entropy analysis.

This module provides utility functions for data processing, entropy calculations,
and visualization in spatial entropy analysis.
"""

from .data_utils import (
    generate_fibonacci_lattice,
    get_fb_tile_boundaries,
    get_lat_lon_tiles,
    normalize_to_pixel,
    pixel_to_spherical,
    process_viewport_data,
    format_trajectory_data,
    validate_video_dimensions,
    normalize,
    find_perpendicular_on_tangent_plane,
    great_circle_intersection,
    get_line_segment,
    find_nearest_point,
    spherical_interpolation,
    get_tile_corners,
    triangulate_spherical_polygon,
    angle_at_vertex,
    calculate_spherical_triangle_area,
    compute_spherical_polygon_area,
)

from .entropy_utils import (
    vector_angle_distance,
    find_angular_distances,
    find_nearest_tile,
    calculate_tile_weights,
    compute_spatial_entropy,
    compute_transition_entropy,
    calculate_naive_tile_weights,
    find_naive_tile_index,
    compute_naive_spatial_entropy,
    EntropyConfig
)

from .visualization_utils import (
    PlotManager,
    VisualizationConfig,
    create_animation,
    save_video,
    save_graph,
    save_fb_tiling_visualization_video,
    save_fb_tiling_visualization_image,
    save_tiling_visualization_video,
    save_tiling_visualization_image,
)

__all__ = [
    # Data utilities
    'generate_fibonacci_lattice',
    'get_fb_tile_boundaries',
    'get_lat_lon_tiles',
    'normalize_to_pixel',
    'pixel_to_spherical',
    'process_viewport_data',
    'format_trajectory_data',
    'validate_video_dimensions',
    'normalize',
    'find_perpendicular_on_tangent_plane',
    'great_circle_intersection',
    'get_line_segment',
    'find_nearest_point',
    'spherical_interpolation',
    'get_tile_corners',
    'triangulate_spherical_polygon',
    'angle_at_vertex',
    'calculate_spherical_triangle_area',
    'compute_spherical_polygon_area',
    
    # Entropy utilities
    'vector_angle_distance',
    'find_angular_distances',
    'find_nearest_tile',
    'calculate_tile_weights',
    'compute_spatial_entropy',
    'compute_transition_entropy',
    'calculate_naive_tile_weights',
    'find_naive_tile_index',
    'compute_naive_spatial_entropy',
    'EntropyConfig',
    
    # Visualization utilities
    'PlotManager',
    'VisualizationConfig',
    'create_animation',
    'save_video',
    'save_graph',
    'save_fb_tiling_visualization_video',
    'save_fb_tiling_visualization_image',
    'save_tiling_visualization_video',
    'save_tiling_visualization_image',
]