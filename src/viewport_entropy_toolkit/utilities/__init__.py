"""Utilities module for spatial entropy analysis.

This module provides utility functions for data processing, entropy calculations,
and visualization in spatial entropy analysis.
"""

from .data_utils import (
    normalize_to_pixel,
    pixel_to_spherical,
    process_viewport_data,
    format_trajectory_data,
    validate_video_dimensions
)

from .entropy_utils import (
    vector_angle_distance,
    find_angular_distances,
    generate_fibonacci_lattice,
    calculate_tile_weights,
    compute_spatial_entropy,
    EntropyConfig
)

from .visualization_utils import (
    PlotManager,
    VisualizationConfig,
    create_animation,
    save_video
)

__all__ = [
    # Data utilities
    'normalize_to_pixel',
    'pixel_to_spherical',
    'process_viewport_data',
    'format_trajectory_data',
    'validate_video_dimensions',
    
    # Entropy utilities
    'vector_angle_distance',
    'find_angular_distances',
    'generate_fibonacci_lattice',
    'calculate_tile_weights',
    'compute_spatial_entropy',
    'EntropyConfig',
    
    # Visualization utilities
    'PlotManager',
    'VisualizationConfig',
    'create_animation',
    'save_video'
]