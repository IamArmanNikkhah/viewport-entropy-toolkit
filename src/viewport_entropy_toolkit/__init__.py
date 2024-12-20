"""Spatial Entropy Analysis Package.

This package provides tools for analyzing spatial entropy in 360-degree video viewport data.
It includes functionality for data processing, entropy calculation, and visualization.
"""

from .data_types import Point, RadialPoint, Vector, ValidationError, SpatialError, convert_vectors_to_coordinates
from .config import AnalyzerConfig, DEFAULT_VIDEO_DIMENSIONS, DEFAULT_TILE_COUNTS
from viewport_entropy_toolkit.analyzers import SpatialEntropyAnalyzer

__version__ = "1.0.0"
__author__ = "Prakash Lab"
__all__ = [
    'Point',
    'RadialPoint',
    'Vector',
    'ValidationError',
    'SpatialError',
    'convert_vectors_to_coordinates',
    'AnalyzerConfig',
    'SpatialEntropyAnalyzer',
    'DEFAULT_VIDEO_DIMENSIONS',
    'DEFAULT_TILE_COUNTS'
]
