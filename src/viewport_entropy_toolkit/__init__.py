"""Spatial Entropy Analysis Package.

This package provides tools for analyzing spatial entropy in 360-degree video viewport data.
It includes functionality for data processing, entropy calculation, and visualization.
"""

from .data_types import Point, RadialPoint, Vector, ValidationError, SpatialError
from .config import AnalyzerConfig, DEFAULT_VIDEO_DIMENSIONS, DEFAULT_TILE_COUNTS
from .analyzers.spatial_entropy import SpatialEntropyAnalyzer

__version__ = "1.0.0"
__author__ = "Prakash Lab"
__all__ = [
    'Point',
    'RadialPoint',
    'Vector',
    'ValidationError',
    'SpatialError',
    'AnalyzerConfig',
    'SpatialEntropyAnalyzer',
    'DEFAULT_VIDEO_DIMENSIONS',
    'DEFAULT_TILE_COUNTS'
]