"""Tests for spatial entropy analyzer."""

import pytest
from viewport_entropy_toolkit.data_types import Point, RadialPoint, Vector
from viewport_entropy_toolkit.analyzers.spatial_entropy import SpatialEntropyAnalyzer
from viewport_entropy_toolkit.config import AnalyzerConfig

def test_point_creation():
    """Test Point class creation."""
    point = Point(pixel_x=100, pixel_y=200)
    assert point.pixel_x == 100
    assert point.pixel_y == 200

def test_radial_point_creation():
    """Test RadialPoint class creation."""
    point = RadialPoint(lon=45.0, lat=30.0)
    assert point.lon == 45.0
    assert point.lat == 30.0

def test_vector_creation():
    """Test Vector class creation."""
    vector = Vector(x=1.0, y=2.0, z=3.0)
    assert vector.x == 1.0
    assert vector.y == 2.0
    assert vector.z == 3.0

def test_analyzer_initialization():
    """Test SpatialEntropyAnalyzer initialization."""
    analyzer = SpatialEntropyAnalyzer(config=AnalyzerConfig(
        video_width=100,
        video_height=200
    ))
    assert analyzer.config.video_width == 100
    assert analyzer.config.video_height == 200
