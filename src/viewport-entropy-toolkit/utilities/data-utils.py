"""Data utilities module for spatial entropy analysis.

This module provides utilities for data processing and coordinate transformations
in spatial entropy analysis of 360-degree videos. It includes functions for
converting between different coordinate systems and processing viewport data.

Functions:
    normalize_to_pixel: Converts normalized coordinates to pixel coordinates.
    pixel_to_spherical: Converts pixel coordinates to spherical coordinates.
    process_viewport_data: Processes viewport center trajectory data.
    validate_video_dimensions: Validates video dimensions.
    format_trajectory_data: Formats trajectory data for analysis.
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path

from .data_types import Point, RadialPoint, Vector, ValidationError


def validate_video_dimensions(width: int, height: int) -> None:
    """Validates video dimensions.
    
    Args:
        width (int): Video width in pixels.
        height (int): Video height in pixels.
    
    Raises:
        ValidationError: If dimensions are invalid.
    """
    if width <= 0 or height <= 0:
        raise ValidationError("Video dimensions must be positive")
    if width % 2 != 0 or height % 2 != 0:
        raise ValidationError("Video dimensions must be even numbers")


def normalize_to_pixel(normalized: np.ndarray, dimension: int) -> np.ndarray:
    """Converts normalized coordinates to pixel coordinates.
    
    Args:
        normalized (np.ndarray): Normalized coordinates (range [0, 1]).
        dimension (int): Dimension (width or height) for scaling.
    
    Returns:
        np.ndarray: Pixel coordinates.
    
    Raises:
        ValidationError: If normalized values are outside [0, 1] range.
    """
    if np.any((normalized < 0) | (normalized > 1)):
        raise ValidationError("Normalized coordinates must be between 0 and 1")
    if dimension <= 0:
        raise ValidationError("Dimension must be positive")
    
    return (normalized * dimension).astype(int)


def pixel_to_spherical(point: Point, video_width: int, video_height: int) -> RadialPoint:
    """Converts pixel coordinates to spherical coordinates.
    
    Args:
        point (Point): Point in pixel coordinates.
        video_width (int): Width of the video in pixels.
        video_height (int): Height of the video in pixels.
    
    Returns:
        RadialPoint: Point in spherical coordinates.
    
    Raises:
        ValidationError: If coordinates are invalid.
    """
    validate_video_dimensions(video_width, video_height)
    
    if point.pixel_x >= video_width or point.pixel_y >= video_height:
        raise ValidationError("Pixel coordinates exceed video dimensions")
    
    lon = (point.pixel_x / video_width) * 360 - 180
    lat = 90 - (point.pixel_y / video_height) * 180
    
    return RadialPoint(lon=lon, lat=lat)


def process_viewport_data(
    filepath: Union[str, Path],
    video_width: int,
    video_height: int
) -> Tuple[pd.DataFrame, str]:
    """Processes viewport center trajectory data from a CSV file.
    
    Args:
        filepath (Union[str, Path]): Path to the CSV file.
        video_width (int): Width of the video in pixels.
        video_height (int): Height of the video in pixels.
    
    Returns:
        Tuple[pd.DataFrame, str]: Processed data and trajectory identifier.
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValidationError: If data format is invalid.
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Read and validate data
        data = pd.read_csv(filepath, usecols=["time", "2dmu", "2dmv"]).dropna()
        if data.empty:
            raise ValidationError(f"No valid data found in {filepath}")
        
        # Normalize time to start at 0
        data["time"] -= data["time"].min()
        
        # Convert normalized coordinates to pixel coordinates
        data["pixel_x"] = normalize_to_pixel(data["2dmu"].values, video_width)
        data["pixel_y"] = normalize_to_pixel(data["2dmv"].values, video_height)
        
        # Convert to spherical coordinates
        spherical_coords = data.apply(
            lambda row: pixel_to_spherical(
                Point(row["pixel_x"], row["pixel_y"]),
                video_width,
                video_height
            ),
            axis=1
        )
        data["lon"], data["lat"] = zip(*[(p.lon, p.lat) for p in spherical_coords])
        
        # Generate identifier from filename
        identifier = filepath.stem
        
        return data, identifier
    
    except Exception as e:
        raise ValidationError(f"Error processing viewport data: {str(e)}")


def format_trajectory_data(
    trajectory_data: List[Tuple[str, pd.DataFrame]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Formats trajectory data for analysis.
    
    Args:
        trajectory_data: List of tuples containing (identifier, data) pairs.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Points and vectors at each time step.
    
    Raises:
        ValidationError: If data format is invalid.
    """
    if not trajectory_data:
        raise ValidationError("No trajectory data provided")
    
    # Initialize column names
    columns = ["time"] + [pair[0] for pair in trajectory_data]
    
    # Initialize dictionaries for points and vectors
    points_dict = {col: [] for col in columns}
    vectors_dict = {col: [] for col in columns}
    
    # Process each trajectory
    for identifier, data in trajectory_data:
        data["time"] = data["time"].round(1)  # Round time to nearest tenth
        
        for row in data.itertuples(index=False):
            time = row.time
            
            # Add time point if not exists
            if time not in points_dict["time"]:
                points_dict["time"].append(time)
                vectors_dict["time"].append(time)
                
                # Initialize None for all trajectories at this time
                for col in columns[1:]:
                    points_dict[col].append(None)
                    vectors_dict[col].append(None)
            
            # Get index for this time
            idx = points_dict["time"].index(time)
            
            # Create and store RadialPoint
            lon = round(row.lon, 1)
            lat = round(row.lat, 1)
            
            # Normalize coordinates if needed
            if lon <= -180:
                lon = (lon + 360) % 360 - 180
            if lat <= -90:
                lat = (lat + 180) % 180 - 90
            
            radial_point = RadialPoint(lon=lon, lat=lat)
            points_dict[identifier][idx] = radial_point
            
            # Create and store Vector
            vector = Vector.from_spherical(lon, lat)
            vectors_dict[identifier][idx] = vector
    
    # Create DataFrames
    points_df = pd.DataFrame(points_dict)
    vectors_df = pd.DataFrame(vectors_dict)
    
    return points_df, vectors_df
