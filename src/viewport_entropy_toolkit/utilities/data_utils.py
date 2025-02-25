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

from viewport_entropy_toolkit import Point, RadialPoint, Vector, ValidationError
from entropy_utils import generate_fibonacci_lattice

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
    
    if point.pixel_x > video_width or point.pixel_y > video_height:
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

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector.

Args:
    v: A np.ndarray for a vector.

Returns:
    np.ndarray: The normalized vector.
"""
    return v / np.linalg.norm(v)

def find_perpendicular_on_tangent_plane(vec: np.ndarray, midpoint: np.ndarray) -> np.ndarray:
    """
    Find a vector perpendicular to `vec` that lies on the tangent plane at the midpoint on the sphere.

    Args:
        vec: Vector between two points on the sphere.
        midpoint: Midpoint between the two tile centers (point on the sphere).
    Returns:
        np.ndarray: A vector perpendicular to `vec` on the tangent plane at the midpoint.
    """
    # Normalize the midpoint to get the radial vector
    radial_vec = normalize(midpoint)

    # Compute the cross product between the radial vector and the vector `vec`
    perp_vec = np.cross(radial_vec, vec)

    # Normalize the perpendicular vector to ensure it lies on the tangent plane
    perp_vec = normalize(perp_vec)

    return perp_vec

def great_circle_intersection(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """
    Calculate the intersection points of two great circles on a unit sphere.

    Args:
        n1: Normal vector of the first great circle.
        n2: Normal vector of the second great circle.
    Returns:
        Two intersection points on the unit sphere (each is a 3D vector).
    """
    # Normalize the normal vectors (ensures they are unit vectors)
    n1 = normalize(n1)
    n2 = normalize(n2)

    # Find the direction of the line of intersection (cross product of the two normal vectors)
    line_direction = np.cross(n1, n2)
    line_direction = normalize(line_direction)  # Normalize the direction vector

    # The two intersection points are the normalized line direction and its inverse
    p1 = line_direction
    p2 = -line_direction

    return p1, p2

def get_line_segment(v1: Vector, v2: Vector) -> np.ndarray:
    """
    Calculate the line segment between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.
    Returns:
        np.ndarray: the line segment between the vectors.
    """
    line_segment = np.array([v1.x - v2.x, v1.y - v2.y, v1.z - v2.z])

    return line_segment

def find_nearest_point(v1: Vector, v2: Vector, compare_vector: Vector) -> Vector:
    """
    Calculate the nearest point to the compared point.

    Args:
        p1: First vector.
        p2: Second vector.
        compare_vector: The point to compare.
    Returns:
        The nearest vector.
    """
    v1_i_seg = get_line_segment(compare_vector, v1)
    v2_i_seg = get_line_segment(compare_vector, v2)
    length_i_v1 = np.linalg.norm(v1_i_seg).round(4)
    length_i_v2 = np.linalg.norm(v2_i_seg).round(4)
    
    if (length_i_v1 < length_i_v2):
        return v1
    else:
        return v2

def spherical_interpolation(v1: Vector, v2: Vector, t: float) -> np.ndarray:
    """Perform spherical linear interpolation (slerp) between two points on the sphere (takes the shorter path).
    
    Args:
        v1: The Vector representing the first point on the sphere.
        v2: The Vector representing the second point on the sphere.

    Returns:
        np.ndarray: An array for the vector of the spherical linear interpolation.
    """
    p1 = np.array([v1.x, v1.y, v1.z])
    p2 = np.array([v2.x, v2.y, v2.z])

    # Normalize the vectors
    p1 = normalize(p1)
    p2 = normalize(p2)

    dot_product = np.dot(p1, p2)
    # Clip dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle between vectors
    theta = np.arccos(dot_product)

    # Compute slerp
    return (np.sin((1 - t) * theta) * p1 + np.sin(t * theta) * p2) / np.sin(theta)

def get_fb_tile_boundaries(tile_count: int) -> Dict:
    """Perform spherical linear interpolation (slerp) between two points on the sphere (takes the shorter path).
    
    Args:
        tile_count: The number of tiles in the fibonacci lattice.

    Returns:
        Dict: A dictionary where each tile index has a list of tile boundaries.

    Raises:
        ValidationError: Error if tile count is less than 1.
    """

    if (tile_count <= 0):
        raise ValidationError("Tile counts cannot be less than 1 for to visualize tiling!")

    # Grab tile center points
    tile_centers_vectors = generate_fibonacci_lattice(tile_count)
    tile_boundaries = {}

    for index_i in range(len(tile_centers_vectors)):
        tile_center_i = tile_centers_vectors[index_i]
        tile_boundaries[index_i] = []

        neighbors = []
        great_circle_vectors = {}
        for index_j in range(0, len(tile_centers_vectors)):
            if (index_j == index_i):
                continue

            tile_center_j = tile_centers_vectors[index_j]


            # Find the vector between the two tile centers
            line_segment = get_line_segment(tile_center_i, tile_center_j)
            line_vector = Vector(line_segment[0], line_segment[1], line_segment[2])

            # Compute the length of the line segment
            length = np.linalg.norm(line_segment)
            neighbors.append([index_j, length])

            # Calculate the midpoint of the line between the two centers
            midpoint = np.array([(tile_center_i.x + tile_center_j.x) / 2, (tile_center_i.y + tile_center_j.y) / 2, (tile_center_i.z + tile_center_j.z) / 2])

            # Normalize the midpoint to project it onto the sphere's surface
            midpoint = normalize(midpoint)
            # Grab the perpendicular segment to the line segment that uses the plane tangeant to the sphere at the midpoint.
            perp_segment = find_perpendicular_on_tangent_plane(line_segment, midpoint)
            perp_vector = Vector(perp_segment[0], perp_segment[1], perp_segment[2])

            # We use the normal vector of the plane of the great circle.
            # This plane passes through the center of the sphere and defines the great circle by its intersection with the sphere.
            great_circle_vectors[index_j] = line_vector

        neighbors.sort(key=lambda x: x[1])
        smallest_distance = neighbors[0][1]
        nearest_tile_boundaries = {}

        for index_j in range(len(neighbors)):
            neighbor_j = neighbors[index_j]
            if (neighbor_j[1] >= smallest_distance * 1.7):
                break

            tile_index_j = neighbor_j[0]
            tile_center_j = tile_centers_vectors[tile_index_j]
            great_circle_j = great_circle_vectors[tile_index_j]

            intersections = []

            for index_k in range(len(neighbors)):
                neighbor_k = neighbors[index_k]
                if (index_k == index_j):
                    continue
                if (neighbor_k[1] >= smallest_distance * 1.7):
                    break

                tile_index_k = neighbor_k[0]
                tile_center_k = tile_centers_vectors[tile_index_k]
                great_circle_k = great_circle_vectors[tile_index_k]

                p1, p2 = great_circle_intersection(np.array([great_circle_j.x, great_circle_j.y, great_circle_j.z]), np.array([great_circle_k.x, great_circle_k.y, great_circle_k.z]))

                p1_vec = Vector(p1[0], p1[1], p1[2])
                p2_vec = Vector(p2[0], p2[1], p2[2])

                intersection_vector = find_nearest_point(p1_vec, p2_vec, tile_center_i)


                intersection_i_seg = get_line_segment(tile_center_i, intersection_vector)
                length_i = np.linalg.norm(intersection_i_seg).round(4)

                intersections.append([intersection_vector, length_i, tile_index_k])

            if (len(intersections) < 2):
                continue

            intersections.sort(key=lambda x: x[1])

            # Check if the two shortest intersection points are shorter than the intersection of their great circles.
            shortest_intersection = intersections[0]
            second_shortest_intersection = intersections[1]

            tile_index_a = shortest_intersection[2]
            tile_index_b = second_shortest_intersection[2]

            great_circle_a = great_circle_vectors[tile_index_a]
            great_circle_b = great_circle_vectors[tile_index_b]

            p1, p2 = great_circle_intersection(np.array([great_circle_a.x, great_circle_a.y, great_circle_a.z]), np.array([great_circle_b.x, great_circle_b.y, great_circle_b.z]))

            p1_vec = Vector(p1[0], p1[1], p1[2])
            p2_vec = Vector(p2[0], p2[1], p2[2])

            intersection_vector = find_nearest_point(p1_vec, p2_vec, tile_center_i)

            intersection_i_seg = get_line_segment(tile_center_i, intersection_vector)
            length_intersection = np.linalg.norm(intersection_i_seg).round(4)
            midpoint_ij = np.array([(tile_center_i.x + tile_center_j.x) / 2, (tile_center_i.y + tile_center_j.y) / 2, (tile_center_i.z + tile_center_j.z) / 2])
            midpoint_ij = normalize(midpoint_ij)
            midpoint_ij_vec = Vector(midpoint_ij[0], midpoint_ij[1], midpoint_ij[2])
            midpoint_i_seg = get_line_segment(tile_center_i, midpoint_ij_vec)
            length_midpoint = np.linalg.norm(midpoint_i_seg).round(4)

            # If the intersection of great circles a and b is closer to i than the midpoint of i and j,
            # then these intersection points do not form a valid tile boundary.

            # If the intersection is further away, then it is a valid tile boundary.
            if (length_intersection > length_midpoint):
                tile_boundary = [shortest_intersection[0], second_shortest_intersection[0]]
                tile_boundaries[index_i].append(tile_boundary)
    
    return tile_boundaries
