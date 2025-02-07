"""Entropy utilities module for spatial entropy analysis.

This module provides utilities for calculating spatial entropy and managing tile-based
analysis of viewport trajectories in 360-degree videos. It includes functions for
generating Fibonacci lattices, calculating angular distances, and computing spatial entropy.

Functions:
    compute_spatial_entropy: Calculates spatial entropy for a set of vectors.
    generate_fibonacci_lattice: Generates uniformly distributed points on a sphere.
    calculate_tile_weights: Computes weight distribution across tiles.
    vector_operations: Various vector calculation utilities.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from viewport_entropy_toolkit import Vector, RadialPoint, ValidationError


@dataclass
class EntropyConfig:
    """Configuration for entropy calculations.
    
    Attributes:
        fov_angle (float): Field of view angle in degrees.
        use_weight_distribution (bool): Whether to use weighted distribution.
        power_factor (float): Power factor for weight calculation.
    """
    fov_angle: float = 120.0
    use_weight_distribution: bool = True
    power_factor: float = 2.0

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        if not 0 < self.fov_angle <= 360:
            raise ValidationError("FOV angle must be between 0 and 360 degrees")
        if self.power_factor <= 0:
            raise ValidationError("Power factor must be positive")


def vector_angle_distance(v1: Vector, v2: Vector) -> float:
    """Computes the angle between two vectors in radians.
    
    Args:
        v1: First vector.
        v2: Second vector.
    
    Returns:
        float: Angle between vectors in radians.
    
    Raises:
        ValidationError: If vectors are invalid.
    """
    try:
        v1_np = np.array([v1.x, v1.y, v1.z])
        v2_np = np.array([v2.x, v2.y, v2.z])
        
        v1_normalized = v1_np / np.linalg.norm(v1_np)
        v2_normalized = v2_np / np.linalg.norm(v2_np)
        
        dot_product = np.dot(v1_normalized, v2_normalized)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        return np.arccos(dot_product)
        
    except Exception as e:
        raise ValidationError(f"Error calculating vector angle: {str(e)}")


def find_angular_distances(
    vector: Vector,
    tile_centers: List[Vector]
) -> np.ndarray:
    """Finds angular distances between a vector and tile centers.
    
    Args:
        vector: Reference vector.
        tile_centers: List of tile center vectors.
    
    Returns:
        np.ndarray: Array of [tile_index, angular_distance] pairs.
    """
    distances = np.array([
        [i, vector_angle_distance(vector, center)]
        for i, center in enumerate(tile_centers)
    ])
    return distances


def generate_fibonacci_lattice(num_points: int) -> List[Vector]:
    """Generates Fibonacci lattice points on a sphere.
    
    Args:
        num_points: Number of points to generate.
    
    Returns:
        List[Vector]: List of unit vectors representing lattice points.
    
    Raises:
        ValidationError: If number of points is invalid.
    """
    if num_points <= 0:
        raise ValidationError("Number of points must be positive")
        
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vectors = []
    
    N = int(num_points / 2)
    
    for i in range(-N, N + 1):
        lat = np.arcsin(2 * i / (2 * N + 1)) * 180 / np.pi
        lon = (i % phi) * 360 / phi
        
        # Normalize longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180
        
        # Convert to vector
        vector = Vector.from_spherical(lon, lat)
        vectors.append(vector)
    
    return vectors

def find_nearest_tile(
        vector: Vector,
        tile_centers: List[Vector]
)-> int:
    """
    Find nearest tile for a given vector.

    Args:
        vector: Input vector.
        tile_centers: List of tile center vectors.
    
    Returns:
        int: the index of tile_centers for the closest tile.
    """
    distances = find_angular_distances(vector, tile_centers)
    nearest_tile = int(distances[np.argmin(distances[:, 1])][0])

    return nearest_tile

def calculate_tile_weights(
    vector: Vector,
    tile_centers: List[Vector],
    config: EntropyConfig
) -> Dict[Vector, float]:
    """Calculates weight distribution across tiles for a vector using fibonacci lattice tiling.
    
    Args:
        vector: Input vector.
        tile_centers: List of tile center vectors.
        config: Entropy calculation configuration.
    
    Returns:
        Dict[Vector, float]: Dictionary mapping tile centers to weights.
    """
    weights = {}
    max_angular_distance = np.radians(config.fov_angle / 2.0)
    
    # Calculate angular distances
    distances = find_angular_distances(vector, tile_centers)
    distances = sorted(distances, key=lambda x: x[1])
    
    if config.use_weight_distribution:
        # Distribute weights based on angular distance
        for tile_idx, distance in distances:
            if distance < max_angular_distance:
                tile = tile_centers[int(tile_idx)]
                weight = ((max_angular_distance - distance) / max_angular_distance) ** config.power_factor
                weights[tile] = weight
            else:
                break
    else:
        # Assign weight only to nearest tile
        nearest_idx = int(distances[0][0])
        weights[tile_centers[nearest_idx]] = 1.0
    
    return weights


def compute_spatial_entropy(
    vector_dict: Dict[str, Vector],
    tile_centers: List[Vector],
    config: EntropyConfig
) -> Tuple[float, Dict[Vector, float], Dict[str, int]]:
    """Computes spatial entropy for a set of vectors.
    
    Args:
        vector_dict: Dictionary mapping identifiers to vectors.
        tile_centers: List of tile center vectors.
        config: Entropy calculation configuration.
    
    Returns:
        Tuple containing:
        - float: Normalized spatial entropy.
        - Dict[Vector, float]: Tile weight distribution.
        - Dict[str, int]: Tile assignments for each vector.
    
    Raises:
        ValidationError: If input data is invalid.
    """
    if not vector_dict:
        raise ValidationError("Empty vector dictionary")
    if not tile_centers:
        raise ValidationError("No tile centers provided")
    
    num_tiles = len(tile_centers)
    weight_per_tile: Dict[Vector, float] = {}
    total_weight = 0.0
    tile_assignments: Dict[str, int] = {}
    
    # Calculate weights for each vector
    for identifier, vector in vector_dict.items():
        if vector is None:
            continue
            
        weights = calculate_tile_weights(vector, tile_centers, config)
        
        # Record tile assignment (nearest tile)
        nearest_tile = find_nearest_tile(vector, tile_centers)
        tile_assignments[identifier] = nearest_tile

        # Accumulate weights
        for tile, weight in weights.items():
            weight_per_tile[tile] = weight_per_tile.get(tile, 0.0) + weight
            total_weight += weight
    
    # Calculate entropy
    spatial_entropy = 0.0
    for weight in weight_per_tile.values():
        proportion = weight / total_weight
        spatial_entropy -= proportion * np.log2(proportion)
    
    # Calculate maximum possible entropy
    if config.use_weight_distribution or total_weight > num_tiles:
        max_proportion = 1.0 / num_tiles
        max_entropy = -num_tiles * max_proportion * np.log2(max_proportion)
    else:
        max_proportion = 1.0 / total_weight
        max_entropy = -total_weight * max_proportion * np.log2(max_proportion)
    
    # Normalize entropy
    normalized_entropy = spatial_entropy / max_entropy
    
    return normalized_entropy, weight_per_tile, tile_assignments

def compute_transition_entropy(
        prior_vector_dict: dict,
        current_vector_dict: dict,
        tile_centers: List[Vector],
        config: EntropyConfig,
        FOV_angle: float
        ) -> Tuple[float, Dict[Vector, float], Dict[str, int]]:
    """
    Computes transition entropy for a set of vectors.
    
    Args:
        prior_vector_dict: Dictionary mapping identifiers to prior vectors.
        current_vector_dict: Dictionary mapping identifiers to current vectors.
        tile_centers: List of tile center vectors.
        config: Entropy calculation configuration.
    
    Returns:
        Tuple containing:
        - float: Normalized transition entropy.
        - Dict[Vector, float]: Tile weight distribution.
        - Dict[str, int]: Tile assignments for each vector.
    
    Raises:
        ValidationError: If input data is invalid.
    """

    if not prior_vector_dict or not current_vector_dict:
        raise ValidationError("Empty vector dictionary")
    if not tile_centers:
        raise ValidationError("No tile centers provided")
    
    num_tiles = len(tile_centers)
    weight_per_tile: Dict[Vector, float] = {}
    total_weight = 0.0
    tile_assignments: Dict[str, int] = {}

    weight_per_tile = {}
    transition_weight_per_tile = {}
    total_weight = 0
    tile_assignments = {}
    transition_entropy = 0

    # Maximum angular distance to consider for transition entropy calculation is half of FOV angle
    max_angular_distance = float(FOV_angle) / 2.0

    # Find the proportion of FOVs and of transitions in each tile.
    for identifier, vector in current_vector_dict.items():
        if (identifier not in prior_vector_dict or identifier not in current_vector_dict):
            continue

        prior_vector = prior_vector_dict[identifier]
        current_vector = current_vector_dict[identifier]
        

        # Find nearest tile for prior and current vectors
        distances = find_angular_distances(vector, tile_centers)
        tile_assignments[identifier] = int(distances[np.argmin(distances[:, 1])][0])

        previous_nearest_tile_index = find_nearest_tile(prior_vector, tile_centers)
        previous_nearest_tile = tile_centers[previous_nearest_tile_index]
        current_nearest_tile_index = find_nearest_tile(current_vector, tile_centers)
        current_nearest_tile = tile_centers[current_nearest_tile_index]

        tile_assignments[identifier] = (previous_nearest_tile_index, current_nearest_tile_index)  # Store tile assignments

        weight = 1

        if previous_nearest_tile not in weight_per_tile:
            transition_weight_per_tile[previous_nearest_tile] = {}
            transition_weight_per_tile[previous_nearest_tile][current_nearest_tile_index] = weight
        else:
            if current_nearest_tile not in transition_weight_per_tile[previous_nearest_tile]:
                transition_weight_per_tile[previous_nearest_tile][current_nearest_tile] = weight
            else:
                transition_weight_per_tile[previous_nearest_tile][current_nearest_tile] += weight

        if (previous_nearest_tile not in weight_per_tile):
            weight_per_tile[previous_nearest_tile] = weight
        else:
            weight_per_tile[previous_nearest_tile] += weight

        total_weight += weight

    # Given proportion of weights in each tile, calculate the transition entropy
    for vector_key in weight_per_tile:
        tile_weight = weight_per_tile[vector_key]

        # Calculate the tile proportion as the tile weight over the total weight.
        tile_proportion = float(tile_weight) / float(total_weight)

        total_transition_weight = 0
        total_cell_transition_entropy = 0

        # Calculate the total transition weight.
        for transition_vect_key in transition_weight_per_tile[vector_key]:
            transition_weight = transition_weight_per_tile[vector_key][transition_vect_key]
            total_transition_weight += transition_weight

        # Calculate the transition proportions and cell entropy.
        for transition_vect_key in transition_weight_per_tile[vector_key]:
            transition_proportion = float(transition_weight) / float(total_transition_weight)
            cell_transition_entropy = transition_proportion * np.log2(transition_proportion)
            total_cell_transition_entropy += cell_transition_entropy

        cell_entropy = -tile_proportion * total_cell_transition_entropy
        transition_entropy += cell_entropy

    # The maximum transition entropy is either the sum from 1 to number of tiles of:
    if (total_weight > num_tiles):
        tile_proportion_for_max = (1 / num_tiles)
        maximum_transition_entropy = num_tiles * -tile_proportion_for_max * np.log2(tile_proportion_for_max)
    # Or the sum from 1 to total weight of:
    else:
        tile_proportion_for_max = (1 / total_weight)
        maximum_transition_entropy = total_weight * -tile_proportion_for_max * np.log2(tile_proportion_for_max)

    # Normalize the transition entropy by dividing the transition entropy by the maximum transition entropy
    transition_entropy = transition_entropy / maximum_transition_entropy

    return transition_entropy, weight_per_tile, tile_assignments


def calculate_naive_tile_weights(
    point: RadialPoint,
    tile_height: float,
    tile_width: float,
    config: EntropyConfig
) -> Dict[str, float]:
    """Calculates weight distribution across tiles for a vector using naive tiling.
    
    Args:
        point: Input point.
        tile_height: Latitudinal height of the tile, in radians.
        tile_width: Latitudinal width of tile, in radians.
        config: Entropy calculation configuration.
    
    Returns:
        Dict[str, float]: Dictionary mapping tile centers to weights.
    """

    weights = {}
    
    # find the tile this point sits in.
    tile_index = find_naive_tile_index(point, tile_height, tile_width)

    weights[tile_index] = 1.0
    
    return weights

def find_naive_tile_index(
    point: RadialPoint,
    tile_height: float,
    tile_width: float,
) -> str:
    """Calculates the lon and lat indices for the naive tile this point sits in.
    
    Args:
        point: Input point.
        tile_height: Latitudinal height of the tile, in radians.
        tile_width: Latitudinal width of tile, in radians.
    
    Returns:
        Dict[Vector, float]: Dictionary mapping tile centers to weights.
    """
    # calculate the tile this vector sits in.
    tile_lon_index = int((point.lon + 180) / tile_width)
    tile_lat_index = int((point.lat + 90) / tile_height)

    return f"{tile_lon_index}_{tile_lat_index}"

def compute_naive_spatial_entropy(
    points_dict: Dict[str, RadialPoint],
    tile_height: int,
    tile_width: int,
    config: EntropyConfig
) -> Tuple[float, Dict[str, float], Dict[str, str]]:
    """Computes spatial entropy for a set of radial points using naive tiling.
    
    Args:
        points_dict: Dictionary mapping identifiers to radial points.
        tile_height: The tile latitudinal height.
        tile_width: The tile longitudinal width.
        config: Entropy calculation configuration.
    
    Returns:
        Tuple containing:
        - float: Normalized spatial entropy.
        - Dict[str, float]: Tile weight distribution.
        - Dict[str, str]: Tile assignments for each radial point.
    
    Raises:
        ValidationError: If input data is invalid.
    """
    if not points_dict:
        raise ValidationError("Empty radial points dictionary")
    if not tile_height or not tile_width:
        raise ValidationError("No tile dimensions provided")
    if 180 % tile_height != 0:
        raise ValidationError("Tile height must divide 180!")
    if 360 % tile_width != 0:
        raise ValidationError("Tile width must divide 360!")

    num_tiles = int(180.0 / tile_height) * int(360.0 / tile_width)
    weight_per_tile: Dict[str, float] = {}
    total_weight = 0.0
    tile_assignments: Dict[str, str] = {}
    
    # Calculate weights for each radial points
    for identifier, point in points_dict.items():
        if point is None:
            continue
            
        weights = calculate_naive_tile_weights(point, tile_height, tile_width, config)
        
        # Record tile assignment (nearest tile)
        nearest_tile = find_naive_tile_index(point, tile_height, tile_width)
        tile_assignments[identifier] = nearest_tile

        # Accumulate weights
        for tile, weight in weights.items():
            weight_per_tile[tile] = weight_per_tile.get(tile, 0.0) + weight
            total_weight += weight
    
    # Calculate entropy
    spatial_entropy = 0.0
    for weight in weight_per_tile.values():
        proportion = weight / total_weight
        spatial_entropy -= proportion * np.log2(proportion)
    
    # Calculate maximum possible entropy
    if config.use_weight_distribution or total_weight > num_tiles:
        max_proportion = 1.0 / num_tiles
        max_entropy = -num_tiles * max_proportion * np.log2(max_proportion)
    else:
        max_proportion = 1.0 / total_weight
        max_entropy = -total_weight * max_proportion * np.log2(max_proportion)
    
    # Normalize entropy
    normalized_entropy = spatial_entropy / max_entropy
    
    return normalized_entropy, weight_per_tile, tile_assignments