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

from viewport_entropy_toolkit import Vector, ValidationError


@dataclass
class EntropyConfig:
    """Configuration for entropy calculations.

    Attributes:
        fov_angle (float): Field of view angle in degrees.
        use_weight_distribution (bool): Whether to use weighted distribution.
        sigma (float): Bandwidth parameter for Gaussian smoothing.
        use_gaussian_normalization (bool): Whether to include 1/2πσ² normalization constant.
    """
    fov_angle: float = 120.0
    use_weight_distribution: bool = True
    sigma: float = 1.0  # Default bandwidth for moderate smoothing
    use_gaussian_normalization: bool = False  # Default to simpler calculation

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        if not 0 < self.fov_angle <= 360:
            raise ValidationError("FOV angle must be between 0 and 360 degrees")
        if self.sigma <= 0:
            raise ValidationError("Sigma must be positive")


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




def _calculate_gaussian_weight(
    distance: float,
    sigma: float,
    use_normalization: bool = False
) -> float:
    """Calculate Gaussian weight for a given distance.
    
    Args:
        distance (float): Distance between points.
        sigma (float): Bandwidth parameter for Gaussian smoothing.
        use_normalization (bool): Whether to include normalization constant.
    
    Returns:
        float: Calculated Gaussian weight.
    """
    squared_dist = distance ** 2
    
    # Calculate exponential term
    exp_term = np.exp(-squared_dist / (2 * sigma ** 2))
    
    if use_normalization:
        # Include normalization constant 1/2πσ²
        norm_constant = 1 / (2 * np.pi * sigma ** 2)
        return norm_constant * exp_term
    
    return exp_term



def calculate_tile_weights(
    vector: Vector,
    tile_centers: List[Vector],
    config: EntropyConfig
) -> Dict[Vector, float]:
    """Calculates weight distribution across tiles using Gaussian smoothing.
    
    The Gaussian kernel can be applied with or without the normalization constant
    (1/2πσ²). When use_gaussian_normalization is True, the full Gaussian formula
    is used. When False, only the exponential term is used, which is sufficient
    for most purposes since the weights are normalized at the end anyway.
    
    When use_weight_distribution is False, the nearest tile gets weight 1.0 and
    all other tiles get weight 0.0.
    
    Args:
        vector: Input vector.
        tile_centers: List of tile center vectors.
        config: Entropy calculation configuration.
    
    Returns:
        Dict[Vector, float]: Dictionary mapping tile centers to Gaussian weights.
        When use_weight_distribution is False, only nearest tile gets weight 1.0,
        all others get 0.0.
    
    Raises:
        ValidationError: If vector or tile_centers are invalid.
    """
    if not tile_centers:
        raise ValidationError("Empty tile_centers list")
    
    weights = {}
    
    # Calculate angular distances
    try:
        distances = find_angular_distances(vector, tile_centers)
    except Exception as e:
        raise ValidationError(f"Error calculating angular distances: {str(e)}")
    
    if config.use_weight_distribution:
        # Apply Gaussian kernel to all distances
        for tile_idx, distance in distances:
            tile = tile_centers[int(tile_idx)]
            
            # Calculate Gaussian weight
            weight = _calculate_gaussian_weight(
                distance=distance,
                sigma=config.sigma,
                use_normalization=config.use_gaussian_normalization
            )
            
            weights[tile] = weight
    else:
        # Find the nearest tile
        nearest_idx = int(distances[np.argmin(distances[:, 1])][0])
        
        # Assign weights to all tiles
        for i, tile in enumerate(tile_centers):
            weights[tile] = 1.0 if i == nearest_idx else 0.0
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValidationError("Total weight is zero or negative")
        
    for tile in weights:
        weights[tile] /= total_weight
    
    return weights



def compute_spatial_entropy(
    vector_dict: Dict[str, Vector],
    tile_centers: List[Vector],
    config: EntropyConfig
) -> Tuple[float, Dict[Vector, float], Dict[str, int]]:
    """Computes spatial entropy for a set of vectors using fixation counts and Gaussian smoothing.

    The computation follows these steps:
    1. Count fixations per tile (N(i,j))
    2. Calculate initial probabilities p(i,j) = N(i,j)/N_total
    3. Apply Gaussian smoothing to probabilities
    4. Calculate entropy from smoothed distribution

    Args:
        vector_dict: Dictionary mapping identifiers to vectors.
        tile_centers: List of tile center vectors.
        config: Entropy calculation configuration.

    Returns:
        Tuple containing:
        - float: Normalized spatial entropy.
        - Dict[Vector, float]: Smoothed probability distribution.
        - Dict[str, int]: Tile assignments for each vector.

    Raises:
        ValidationError: If input data is invalid.
    """
    if not vector_dict:
        raise ValidationError("Empty vector dictionary")
    if not tile_centers:
        raise ValidationError("No tile centers provided")

    # Initialize counts and assignments
    fixation_counts: Dict[Vector, int] = {tile: 0 for tile in tile_centers}
    tile_assignments: Dict[str, int] = {}
    n_total = 0

    # Count fixations per tile
    for identifier, vector in vector_dict.items():
        if vector is None:
            continue

        # Find nearest tile for this fixation
        distances = find_angular_distances(vector, tile_centers)
        nearest_idx = int(distances[np.argmin(distances[:, 1])][0])
        nearest_tile = tile_centers[nearest_idx]

        # Record assignment and increment count
        tile_assignments[identifier] = nearest_idx
        fixation_counts[nearest_tile] += 1
        n_total += 1

    if n_total == 0:
        raise ValidationError("No valid fixations found")

    # Calculate initial probabilities
    initial_probabilities: Dict[Vector, float] = {
        tile: count/n_total
        for tile, count in fixation_counts.items()
    }

    # Apply Gaussian smoothing to probabilities if enabled
    smoothed_probabilities: Dict[Vector, float] = {}
    if config.use_weight_distribution:
        for target_tile in tile_centers:
            smoothed_prob = 0.0
            for source_tile, prob in initial_probabilities.items():
                # Calculate distance between tiles
                distance = vector_angle_distance(target_tile, source_tile)
                # Apply Gaussian weight
                weight = _calculate_gaussian_weight(
                    distance=distance,
                    sigma=config.sigma,
                    use_normalization=config.use_gaussian_normalization
                )
                smoothed_prob += prob * weight
            smoothed_probabilities[target_tile] = smoothed_prob
    else:
        smoothed_probabilities = initial_probabilities

    # Normalize smoothed probabilities
    total_prob = sum(smoothed_probabilities.values())
    if total_prob > 0:
        for tile in smoothed_probabilities:
            smoothed_probabilities[tile] /= total_prob

    # Calculate entropy from smoothed probabilities
    spatial_entropy = 0.0
    for prob in smoothed_probabilities.values():
        if prob > 0:  # Avoid log(0)
            spatial_entropy -= prob * np.log2(prob)

    # Calculate maximum possible entropy
    num_tiles = len(tile_centers)
    max_proportion = 1.0 / num_tiles
    max_entropy = -num_tiles * max_proportion * np.log2(max_proportion)

    # Normalize entropy
    normalized_entropy = spatial_entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy, smoothed_probabilities, tile_assignments
