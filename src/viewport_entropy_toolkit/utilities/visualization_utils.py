"""Visualization utilities module for spatial entropy analysis.

This module provides utilities for creating visualizations of spatial entropy analysis
results, including heatmaps, trajectory plots, and animations.

Classes:
    VisualizationConfig: Configuration for visualization parameters.
    PlotManager: Manages plot creation and updates.

Functions:
    setup_plot: Configures matplotlib plot with given parameters.
    create_animation: Creates animation from trajectory data.
    save_video: Saves animation as video file.
    generate_color_map: Generates colors for heatmap visualization.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from dataclasses import dataclass

from viewport_entropy_toolkit import Vector, RadialPoint, ValidationError, convert_vectors_to_coordinates
from entropy_utils import generate_fibonacci_lattice
from data_utils import normalize, get_line_segment, find_perpendicular_on_tangent_plane, great_circle_intersection, find_nearest_point,spherical_interpolation

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters.
    
    Attributes:
        figure_size (Tuple[int, int]): Size of the figure in inches.
        fov_point_size (int): Size of FOV points in scatter plot.
        tile_point_size (int): Size of tile points in scatter plot.
        fps (int): Frames per second for video output.
        dpi (int): DPI for video output.
    """
    figure_size: Tuple[int, int] = (12, 6)
    fov_point_size: int = 10
    tile_point_size: int = 40
    fps: int = 10
    dpi: int = 100

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        if any(x <= 0 for x in self.figure_size):
            raise ValidationError("Figure dimensions must be positive")
        if self.fov_point_size <= 0:
            raise ValidationError("FOV point size must be positive")
        if self.tile_point_size <= 0:
            raise ValidationError("Tile point size must be positive")
        if self.fps <= 0:
            raise ValidationError("FPS must be positive")
        if self.dpi <= 0:
            raise ValidationError("DPI must be positive")


class PlotManager:
    """Manages plot creation and updates for spatial entropy visualization.
    
    Attributes:
        config (VisualizationConfig): Visualization configuration.
        fig (Figure): Matplotlib figure object.
        ax (Axes): Matplotlib axes object.
        time_text: Text object for displaying time and entropy information.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initializes the PlotManager.
        
        Args:
            config: Optional visualization configuration.
        """
        self.config = config or VisualizationConfig()
        self.fig, self.ax = plt.subplots(figsize=self.config.figure_size)
        self.time_text = None
        self._setup_plot()
        print("Fix applied and code reached here")

    def _setup_plot(self) -> None:
        """Sets up the initial plot configuration."""
        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)
        self.ax.set_xlabel("Longitude (degrees)")
        self.ax.set_ylabel("Latitude (degrees)")
        self.ax.grid(True)
        self.time_text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes)

    def clear_plot(self) -> None:
        """Clears the current plot."""
        self.ax.clear()
        self._setup_plot()

    def update_frame(
        self,
        time_index: int,
        points_data: pd.DataFrame,
        tile_centers: np.ndarray,
        tile_weights: Dict[Vector, float],
        entropy_data: pd.DataFrame
    ) -> List[Any]:
        """Updates the plot for animation frame.
        
        Args:
            time_index: Current time index.
            points_data: DataFrame containing point trajectories.
            tile_centers: Array of tile center coordinates.
            tile_weights: Dictionary of tile weights.
            entropy_data: DataFrame containing entropy values.
        
        Returns:
            List of plot elements to update.
        """
        self.clear_plot()
        
        if time_index >= len(entropy_data):
            return [self.time_text]
        
        points_row = points_data.iloc[time_index]
        points_list = []
        colors = []
        
        # Process FOV points
        for col_name in points_data.columns:
            if col_name != "time" and points_row[col_name] is not None:
                point = points_row[col_name]
                if isinstance(point, RadialPoint):
                    points_list.append([point.lon, point.lat])

        # Convert tile centers to plottable coordinates using new function
        try:
            tile_lons, tile_lats = convert_vectors_to_coordinates(tile_centers)
        except ValidationError as e:
            logger.error(f"Failed to convert tile centers: {str(e)}")
            return [self.time_text]
        
        # Process tile colors
        largest_possible_weight = len(points_list)
        for tile_center in tile_centers:
            vector_key = Vector(x=tile_center.x, y=tile_center.y, z=tile_center.z)
            weight = tile_weights.get(vector_key, 0)
            intensity = weight / largest_possible_weight if largest_possible_weight > 0 else 0
            colors.append(self._get_color_from_intensity(intensity))
        
        # Plot points
        if points_list:
            points_array = np.array(points_list)
            self.ax.scatter(
                points_array[:, 0],
                points_array[:, 1],
                color='black',
                label="FOVs",
                s=self.config.fov_point_size
            )
        
        # Plot tile centers
        self.ax.scatter(
            tile_lons,
            tile_lats,
            color=colors,
            label="Fibonacci Lattice",
            s=self.config.tile_point_size
        )
        
        # Update text
        self._update_text(time_index, entropy_data)
        self.ax.legend()
        
        return [self.time_text]

    def _update_text(self, time_index: int, entropy_data: pd.DataFrame) -> None:
        """Updates the time and entropy text display.
        
        Args:
            time_index: Current time index.
            entropy_data: DataFrame containing entropy values.
        """
        row = entropy_data.iloc[time_index]
        text = (
            f"Entropy: {row['entropy']:.2f}\n"
            f"Time: {row['time']:.1f}"
        )
        self.time_text.set_text(text)

    @staticmethod
    def _get_color_from_intensity(intensity: float) -> Tuple[float, float, float, float]:
        """Gets color from intensity value.
        
        Args:
            intensity: Value between 0 and 1.
        
        Returns:
            RGBA color tuple.
        """
        intensity = np.clip(intensity, 0, 1)
        grey_intensity = 0.8
        red = (intensity * (1 - grey_intensity)) + grey_intensity
        green = blue = grey_intensity - (intensity * grey_intensity)
        return (red, green, blue, 1.0)


def create_animation(
    plot_manager: PlotManager,
    points_data: pd.DataFrame,
    tile_centers: np.ndarray,
    tile_weights: Dict[int, Dict[Vector, float]],
    entropy_data: pd.DataFrame
) -> FuncAnimation:
    """Creates animation of spatial entropy visualization.
    
    Args:
        plot_manager: PlotManager instance.
        points_data: DataFrame containing point trajectories.
        tile_centers: Array of tile center coordinates.
        tile_weights: Dictionary mapping time index to tile weights.
        entropy_data: DataFrame containing entropy values.
    
    Returns:
        FuncAnimation object.
    """
    def animate(time_index: int) -> List[Any]:
        return plot_manager.update_frame(
            time_index,
            points_data,
            tile_centers,
            tile_weights[time_index],
            entropy_data
        )
    
    def init() -> List[Any]:
        plot_manager.clear_plot()
        return [plot_manager.time_text]
    
    return FuncAnimation(
        plot_manager.fig,
        animate,
        init_func=init,
        frames=len(entropy_data),
        blit=True
    )


def save_video(
    animation: FuncAnimation,
    output_path: Path,
    config: Optional[VisualizationConfig] = None
) -> None:
    """Saves animation as video file.
    
    Args:
        animation: FuncAnimation object.
        output_path: Path for output video file.
        config: Optional visualization configuration.
    """
    config = config or VisualizationConfig()
    writer = FFMpegWriter(fps=config.fps)
    
    try:
        animation.save(
            str(output_path),
            writer=writer,
            dpi=config.dpi
        )
        print(f"Video saved to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving video: {str(e)}")

def save_graph(
        entropy_values: List[float],
        time_values: List[float],
        output_path: Path,
        config: Optional[VisualizationConfig] = None
) -> None:
    """Saves graph as png file.
    
    Args:
        entropy_value: The list of entropy values.
        output_path: Path for output video file.
        config: Optional visualization configuration.
    """
    
    if (len(entropy_values) != len(time_values)):
        raise ValueError("Entropy values length must match time values length!")

    plt.figure(figsize=(10, 6))
    plt.plot(time_values, entropy_values, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.title('Entropy over Time')
    plt.grid(True)

    # Save the figure to a file
    plt.savefig(output_path)

    # Show the plot (optional, if you want to display it as well)
    plt.show()

    # Close the figure to free up memory
    plt.close()

def save_tiling_visualization_video(tile_count: int, output_path: Path, horizontal_pan: bool=True, vertical_pan: bool=True):
    """Saves a video of the tiling on a sphere for fibonacci lattice.
    
    Args:
        entropy_value: The list of entropy values.
        output_path: Path for output video file.
        config: Optional visualization configuration.
    """

    if (tile_count <= 0):
        raise ValidationError("Tile counts cannot be less than 1 for to visualize tiling!")

    if (horizontal_pan is False and vertical_pan is False):
        raise ValidationError("Video must pan horizontally or vertically or both!")

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


    # Convert spherical coordinates to Cartesian coordinates for plotting
    x = [vec.x for vec in tile_centers_vectors]
    y = [vec.y for vec in tile_centers_vectors]
    z = [vec.z for vec in tile_centers_vectors]

    # Extract lines for tile boundary
    tile_boundary_list = []
    for boundaries in tile_boundaries.values():
        for boundary in boundaries:
            tile_boundary_list.append(boundary)

    pv.start_xvfb()  # Start the virtual framebuffer

    sphere_radius = 1 # Change the radius here
    sphere_opacity = 0.3  # Change the opacity here
    sphere = pv.Sphere(radius=sphere_radius)
    sphere.opacity = sphere_opacity
    sphere.color = 'grey'

    # Generate points for each arc in the boundaries
    num_points = 50  # Number of points on the arc
    t_values = np.linspace(0, 1, num_points)

    arc_points_list = []
    line_segments = [] #redefine line_segments.
    line_segment_count = 0 #keep track of segment number.

    for boundary in tile_boundary_list:
        vec_start, vec_end = boundary
        arc_points = np.array([spherical_interpolation(vec_start, vec_end, t) for t in t_values])
        for i in range(len(arc_points) - 1):
            arc_points_list.extend(arc_points[i:i+2]) #add the two points of the segment.
            line_segments.append([2, line_segment_count*2, line_segment_count*2+1]) #create a line segment.
            line_segment_count +=1

    # Create PolyData for lines
    lines = pv.PolyData(np.array(arc_points_list)) #create the points.
    lines.lines = np.array(line_segments).flatten() #define the lines.

    # Create PolyData for tile centers
    points = pv.PolyData(np.column_stack((x, y, z)))
    points['colors'] = np.array([[255, 0, 0]] * len(x))  # Red points

    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(sphere)
    plotter.add_mesh(lines, color='black', line_width=2)
    plotter.add_mesh(points, color='red', point_size=10) #plot tile centers

    # Set view and remove axes
    plotter.camera.position = (0, 0, 5) #adjust camera location.
    plotter.camera.up = (0, 1, 0)
    plotter.camera.focal_point = (0,0,0)

    plotter.enable_parallel_projection()
    plotter.show_axes_all()
    plotter.remove_bounds_axes()

    # Open movie file
    video_file_name = os.path.join(str(output_path), f'fibonacci_lattice-{tile_count}_tiles.mp4')
    plotter.open_movie(video_file_name)


    horizontal_pan_val = 0
    vertical_pan_val = 0

    if (horizontal_pan and vertical_pan):
        horizontal_pan_val = 0.5
        vertical_pan_val = 0.5
    elif (horizontal_pan):
        horizontal_pan_val = 1
    elif (vertical_pan):
        vertical_pan_val = 1

    # Rotate camera and write frames
    for i in range(180):
        plotter.camera.azimuth += horizontal_pan_val  # Rotate camera 1 degree
        plotter.camera.elevation += vertical_pan_val
        plotter.write_frame()

    plotter.close()

    print(f"Video saved: {video_file_name}")