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

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from dataclasses import dataclass

from viewport_entropy_toolkit import Vector, RadialPoint, ValidationError, convert_vectors_to_coordinates


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