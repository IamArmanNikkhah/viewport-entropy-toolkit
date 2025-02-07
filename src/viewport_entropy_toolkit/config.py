"""Configuration module for spatial entropy analysis.

This module defines constants and configuration classes used throughout the
spatial entropy analysis system.

Classes:
    AnalyzerConfig: Configuration for spatial entropy analysis.
    
Constants:
    DEFAULT_VIDEO_DIMENSIONS: Default dimensions for video processing.
    DEFAULT_TILE_COUNTS: Default number of tiles for different analysis scales.
    DEFAULT_OUTPUT_FORMATS: Default output file formats and extensions.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from pathlib import Path

from viewport_entropy_toolkit.utilities import EntropyConfig, VisualizationConfig


# Default video dimensions
DEFAULT_VIDEO_DIMENSIONS = {
    'width': 100,
    'height': 200
}

# Default tile counts for multi-scale analysis
DEFAULT_TILE_COUNTS = [20, 50, 100, 250, 1000]

# Default output formats
DEFAULT_OUTPUT_FORMATS = {
    'video': '.mp4',
    'data': '.csv',
    'plot': '.png'
}


@dataclass
class AnalyzerConfig:
    """Configuration for entropy analysis using fibonacci lattice tiling.
    
    Attributes:
        video_width (int): Width of the video in pixels.
        video_height (int): Height of the video in pixels.
        tile_counts (List[int]): List of tile counts for analysis.
        output_dir (Path): Directory for output files.
        entropy_config (EntropyConfig): Configuration for entropy calculations.
        visualization_config (VisualizationConfig): Configuration for visualization.
        use_weight_distribution (bool): Whether to use weighted distribution.
    """
    video_width: int = DEFAULT_VIDEO_DIMENSIONS['width']
    video_height: int = DEFAULT_VIDEO_DIMENSIONS['height']
    tile_counts: List[int] = field(default_factory=lambda: DEFAULT_TILE_COUNTS)
    output_dir: Path = Path('output')
    entropy_config: EntropyConfig = field(default_factory=EntropyConfig)
    visualization_config: VisualizationConfig = field(default_factory=VisualizationConfig)
    use_weight_distribution: bool = True
    

    def __post_init__(self) -> None:
        """Validates configuration and creates output directory."""
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("Video dimensions must be positive")
        if not self.tile_counts:
            raise ValueError("Must specify at least one tile count")
        if any(count <= 0 for count in self.tile_counts):
            raise ValueError("Tile counts must be positive")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, base_name: str, extension: str) -> Path:
        """Generates output file path.
        
        Args:
            base_name (str): Base name for the output file.
            extension (str): File extension.
            
        Returns:
            Path: Complete output file path.
        """
        return self.output_dir / f"{base_name}{extension}"
    
@dataclass
class NaiveAnalyzerConfig:
    """Configuration for spatial entropy analysis using latitude-longitude cut tiling.
    
    Attributes:
        video_width (int): Width of the video in pixels.
        video_height (int): Height of the video in pixels.
        tile_counts (List[int]): List of tile counts for analysis.
        output_dir (Path): Directory for output files.
        entropy_config (EntropyConfig): Configuration for entropy calculations.
        visualization_config (VisualizationConfig): Configuration for visualization.
        use_weight_distribution (bool): Whether to use weighted distribution.
    """
    video_width: int = DEFAULT_VIDEO_DIMENSIONS['width']
    video_height: int = DEFAULT_VIDEO_DIMENSIONS['height']
    output_dir: Path = Path('output')
    entropy_config: EntropyConfig = field(default_factory=EntropyConfig)
    visualization_config: VisualizationConfig = field(default_factory=VisualizationConfig)
    use_weight_distribution: bool = False
    
    tile_width: int = -1
    tile_height: int = -1

    def __post_init__(self) -> None:
        """Validates configuration and creates output directory."""
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("Video dimensions must be positive")
        if not self.tile_height or self.tile_width:
            raise ValueError("Must specify both tile_height and tile_width")
        if any(count <= 0 for count in self.tile_counts):
            raise ValueError("Tile counts must be positive")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, base_name: str, extension: str) -> Path:
        """Generates output file path.
        
        Args:
            base_name (str): Base name for the output file.
            extension (str): File extension.
            
        Returns:
            Path: Complete output file path.
        """
        return self.output_dir / f"{base_name}{extension}"
