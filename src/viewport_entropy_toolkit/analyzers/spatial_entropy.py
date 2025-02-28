"""Spatial entropy analyzer module.

This module provides the main analyzer class for computing spatial entropy
in 360-degree video viewport data. It coordinates data processing, entropy
calculations, and visualization generation.

Classes:
    SpatialEntropyAnalyzer: Main class for spatial entropy analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from ..config import AnalyzerConfig, DEFAULT_OUTPUT_FORMATS
from ..utilities.data_utils import (
    generate_fibonacci_lattice,
    process_viewport_data,
    format_trajectory_data
)
from ..utilities.entropy_utils import (
    compute_spatial_entropy,
    EntropyConfig
)
from ..utilities.visualization_utils import (
    PlotManager,
    create_animation,
    save_video,
    save_graph
)
from ..data_types import Vector, RadialPoint, ValidationError


logger = logging.getLogger(__name__)


class SpatialEntropyAnalyzer:
    """Analyzer for computing spatial entropy in viewport trajectories.
    
    This class handles the complete pipeline of spatial entropy analysis,
    including data loading, processing, entropy calculation, and visualization.
    
    Attributes:
        config (AnalyzerConfig): Configuration for the analyzer.
        plot_manager (PlotManager): Manager for visualization.
        _data_cache (dict): Cache for processed data.
        _entropy_results (pd.DataFrame): Cached entropy results.
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Initializes the analyzer.
        
        Args:
            config: Optional configuration for the analyzer.
        """
        self.config = config or AnalyzerConfig()
        self.plot_manager = PlotManager(self.config.visualization_config)
        self._data_cache = {}
        self._entropy_results = None
        self._fibonacci_vectors = {
            count: generate_fibonacci_lattice(count)
            for count in self.config.tile_counts
        }
    
    def process_directory(self, directory: Path) -> None:
        """Processes all VCT files in a directory.
        
        Args:
            directory: Path to directory containing VCT files.
            
        Raises:
            FileNotFoundError: If directory doesn't exist.
            ValidationError: If data processing fails.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        try:
            # Process each CSV file
            trajectory_data = []
            for filepath in directory.glob("*.csv"):
                data, identifier = process_viewport_data(
                    filepath,
                    self.config.video_width,
                    self.config.video_height
                )
                trajectory_data.append((identifier, data))
            
            # Format trajectory data
            points_df, vectors_df = format_trajectory_data(trajectory_data)
            
            # Cache the processed data
            self._data_cache = {
                'points': points_df,
                'vectors': vectors_df,
                'trajectory_data': trajectory_data
            }
            
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {str(e)}")
            raise ValidationError(f"Failed to process directory: {str(e)}")
    
    def compute_entropy(self) -> pd.DataFrame:
        """Computes spatial entropy for the processed data.
        
        Returns:
            pd.DataFrame: DataFrame containing entropy results.
            
        Raises:
            ValidationError: If no data has been processed.
        """
        if not self._data_cache:
            raise ValidationError("No data available. Call process_directory first.")
        
        vectors_df = self._data_cache['vectors']
        
        entropy_results = {
            'time': [],
            'entropy': [],
            'tile_weights': [],
            'tile_assignments': []
        }
        
        # Compute entropy for each time step
        for _, row in vectors_df.iterrows():
            time = row['time']
            vector_dict = {
                col: row[col]
                for col in vectors_df.columns
                if col != 'time' and row[col] is not None
            }
            
            # Average entropy across different tile counts
            total_entropy = 0
            weights = None
            assignments = None
            
            for tile_count in self.config.tile_counts:
                tile_centers = self._fibonacci_vectors[tile_count]
                entropy, tile_weights, tile_assignments = compute_spatial_entropy(
                    vector_dict,
                    tile_centers,
                    self.config.entropy_config
                )
                total_entropy += entropy
                
                # Store weights and assignments for first tile count
                if tile_count == self.config.tile_counts[0]:
                    weights = tile_weights
                    assignments = tile_assignments
            
            avg_entropy = total_entropy / len(self.config.tile_counts)
            
            entropy_results['time'].append(time)
            entropy_results['entropy'].append(avg_entropy)
            entropy_results['tile_weights'].append(weights)
            entropy_results['tile_assignments'].append(assignments)
        
        self._entropy_results = pd.DataFrame(entropy_results)
        return self._entropy_results
    
    def create_visualization(self, base_name: str) -> None:
        """Creates and saves visualization outputs.
        
        Args:
            base_name: Base name for output files.
            
        Raises:
            ValidationError: If entropy hasn't been computed.
        """
        if self._entropy_results is None:
            raise ValidationError("No entropy results. Call compute_entropy first.")
        
        try:
            # Create graph plot
            graph_path = self.config.get_output_path(
                f"{base_name}_graph",
                DEFAULT_OUTPUT_FORMATS['plot']
            )
            save_graph(
                entropy_values=self._entropy_results['entropy'],
                time_values=self._entropy_results['time'],
                output_path=graph_path,
                config=self.config.visualization_config
            )

            # Create animation
            animation = create_animation(
                self.plot_manager,
                self._data_cache['points'],
                self._fibonacci_vectors[self.config.tile_counts[0]],
                dict(enumerate(self._entropy_results['tile_weights'])),
                self._entropy_results
            )
            
            # Save video
            video_path = self.config.get_output_path(
                base_name,
                DEFAULT_OUTPUT_FORMATS['video']
            )
            save_video(
                animation,
                video_path,
                self.config.visualization_config
            )
            
            # Save entropy data
            data_path = self.config.get_output_path(
                base_name,
                DEFAULT_OUTPUT_FORMATS['data']
            )
            self._entropy_results[['time', 'entropy']].to_csv(
                data_path,
                index=False
            )
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise RuntimeError(f"Failed to create visualization: {str(e)}")
    
    def run_analysis(self, directory: Path, output_prefix: str = "") -> None:
        """Runs the complete analysis pipeline.
        
        Args:
            directory: Directory containing VCT files.
            output_prefix: Optional prefix for output files.
            
        Raises:
            Exception: If any step of the pipeline fails.
        """
        try:
            # Process data
            self.process_directory(directory)
            
            # Compute entropy
            self.compute_entropy()
            
            # Generate base name for outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{directory.stem}_{output_prefix}_{timestamp}"
            
            # Create visualization
            self.create_visualization(base_name)
            
            logger.info(f"Analysis completed successfully: {base_name}")
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
