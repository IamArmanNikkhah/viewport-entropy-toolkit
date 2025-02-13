# viewport-entropy-toolkit

A Python package for analyzing spatial entropy in 360-degree video viewport trajectories. This tool helps quantify and visualize how users' attention is distributed when watching 360-degree videos.

## Features

- Analyze viewport center trajectories (VCT) from 360-degree video viewing data
- Calculate spatial and transition entropy using configurable tile-based analysis
- Generate visualizations including heatmaps and animations
- Support for batch processing of multiple videos
- Configurable analysis parameters and visualization options
- Comprehensive error handling and validation

## Installation

### Requirements

- Python 3.8 or higher
- numpy
- pandas
- matplotlib
- ffmpeg (for video generation)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/IamArmanNikkhah/viewport-entropy-toolkit.git
cd spatial-entropy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

Here's a simple example to get you started:

```python
from spatial_entropy.analyzers.spatial_entropy import SpatialEntropyAnalyzer
from pathlib import Path

# Create analyzer with default configuration
analyzer = SpatialEntropyAnalyzer()

# Run analysis on a directory of VCT files
input_dir = Path("path/to/vct/files")
analyzer.run_analysis(input_dir, output_prefix="my_analysis")
```

## Usage

### Basic Usage

The simplest way to use the analyzer:

```python
from spatial_entropy.analyzers.spatial_entropy import SpatialEntropyAnalyzer
from pathlib import Path

# Create and run analyzer
analyzer = SpatialEntropyAnalyzer()
analyzer.run_analysis(Path("data/video1"))
```

### Custom Configuration

Customize the analysis parameters:

```python
from spatial_entropy.config import AnalyzerConfig, EntropyConfig
from spatial_entropy.analyzers.spatial_entropy import SpatialEntropyAnalyzer

# Create custom configuration
config = AnalyzerConfig(
    video_width=200,
    video_height=400,
    tile_counts=[50, 100, 200],
    entropy_config=EntropyConfig(
        fov_angle=90.0,
        use_weight_distribution=True
    )
)

# Create analyzer with custom config
analyzer = SpatialEntropyAnalyzer(config)
```

### Batch Processing

Process multiple videos:

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

directories = [Path(f"data/video{i}") for i in range(1, 4)]

def process_directory(directory):
    analyzer = SpatialEntropyAnalyzer()
    analyzer.run_analysis(directory)

with ProcessPoolExecutor() as executor:
    executor.map(process_directory, directories)
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Installation Guide](docs/installation.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Example Scripts](examples/README.md)

## Project Structure

```
spatial_entropy/
├── spatial_entropy/       # Main package
│   ├── analyzers/        # Analysis implementations
│   └── utilities/        # Utility functions
├── tests/                # Test suite
├── docs/                 # Documentation
├── examples/             # Example scripts
└── data/                 # Sample datasets
```

## Input Data Format

The analyzer expects CSV files with the following columns:
- `time`: Timestamp in seconds
- `2dmu`: Normalized x-coordinate [0,1]
- `2dmv`: Normalized y-coordinate [0,1]

Example:
```csv
time,2dmu,2dmv
0.0,0.5,0.5
0.1,0.52,0.48
...
```

## Output

The analyzer generates:
1. CSV files with entropy measurements
2. Video visualizations of spatial entropy
3. Static plots of entropy over time

## Configuration Options

Key configuration parameters:

```python
AnalyzerConfig(
    video_width=100,         # Video width in pixels
    video_height=200,        # Video height in pixels
    tile_counts=[20, 50],    # Number of tiles for analysis
    use_weight_distribution=True  # Use weighted distribution
)

EntropyConfig(
    fov_angle=120.0,        # Field of view angle
    power_factor=2.0        # Weight calculation power factor
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{spatial_entropy,
  title = {Spatial Entropy Analysis},
  author = {Arman Nik Khah, Chitsein Htun},
  year = {2024},
  url = {https://github.com/IamArmanNikkhah/viewport-entropy-toolkit}
}
```

## Contact

For questions and support:
- Create an issue on GitHub
- Email: rmnnikkhah@gmail.com

## Acknowledgments

This project is based on research from [Your Institution] and builds upon the concepts presented in [Reference Paper].
