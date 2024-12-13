# Installation Guide

This guide provides detailed instructions for installing the Spatial Entropy Analysis package and its dependencies.

## System Requirements

### Operating System
- Linux (Ubuntu 18.04 or later recommended)
- macOS (10.14 or later)
- Windows 10 or later

### Required Software
- Python 3.8 or later
- pip (Python package installer)
- ffmpeg (for video visualization generation)
- Git (for cloning the repository)

### Hardware Requirements
- Minimum 4GB RAM (8GB recommended)
- 1GB free disk space
- CPU with x86_64 architecture

## Installing Required System Software

### Installing FFmpeg

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS (using Homebrew)
```bash
brew install ffmpeg
```

#### Windows
1. Download FFmpeg from the official website: https://ffmpeg.org/download.html
2. Extract the files to a directory (e.g., `C:\ffmpeg`)
3. Add the `bin` directory to your system's PATH environment variable

### Installing Git

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install git
```

#### macOS (using Homebrew)
```bash
brew install git
```

#### Windows
Download and install Git from: https://git-scm.com/download/win

## Python Environment Setup

### Installing Python

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.8 python3.8-dev python3-pip
```

#### macOS (using Homebrew)
```bash
brew install python@3.8
```

#### Windows
Download and install Python from: https://www.python.org/downloads/

### Setting up a Virtual Environment (Recommended)

It's recommended to install the package in a virtual environment to avoid conflicts with other Python packages.

```bash
# Create a virtual environment
python -m venv spatial-env

# Activate the virtual environment
# On Linux/macOS:
source spatial-env/bin/activate
# On Windows:
spatial-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

## Installing the Package

### Method 1: Installing from PyPI (Recommended for Users)

```bash
pip install spatial-entropy
```

### Method 2: Installing from Source (Recommended for Developers)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spatial-entropy.git
cd spatial-entropy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in editable mode:
```bash
pip install -e .
```

### Method 3: Installing with Extra Features

For development or documentation:
```bash
# For development tools
pip install spatial-entropy[dev]

# For documentation tools
pip install spatial-entropy[docs]

# For all extra features
pip install spatial-entropy[all]
```

## Verifying Installation

After installation, you can verify that everything is working correctly:

```python
# Start Python interpreter
python

# Try importing the package
>>> from spatial_entropy.analyzers.spatial_entropy import SpatialEntropyAnalyzer
>>> analyzer = SpatialEntropyAnalyzer()
```

## Common Issues and Troubleshooting

### FFmpeg Not Found
If you encounter FFmpeg-related errors:
1. Ensure FFmpeg is installed correctly
2. Verify FFmpeg is in your system's PATH
3. Try running `ffmpeg -version` in your terminal

### Import Errors
If you encounter import errors:
1. Ensure you're in the correct virtual environment
2. Verify the package is installed: `pip list | grep spatial-entropy`
3. Check Python version: `python --version`

### Installation Fails
If installation fails:
1. Update pip: `pip install --upgrade pip`
2. Install build tools:
   ```bash
   # Ubuntu/Debian
   sudo apt install python3-dev build-essential
   
   # Windows
   # Install Visual C++ Build Tools
   ```

## Next Steps

After successful installation:
1. Review the [Quick Start Guide](../README.md#quick-start)
2. Check the [API Reference](API_REFERENCE.md)
3. Try the [example scripts](../examples/)

## Getting Help

If you encounter any issues:
1. Check the [troubleshooting section](#common-issues-and-troubleshooting)
2. Create an issue on GitHub
3. Contact the maintainers at: your.email@example.com

## Updating the Package

To update to the latest version:

```bash
# For PyPI installation
pip install --upgrade spatial-entropy

# For source installation
git pull origin main
pip install -e .
```