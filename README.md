# Kernel Visualizer

A kernel visualization framework for statistical learning, implementing a kernelized perceptron with comprehensive visualization capabilities for STAT 671: Statistical Learning I at Portland State University.

## Installation

This project uses Python 3.13+ with [uv](https://docs.astral.sh/uv/) for fast dependency management.

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (optional, for saving animations as videos)

```bash
# Install ffmpeg (optional, for video export)
sudo apt-get install ffmpeg  # Ubuntu/Debian
# or
brew install ffmpeg         # macOS
```

### Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project
git clone <repository-url>
cd kernel-viz-public

# Create virtual environment and install dependencies
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate
```

# To-Do
- [ ] **Core Implementation**
  - [ ] Juptyer notebook for kernelized perceptron usage examples

  - **Visualizations**
    - [x] Multiple component animations
    - [x] Single component animations
    - [ ] Kernel response component - WIP
    - [ ] Kernel matrix heatmap
    - [x] Alpha evolution component
    - [x] 2D decision boundary plotting

- **Documentation**
  - [ ] Add usage examples
  - [ ] Provide step-by-step instructions for running the code
  - [ ] Include mathematical derivations or references for kernelized methods

- **Testing**
  - [ ] Comprehensive unit tests for all kernels
  - [ ] Add tests for visualizations

# Future Work
- **Core Extensions**
  - [ ] Implement kernelized PCA
  - [ ] Implement kernelized SVM
  - [ ] Implement kernelized K-means
  - [ ] Explore other kernelized methods (e.g., kernelized regression)

- **Advanced Visualizations**
  - [ ] Evolution of decision boundaries during training
  - [ ] Visualization of support vectors
  - [ ] Interactive visualizations for parameter adjustments
  - [ ] 3D visualization of decision surfaces
  - [ ] Loss/accuracy tracking during training
  - [ ] Performance optimizations for large datasets
## Usage

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov

# Run specific test file
pytest tests/test_kernels.py

# Run tests with matplotlib plot comparison
pytest --mpl
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Running Demos
```bash
python presentation_demos.py
```

## Features

- **Kernel Functions**: Linear, affine, quadratic, polynomial, RBF (Gaussian), exponential, and Laplacian kernels
- **Kernelized Perceptron**: Complete implementation with training metrics logging
- **Comprehensive Visualizations**:
  - Decision boundary plotting
  - Alpha evolution tracking
  - Multi-component animations
  - Training progress visualization
- **Modern Development**: Full type hints, comprehensive testing, and automated code quality checks
