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
  - [x] Juptyer notebook for kernelized perceptron usage examples

  - **Visualizations**
    - [x] Multiple component animations
    - [x] Single component animations
    - [x] Kernel response surface visualization (with colorbar)
    - [ ] Kernel matrix heatmap animation (static version exists)
    - [x] Alpha evolution component
    - [x] 2D decision boundary plotting

- **Documentation**
  - [x] Add usage examples
  - [x] Provide step-by-step instructions for running the code
  - [x] Include mathematical derivations or references for kernelized methods

- **Testing**
  - [x] Comprehensive unit tests for all kernels
  - [x] Add tests for visualizations

# Future Work
- **Core Extensions**
  - [ ] Implement kernelized PCA
  - [ ] Implement kernelized SVM
  - [ ] Implement kernelized K-means
  - [ ] Explore other kernelized methods (e.g., kernelized regression)

- **Performance Features** (New)
  - [x] Optimized kernel response surface visualization with ~2x speedup
  - [x] Vectorized kernel computations for common kernels
  - [x] Pre-computed kernel matrices with caching
  - [x] Configurable grid resolution with interpolation

- **Advanced Visualizations**
  - [x] Evolution of decision boundaries during training
  - [x] Visualization of support vectors
  - [ ] Interactive visualizations for parameter adjustments
  - [ ] 3D visualization of decision surfaces
  - [ ] Loss/accuracy tracking during training
  - [x] Performance optimizations for large datasets
## Usage

### Quick Start with Makefile
For convenience, common tasks are available via Makefile shortcuts:

```bash
# Show all available commands
make help

# Setup environment
make env           # install package with dev dependencies

# Run tests
make test          # run all tests (includes matplotlib plot comparison)
make test-cov      # with coverage
make test-simple   # without matplotlib comparison (faster)

# Dependency management
make add dep=package_name      # add new dependency
make add-dev dep=package_name  # add new dev dependency

# Setup utilities
make pre-commit    # install pre-commit hooks

# Run demos
make demo

# Clean cache files
make clean
```

### Direct Commands
You can also run commands directly using uv:

```bash
# Running Tests
uv run pytest                    # Run all tests
uv run pytest --cov             # Run tests with coverage
uv run pytest tests/test_kernels.py  # Run specific test file
uv run pytest --mpl             # Run tests with matplotlib plot comparison

# Code Quality (handled automatically by pre-commit on commit)
uvx pre-commit run --all-files           # Manually run all hooks
uvx pre-commit install                   # Install pre-commit hooks

# Running Demos
uv run python presentation_demos.py
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
