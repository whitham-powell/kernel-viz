# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project uses Python 3.13 with pipenv for dependency management. Set up the environment with:
```bash
pipenv install --dev
pipenv shell
```

## Common Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov

# Run specific test file
pytest tests/test_kernels.py

# Run tests with matplotlib plot comparison (configured in setup.cfg)
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

## Architecture Overview

This is a kernel visualization framework for statistical learning, specifically implementing a kernelized perceptron with comprehensive visualization capabilities.

### Core Components

**src/kernelized_perceptron.py**: Main algorithm implementation
- `kernelized_perceptron()` function: Core training algorithm
- `PerceptronLogger`: Tracks training metrics (alphas, misclassifications, kernel matrices)

**src/kernels.py**: Kernel function implementations
- Contains linear, affine, quadratic, polynomial, RBF, and Laplacian kernels
- All kernels follow signature: `kernel(x, y, **params) -> Union[float, ArrayLike]`

**src/kernel_visualizer.py**: Visualization engine
- `PerceptronVisualizer`: Main visualization class with multiple animation components
- `compute_decision_boundary()`: Generates decision boundaries for plotting
- Supports single/multi-component animations, alpha evolution, and 2D decision boundaries

**src/transforms.py**: Data transformation utilities for preprocessing

### Key Design Patterns

- All visualization components are dataclasses with consistent interfaces
- Kernels are implemented as standalone functions with optional parameters
- The logger captures training state for post-hoc analysis and visualization
- Decision boundary computation supports fixed dimensions for high-dimensional data visualization

### Testing Configuration

Tests use pytest with matplotlib plot comparison (pytest-mpl). Test configuration in setup.cfg includes:
- `--mpl` flag for plot comparisons
- Type checking with mypy (strict mode)
- Code formatting with black (88 character line length)
- Import sorting with isort

### Dependencies

Core: numpy, matplotlib, scikit-learn, scipy, pandas, seaborn
Video: ffmpeg-python (requires system ffmpeg for animation export)
Dev: pytest, black, flake8, mypy, isort, bandit
