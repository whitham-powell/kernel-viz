# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project uses Python 3.13 with uv for dependency management. Set up the environment with:
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate
```

## Common Commands

Use Makefile shortcuts for convenience, or run uv commands directly.

### Testing
```bash
# Makefile shortcuts
make test          # Run all tests
make test-cov      # Run tests with coverage
make test-simple   # Run tests without matplotlib comparison

# Direct uv commands
uv run pytest                    # Run all tests
uv run pytest --cov             # Run tests with coverage
uv run pytest tests/test_kernels.py  # Run specific test file
```

### Code Quality
Code quality is handled automatically by pre-commit hooks on commit. To run manually:
```bash
# Run all pre-commit hooks
uvx pre-commit run --all-files

# Install pre-commit hooks (run once after setup)
uvx pre-commit install
```

### Running Demos
```bash
# Makefile shortcut
make demo

# Direct uv command
uv run python presentation_demos.py
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

Tests use pytest with matplotlib plot comparison (pytest-mpl). Test configuration in pyproject.toml includes:
- `--mpl` flag for plot comparisons
- Type checking with mypy (strict mode)
- Code formatting with black (88 character line length)
- Import sorting with isort

### Dependencies

Core: numpy, matplotlib, scikit-learn, scipy, pandas, seaborn
Video: ffmpeg-python (requires system ffmpeg for animation export)
Dev: pytest, pytest-cov, pytest-mpl, ipykernel

Note: pre-commit is managed via uvx (system-wide tool), not as a dev dependency.

### Pre-commit Hooks

Pre-commit is configured with the following tools:
- **Formatters**: black, isort, yamlfmt, add-trailing-comma
- **Linters**: flake8 (with flake8-pyproject), mypy
- **General**: trailing-whitespace, end-of-file-fixer, check-yaml

All tool configurations are centralized in pyproject.toml for consistency.

## Memories

- Always review the claude.md, readme.md when starting a new session.
- Always update the claude.md and readme.md when meaningful changes have been made to the project.
- Ensure the tests pass, resolve them first in the code base assuming the code is wrong until you can prove the test itself is bad.
- Commit with a meaningful summary of what you did after each major implementation change and logical unit of complete work. Do not skip pre-commit checks to bypass errors. Resolve the errors instead.
- Assuming tests pass, commit is successful and documentation is updated and committed, push the changes to origin.
- Using git commit -am"message" is helpful to recommit files changed by the pre-commit hooks.
