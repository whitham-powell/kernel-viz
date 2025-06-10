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

This is a kernel visualization framework for statistical learning, organized as a modular package structure to support extensibility and future algorithm implementations.

### Package Structure

```
src/kernel_viz/
├── algorithms/          # Kernelized machine learning algorithms
│   ├── perceptron.py    # Kernelized perceptron implementation
│   ├── pca.py           # Future: Kernelized PCA
│   ├── svm.py           # Future: Kernelized SVM
│   └── kmeans.py        # Future: Kernelized K-means
├── kernels/             # Kernel function implementations
│   ├── base.py          # Basic kernels (linear, polynomial, RBF, etc.)
│   └── composite.py     # Composite kernels (additive, multiplicative)
├── visualization/       # Visualization components
│   ├── core.py          # Main visualization engine
│   ├── boundaries.py    # Future: Decision boundary utilities
│   ├── animations.py    # Future: Animation components
│   └── interactive.py   # Future: Interactive visualizations
└── utils/               # Utility functions
    └── transforms.py    # Data transformation utilities
```

### Core Components

**kernel_viz.algorithms.perceptron**: Main algorithm implementation
- `kernelized_perceptron()` function: Core training algorithm
- `PerceptronLogger`: Tracks training metrics (alphas, misclassifications, kernel matrices)

**kernel_viz.kernels**: Kernel function implementations
- `base.py`: Contains linear, affine, quadratic, polynomial, RBF, and Laplacian kernels
- `composite.py`: Additive and multiplicative kernel combinations
- All kernels follow signature: `kernel(x, y, **params) -> Union[float, ArrayLike]`

**kernel_viz.visualization.core**: Visualization engine
- `PerceptronVisualizer`: Main visualization class with multiple animation components
- `compute_decision_boundary()`: Generates decision boundaries for plotting
- Supports single/multi-component animations, alpha evolution, and 2D decision boundaries

**kernel_viz.utils.transforms**: Data transformation utilities for preprocessing

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

## Priority Tasks

**IMPORTANT**: Before implementing new features from README.md, check TODOS.md for critical fixes and refactoring tasks that need to be addressed first. The TODOS.md file contains:
- Critical fixes affecting correctness (HIGH priority)
- Code organization and refactoring needs (MEDIUM priority)
- External dependency removals (LOW priority)

Always prioritize fixing existing issues before adding new features.

## Recent Work Summary (Last Updated: 2025-01-10)

### Completed Tasks
- Created Jupyter notebook for kernelized perceptron usage examples (in notebooks/)
- Fixed all critical visualization issues (frame count, error handling)
- Restored kernel response surface visualization as distinct from decision boundary
- Both visualizations now work properly with appropriate visual elements
- Split large XOR test into focused unit tests
- Parameterized kernel tests to reduce duplication (LinearKernel, AffineKernel)
- Refactored visualization components into separate files:
  - base.py: AnimationComponent dataclass
  - visualizer.py: PerceptronVisualizer core class
  - core.py: Component creation methods (attached via monkey patching)
- Implemented kernel matrix heatmap animation using factory pattern as proof of concept:
  - Created KernelMatrixHeatmapFactory in component_factory.py
  - Shows kernel matrix as heatmap with animated support vector indicators
  - Highlights rows/columns for active support vectors
  - Includes comprehensive tests (12 tests, all passing)
- Added comprehensive tests for all visualization components:
  - TestKernelMatrixComponent: Tests for the original kernel matrix component
  - TestMisclassificationTrackerComponent: Tests for misclassification tracking
  - TestDecisionBoundaryAdvanced: Advanced tests for decision boundary parameters
  - TestComputeDecisionBoundary: Tests for utility function
  - TestVisualizationEdgeCases: Edge case and error handling tests
  - Fixed matplotlib deprecation warning for contour.collections
- All tests passing (380+ total)
- Removed sklearn dependency from polynomial features:
  - Implemented custom polynomial feature generation
  - Added edge case tests (negative degree, empty features)
  - Added comparison test to verify matching sklearn behavior
- Completed comprehensive documentation:
  - Added test_kernels_comprehensive.py with 29 tests for kernel properties
  - Created docs/getting_started.md with installation and setup instructions
  - Created docs/usage_examples.md with code examples and best practices
  - Created docs/mathematical_background.md with theory and references
  - Created docs/index.md as main documentation hub

### Current Status
- **Where we left off**: Completed all documentation tasks
- Added comprehensive kernel tests (29 new tests covering edge cases and properties)
- Created full documentation suite:
  - Getting started guide with step-by-step instructions
  - Usage examples with code snippets
  - Mathematical background with derivations and references
  - Documentation index page
- All TODOs from TODOS.md are now complete

### Next Priority Tasks
1. Consider refactoring other component creation methods to use factory pattern
2. Consider performance optimizations for kernel response surface
3. Implement polynomial features without sklearn dependency (low priority)
4. Add comprehensive documentation and usage examples

## Memories

- Always review the claude.md, readme.md, and TODOS.md when starting a new session.
- Always update the claude.md and readme.md when meaningful changes have been made to the project.
- Ensure the tests pass, resolve them first in the code base assuming the code is wrong until you can prove the test itself is bad.
- Commit with a meaningful summary of what you did after each major implementation change and logical unit of complete work. Do not skip pre-commit checks to bypass errors. Resolve the errors instead.
- Assuming tests pass, commit is successful and documentation is updated and committed, push the changes to origin.
- Using git commit -am"message" is helpful to recommit files changed by the pre-commit hooks.
- You can run individual pre-commit hooks using "pre-commit run [hook-id] [options]" alternatively "uvx pre-commit run [hook-id] [options]"
- Logs and loggers in the context of this project are for tracking algorithm state and kernel state rather than logging.Logger objects in the python standard library