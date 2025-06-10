# Getting Started Guide

This guide provides step-by-step instructions for setting up and running the kernel visualization framework.

## Prerequisites

- Python 3.13+ (tested with 3.13)
- Git for cloning the repository
- FFmpeg (optional, for saving animations as videos)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/kernel-viz.git
cd kernel-viz
```

### Step 2: Install uv (Package Manager)

This project uses `uv` for dependency management:

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
# Download from https://github.com/astral/uv/releases
```

### Step 3: Set Up the Environment

```bash
# Create virtual environment and install dependencies
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Step 4: Install Pre-commit Hooks (Optional)

For contributors:

```bash
uvx pre-commit install
```

## Running Your First Example

### Step 1: Create a Simple Script

Create a file `my_first_kernel_viz.py`:

```python
import numpy as np
from kernel_viz.algorithms.perceptron import kernelized_perceptron, PerceptronLogger
from kernel_viz.kernels import rbf_gaussian_kernel
from kernel_viz.visualization import PerceptronVisualizer

# Create a simple dataset
X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.float64)
y = np.array([1, 1, -1, -1], dtype=np.float64)

# Initialize logger
logger = PerceptronLogger()

# Train the model
print("Training kernelized perceptron...")
alphas = kernelized_perceptron(
    X, y,
    kernel=rbf_gaussian_kernel,
    kernel_params={'sigma': 1.0},
    max_epochs=10,
    logger=logger
)

print(f"Training complete! Alphas: {alphas}")

# Visualize results
print("Creating visualization...")
visualizer = PerceptronVisualizer()
logs = logger.get_logs()

# Add decision boundary component
component = visualizer.create_decision_boundary_component(logs)
visualizer.add_component(component)

# Show the result
visualizer.show()
```

### Step 2: Run the Script

```bash
uv run python my_first_kernel_viz.py
```

You should see:
1. Training progress in the console
2. A matplotlib window showing the decision boundary

### Step 3: Save the Visualization

Modify the script to save the output:

```python
# ... previous code ...

# Save as static image
import matplotlib.pyplot as plt
plt.savefig("decision_boundary.png", dpi=150, bbox_inches='tight')
print("Saved as decision_boundary.png")

# Create and save animation
animation = visualizer.animate(interval=500)
animation.save("training_animation.gif", writer="pillow", fps=2)
print("Saved as training_animation.gif")
```

## Running the Demo Scripts

### Basic Demos

The repository includes several demo scripts:

```bash
# Run the main presentation demos
uv run python presentation_demos.py

# Run specific examples from notebooks
uv run python notebooks/kernelized_perceptron_usage.py
```

### Interactive Jupyter Notebook

```bash
# Install Jupyter kernel
uv run python -m ipykernel install --user --name kernel-viz

# Start Jupyter
uv run jupyter notebook

# Navigate to notebooks/ directory and open examples
```

## Running Tests

### All Tests

```bash
# Using Makefile
make test

# Or directly with uv
uv run pytest
```

### Specific Test Categories

```bash
# Run kernel tests
uv run pytest tests/test_kernels.py -v

# Run visualization tests
uv run pytest tests/test_visualization.py -v

# Run with coverage
uv run pytest --cov=kernel_viz --cov-report=html
```

## Common Tasks

### 1. Visualizing XOR Problem

```python
import numpy as np
from kernel_viz.algorithms.perceptron import kernelized_perceptron, PerceptronLogger
from kernel_viz.kernels import rbf_gaussian_kernel
from kernel_viz.visualization import PerceptronVisualizer

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([1, -1, -1, 1], dtype=np.float64)

# Train with logging
logger = PerceptronLogger()
alphas = kernelized_perceptron(
    X, y,
    kernel=rbf_gaussian_kernel,
    kernel_params={'sigma': 0.5},
    max_epochs=20,
    logger=logger
)

# Visualize with multiple components
visualizer = PerceptronVisualizer()
logs = logger.get_logs()

# Add multiple visualization components
visualizer.add_component(visualizer.create_decision_boundary_component(logs))
visualizer.add_component(visualizer.create_alpha_evolution_component(logs))

# Show the visualization
visualizer.show()
```

### 2. Comparing Different Kernels

```python
from kernel_viz.kernels import linear_kernel, polynomial_kernel, rbf_gaussian_kernel

kernels = [
    ("Linear", linear_kernel, {}),
    ("Polynomial", polynomial_kernel, {"degree": 3}),
    ("RBF", rbf_gaussian_kernel, {"sigma": 1.0}),
]

for name, kernel, params in kernels:
    logger = PerceptronLogger()
    alphas = kernelized_perceptron(
        X, y,
        kernel=kernel,
        kernel_params=params,
        max_epochs=50,
        logger=logger
    )

    misclassifications = logger.get_logs()['misclassification_count']
    print(f"{name} kernel: Final misclassifications = {misclassifications[-1]}")
```

### 3. Generating Animation

```python
# Create visualizer with all components
visualizer = PerceptronVisualizer()
logs = logger.get_logs()

# Add all available components
components = [
    visualizer.create_decision_boundary_component(logs),
    visualizer.create_alpha_evolution_component(logs),
    visualizer.create_kernel_response_component(logs),
    visualizer.create_kernel_matrix_heatmap_component(logs),
]

for component in components:
    visualizer.add_component(component)

# Create animation
animation = visualizer.animate(interval=300)

# Save as video (requires ffmpeg)
animation.save("kernel_training.mp4", writer="ffmpeg", fps=5)

# Or save as GIF
animation.save("kernel_training.gif", writer="pillow", fps=2)
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'kernel_viz'**
   - Make sure you're in the project directory
   - Ensure the virtual environment is activated
   - Run `uv sync` to install the package

2. **Matplotlib not showing plots**
   - On WSL/Linux: Install `python3-tk`: `sudo apt-get install python3-tk`
   - On macOS: Use a framework build of Python
   - Alternative: Save plots instead of showing: `plt.savefig("plot.png")`

3. **Animation saving fails**
   - For MP4: Install ffmpeg: `sudo apt-get install ffmpeg`
   - For GIF: Already included (uses pillow)
   - Alternative: Use `visualizer.show()` for static view

4. **Tests failing with image comparison**
   - Run `uv run pytest --mpl-generate-path=tests/baseline` to regenerate baseline images
   - Or run tests without image comparison: `make test-simple`

### Debug Mode

Enable debug output for troubleshooting:

```python
visualizer = PerceptronVisualizer()
visualizer.set_debug_mode(True)
```

## Next Steps

1. Explore the [Usage Examples](usage_examples.md) for more advanced scenarios
2. Read about the [Mathematical Background](mathematical_background.md)
3. Check the API documentation in the source code
4. Try modifying the demos to experiment with different kernels and datasets

## Getting Help

- Check the [README.md](../README.md) for project overview
- Look at existing tests for usage patterns
- Review the source code docstrings for detailed API information
- Open an issue on GitHub for bugs or questions
