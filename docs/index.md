# Kernel Visualization Framework Documentation

Welcome to the documentation for the Kernel Visualization Framework, a Python package for visualizing and understanding kernelized machine learning algorithms.

## Overview

This framework provides:
- Implementation of the kernelized perceptron algorithm
- Various kernel functions (linear, polynomial, RBF, etc.)
- Rich visualization components for understanding algorithm behavior
- Extensible architecture for adding new kernels and algorithms

## Documentation Contents

### 📚 Guides

1. **[Getting Started](getting_started.md)**
   - Installation instructions
   - Running your first example
   - Common tasks and troubleshooting

2. **[Usage Examples](usage_examples.md)**
   - Quick start examples
   - Working with different kernels
   - Advanced visualization techniques
   - Real-world datasets

3. **[Mathematical Background](mathematical_background.md)**
   - Kernel function theory
   - Kernelized perceptron algorithm
   - RKHS and theoretical foundations
   - References and further reading

### 🔧 API Reference

The API documentation is available in the source code docstrings. Key modules:

- `kernel_viz.kernels` - Kernel functions
- `kernel_viz.algorithms` - Machine learning algorithms
- `kernel_viz.visualization` - Visualization components
- `kernel_viz.utils` - Utility functions

### 📓 Examples

- `notebooks/kernelized_perceptron_usage.py` - Basic usage notebook
- `presentation_demos.py` - Interactive demonstrations
- `demo_kernel_matrix_heatmap.py` - Kernel matrix visualization example

## Quick Links

- [README](../README.md) - Project overview and features
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [TODOS.md](../TODOS.md) - Current development status

## Key Concepts

### Kernels
Kernel functions measure similarity between data points and enable learning in high-dimensional feature spaces without explicit transformation.

### Kernelized Perceptron
An extension of the classical perceptron that can learn non-linear decision boundaries using kernel functions.

### Visualization Components
- **Decision Boundary**: Shows the classification boundary
- **Alpha Evolution**: Tracks coefficient changes during training
- **Kernel Response**: Visualizes the decision function surface
- **Kernel Matrix Heatmap**: Shows pairwise similarities between points

## Getting Help

1. Check the documentation guides above
2. Review the example scripts
3. Look at the test files for usage patterns
4. Open an issue on GitHub for bugs or questions

## Contributing

See [CLAUDE.md](../CLAUDE.md) for development guidelines and contribution instructions.

---

*This documentation is part of the Kernel Visualization Framework project.*
