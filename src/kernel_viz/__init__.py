"""
Kernel Visualization Framework for Statistical Learning

A comprehensive framework for kernelized machine learning algorithms with
visualization capabilities, including perceptron, PCA, SVM, and K-means.
"""

from kernel_viz.algorithms.perceptron import (
    PerceptronLogger,
    kernelized_perceptron,
    predict,
)
from kernel_viz.kernels import (
    additive_kernel,
    affine_kernel,
    exponential_kernel,
    laplacian_kernel,
    linear_kernel,
    multiplicative_kernel,
    polynomial_kernel,
    quadratic_kernel,
    rbf_gaussian_kernel,
)
from kernel_viz.utils.transforms import (
    affine_transform,
    augment_to_3d,
    composite_transform,
    min_max_scale,
    normalize,
    polynomial_features,
    rotate2D,
    scale,
    shift,
)
from kernel_viz.visualization.core import (
    PerceptronVisualizer,
    compute_decision_boundary,
)

__version__ = "0.1.0"
__all__ = [
    # Algorithms
    "kernelized_perceptron",
    "predict",
    "PerceptronLogger",
    # Kernels
    "linear_kernel",
    "affine_kernel",
    "quadratic_kernel",
    "polynomial_kernel",
    "rbf_gaussian_kernel",
    "exponential_kernel",
    "laplacian_kernel",
    "additive_kernel",
    "multiplicative_kernel",
    # Visualization
    "PerceptronVisualizer",
    "compute_decision_boundary",
    # Transforms
    "affine_transform",
    "augment_to_3d",
    "composite_transform",
    "min_max_scale",
    "normalize",
    "polynomial_features",
    "rotate2D",
    "scale",
    "shift",
]
