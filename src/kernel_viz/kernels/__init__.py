"""
Kernel functions for machine learning algorithms.

This module provides various kernel functions including linear, polynomial,
RBF, and composite kernels for use in kernelized algorithms.
"""

from .base import (
    affine_kernel,
    exponential_kernel,
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    quadratic_kernel,
    rbf_gaussian_kernel,
)
from .composite import additive_kernel, multiplicative_kernel

__all__ = [
    # Basic kernels
    "linear_kernel",
    "affine_kernel",
    "quadratic_kernel",
    "polynomial_kernel",
    "rbf_gaussian_kernel",
    "exponential_kernel",
    "laplacian_kernel",
    # Composite kernels
    "additive_kernel",
    "multiplicative_kernel",
]
