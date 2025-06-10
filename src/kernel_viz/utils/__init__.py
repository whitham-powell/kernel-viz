"""
Utility functions for kernel methods.

This module provides data transformation utilities and helper functions
for kernel-based machine learning algorithms.
"""

from .transforms import (
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

__all__: list[str] = [
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
