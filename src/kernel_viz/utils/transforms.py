# transforms.py

from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def affine_transform(
    features: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.float64]:
    """Applies an affine transformation to the input features."""
    xs = np.asarray(features, dtype=np.float64)
    transform_matrix = np.asarray(A, dtype=np.float64)
    translation = np.asarray(b, dtype=np.float64)
    return (transform_matrix @ xs.T).T + translation


def normalize(features: ArrayLike) -> NDArray[np.float64]:
    """Normalizes the input data."""
    xs = np.asarray(features, dtype=np.float64)
    std = np.std(xs, axis=0)
    std[std == 0] = 1e-8  # Avoid division by zero
    return np.asarray((xs - np.mean(xs, axis=0)) / std, dtype=np.float64)


def min_max_scale(features: ArrayLike) -> NDArray[np.float64]:
    """Scales the input data using min-max scaling."""
    xs = np.asarray(features, dtype=np.float64)
    min_val = np.min(xs, axis=0)
    max_val = np.max(xs, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-8  # Avoid division by zero
    scaled: NDArray[np.float64] = np.asarray(
        (xs - min_val) / range_val,
        dtype=np.float64,
    )
    return scaled


def polynomial_features(features: ArrayLike, degree: int = 2) -> NDArray[np.float64]:
    """Generates polynomial features up to the specified degree.

    For input features [x1, x2, ...], generates all polynomial combinations
    up to the specified degree. For example, with degree=2:
    [1, x1, x2, x1^2, x1*x2, x2^2]

    Args:
        features: Input features of shape (n_samples, n_features)
        degree: Maximum degree of polynomial features

    Returns:
        Polynomial features of shape (n_samples, n_output_features)
    """
    xs = np.asarray(features, dtype=np.float64)

    if degree < 0:
        raise ValueError("degree must be non-negative")

    n_samples, n_features = xs.shape

    # Generate all combinations of powers for each feature
    # that sum to at most 'degree'
    from itertools import combinations_with_replacement

    # Create list to store all polynomial terms
    poly_features = []

    # Always include the bias term (all features to power 0)
    poly_features.append(np.ones((n_samples, 1), dtype=np.float64))

    # Generate polynomial terms for each degree from 1 to degree
    for d in range(1, degree + 1):
        # Generate all combinations of feature indices with replacement
        # of length d (representing the degree)
        for combo in combinations_with_replacement(range(n_features), d):
            # Compute the product of features for this combination
            term = np.ones((n_samples, 1), dtype=np.float64)
            for feature_idx in combo:
                term *= xs[:, feature_idx : feature_idx + 1]
            poly_features.append(term)

    # Concatenate all polynomial terms
    result = np.hstack(poly_features)

    return result


def rotate2D(features: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Rotates 2D input features by the given angle."""
    xs = np.asarray(features, dtype=np.float64)
    if xs.shape[1] != 2:
        raise ValueError("Only supports 2D features.")

    # Rotation matrix
    A = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )

    # No translation for rotation
    b = np.zeros(2, dtype=np.float64)

    return affine_transform(xs, A, b)


def augment_to_3d(features: ArrayLike, constant: float = 1) -> NDArray[np.float64]:
    """
    Augments 2D features (x, y) to 3D features (constant, x, y).
    """
    xs = np.asarray(features, dtype=np.float64)
    A = np.array(
        [
            [0, 0],  # Add constant
            [1, 0],  # Keep x
            [0, 1],  # Keep y
        ],
    )
    b = np.array([constant, 0, 0])
    return affine_transform(xs, A, b)


def scale(features: ArrayLike, scale_factor: float) -> NDArray[np.float64]:
    """Scales the input features by the given factor."""
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive.")
    xs = np.asarray(features, dtype=np.float64)
    return xs * scale_factor


def shift(features: ArrayLike, offset: float) -> NDArray[np.float64]:
    """Shifts the input features by the given factor."""
    xs = np.asarray(features, dtype=np.float64)
    translation = np.asarray(offset, dtype=np.float64)
    A = np.eye(xs.shape[1], dtype=np.float64)
    return affine_transform(xs, A, translation)


def composite_transform(
    features: ArrayLike,
    transforms: Iterable[Callable[[ArrayLike], ArrayLike]],
) -> NDArray[np.float64]:
    """Applies a series of transformations to the input features."""
    xs = np.asarray(features, dtype=np.float64)
    if not transforms:
        raise ValueError("The transforms list cannot be empty.")
    for transform in transforms:
        if not callable(transform):
            raise ValueError("Each transform must be callable.")
        xs = np.asarray(transform(xs), dtype=np.float64)
    return xs
