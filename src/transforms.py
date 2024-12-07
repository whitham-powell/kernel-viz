# transforms.py

from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import PolynomialFeatures


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
    """Generates polynomial features up to the specified degree."""
    xs = np.asarray(features, dtype=np.float64)
    poly_features = PolynomialFeatures(degree=degree)
    transformed: NDArray[np.float64] = np.asarray(
        poly_features.fit_transform(xs),
        dtype=np.float64,
    )
    return transformed


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


def scale(features: ArrayLike, scale_factor: float) -> NDArray[np.float64]:
    """Scales the input features by the given factor."""
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive.")
    xs = np.asarray(features, dtype=np.float64)
    return xs * scale_factor


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
