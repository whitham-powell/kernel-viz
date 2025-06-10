# kernels/base.py

from typing import Union

import numpy as np
from numpy.typing import ArrayLike


def linear_kernel(x: ArrayLike, y: ArrayLike) -> Union[float, ArrayLike]:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    result = np.dot(x_vec, y_vec)
    return float(result) if result.ndim == 0 else np.asarray(result)


def affine_kernel(
    x: ArrayLike,
    y: ArrayLike,
    c: float = 1.0,
) -> Union[float, ArrayLike]:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    result = np.dot(x_vec, y_vec) + c
    return float(result) if result.ndim == 0 else np.asarray(result)


def quadratic_kernel(
    x: ArrayLike,
    y: ArrayLike,
    c: float = 1.0,
) -> Union[float, ArrayLike]:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    result = (np.dot(x_vec, y_vec) + c) ** 2
    return float(result) if result.ndim == 0 else np.asarray(result)


def polynomial_kernel(
    x: ArrayLike,
    y: ArrayLike,
    degree: int = 3,
    c: float = 1.0,
) -> Union[float, ArrayLike]:
    if not isinstance(degree, int):
        raise ValueError("degree must be an integer.")
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    result = (np.dot(x_vec, y_vec) + c) ** degree
    return float(result) if result.ndim == 0 else np.asarray(result)


def rbf_gaussian_kernel(
    x: ArrayLike,
    y: ArrayLike,
    sigma: float = 1.0,
) -> Union[float, ArrayLike]:
    if sigma == 0:
        raise ValueError("sigma must be non-zero")
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    squared_distance = np.sum((x_vec - y_vec) ** 2)
    result = np.exp(-squared_distance / (2 * sigma**2))
    return float(result) if result.ndim == 0 else np.asarray(result)


def exponential_kernel(
    x: ArrayLike,
    y: ArrayLike,
    sigma: float = 1.0,
) -> Union[float, ArrayLike]:
    if sigma == 0:
        raise ValueError("sigma must be non-zero")
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    distance = np.linalg.norm(x_vec - y_vec)
    result = np.exp(-distance / sigma)
    return float(result) if result.ndim == 0 else np.asarray(result)


def laplacian_kernel(
    x: ArrayLike,
    y: ArrayLike,
    gamma: float,
) -> Union[float, ArrayLike]:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    manhattan_distance = np.sum(np.abs(x_vec - y_vec))
    result = np.exp(-gamma * manhattan_distance)
    return float(result) if result.ndim == 0 else np.asarray(result)
