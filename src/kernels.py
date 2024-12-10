# %% kernels.py

from typing import Callable, Iterable, Union

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


def additive_kernel(
    x: ArrayLike,
    y: ArrayLike,
    kernels: Iterable[Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]]],
) -> Union[float, ArrayLike]:
    if not kernels:
        raise ValueError("The kernels list cannot be empty.")
    result = np.sum([kernel(x, y) for kernel in kernels])
    return float(result) if result.ndim == 0 else np.asarray(result)


def multiplicative_kernel(
    x: ArrayLike,
    y: ArrayLike,
    kernels: Iterable[Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]]],
) -> Union[float, ArrayLike]:
    result: Union[float, ArrayLike] = 1.0
    for kernel in kernels:
        output = kernel(x, y)
        output = np.asarray(output)
        result = result * output  # Elementwise or scalar multiplication
    if isinstance(result, np.ndarray) and result.ndim == 0:
        return float(result)  # Convert scalar ndarray to float
    return result


# def weighted_kernel(
#     x: ArrayLike,
#     y: ArrayLike,
#     kernels: Tuple[Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]], ...],
#     weights: ArrayLike,
# ) -> Union[float, ArrayLike]:
#     if len(kernels) != len(weights):
#         raise ValueError("Number of kernels and weights must be equal.")
#     if not np.isclose(np.sum(weights), 1):
#         raise ValueError("Weights must sum to 1.")
#     return np.sum([weight * kernel(x, y) for kernel, weight in zip(kernels, weights)])


# def composite_kernel(
#     x: ArrayLike,
#     y: ArrayLike,
#     inner_kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
#     outer_kernel: Callable[[float, float], Union[float, ArrayLike]],
# ) -> Union[float, ArrayLike]:
#     inner_result = inner_kernel(x, y)
#     return outer_kernel(inner_result, inner_result)


### These are not RKHS kernels, excluding for now.
# def sigmoid_kernel(x: ArrayLike, y: ArrayLike, alpha: float = 0.01, c: float = 0) -> float:
#     x_vec = np.asarray(x)
#     y_vec = np.asarray(y)
#     dot_product = np.dot(x_vec, y_vec)
#     return float(np.tanh(alpha * dot_product + c))

# def cosine_similarity_kernel(x: ArrayLike, y: ArrayLike) -> float:
#     x_vec = np.asarray(x)
#     y_vec = np.asarray(y)
#     numerator = np.dot(x_vec, y_vec)
#     denominator = np.linalg.norm(x_vec) * np.linalg.norm(y_vec)
#     return float(numerator / denominator) if denominator != 0 else 0.0


# %%
