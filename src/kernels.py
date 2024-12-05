# %% kernels.py

import numpy as np
from numpy.typing import ArrayLike


def linear_kernel(x: ArrayLike, y: ArrayLike) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return float(np.dot(x_vec, y_vec))


def affine_kernel(x: ArrayLike, y: ArrayLike, c: float = 1.0) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return float(np.dot(x_vec, y_vec) + c)


def quadratic_kernel(x: ArrayLike, y: ArrayLike) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return float(np.dot(x_vec, y_vec) ** 2)


def polynomial_kernel(
    x: ArrayLike,
    y: ArrayLike,
    degree: int = 3,
    c: float = 1.0,
) -> float:
    if not isinstance(degree, int):
        raise ValueError("degree must be an integer.")
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return float((np.dot(x_vec, y_vec) + c) ** degree)


def gaussian_kernel(x: ArrayLike, y: ArrayLike, sigma: float = 1.0) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    squared_distance = np.sum((x_vec - y_vec) ** 2)
    return float(np.exp(-squared_distance / (2 * sigma**2)))


def exponential_kernel(x: ArrayLike, y: ArrayLike, sigma: float = 1.0) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    distance = np.linalg.norm(x_vec - y_vec)
    return float(np.exp(-distance / sigma))


def laplacian_kernel(x: ArrayLike, y: ArrayLike, sigma: float = 1.0) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    manhattan_distance = np.sum(np.abs(x_vec - y_vec))
    return float(np.exp(-manhattan_distance / sigma))


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
