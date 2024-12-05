# %% kernels.py

import numpy as np
from numpy.typing import ArrayLike


def linear_kernel(x: ArrayLike, y: ArrayLike) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return float(np.dot(x_vec, y_vec))


def affine_kernel(x: ArrayLike, y: ArrayLike, c: int = 1) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return float(np.dot(x_vec, y_vec) + c)


def quadratic_kernel(x: ArrayLike, y: ArrayLike) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return float(np.dot(x_vec, y_vec) ** 2)


def polynomial_kernel(x: ArrayLike, y: ArrayLike, degree: int = 3, c: int = 1) -> float:
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return float((np.dot(x_vec, y_vec) + c) ** degree)


# %%
