# kernels/composite.py

from typing import Callable, Iterable, Union

import numpy as np
from numpy.typing import ArrayLike


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
