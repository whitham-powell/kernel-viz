# kernalized_perception.py

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


def kernelized_perceptron(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
    kernel_params: Optional[Dict[str, Any]] = None,
    max_iter: int = 100,
) -> NDArray[np.float64]:
    n_samples = len(xs)
    alphas = np.zeros(n_samples)
    alphas[0] = ys[0]

    for _ in range(max_iter):
        for i in range(1, n_samples):
            xi = xs[i]
            yi = ys[i]
            f0_xi = np.sum(
                [
                    alphas[j] * kernel(xs[j], xi, **kernel_params)
                    for j in range(n_samples)
                ],
            )
            if yi * f0_xi < 0:
                alphas[i] = yi
    return alphas


def predict(
    xs: NDArray[np.float64],
    alphas: NDArray[np.float64],
    x_new: ArrayLike,
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
    kernel_params: Optional[Dict[str, Any]] = None,
) -> int:
    f_x_new = np.sum(
        [alphas[i] * kernel(xs[i], x_new, **kernel_params) for i in range(len(xs))],
    )
    return 1 if f_x_new > 0 else -1
