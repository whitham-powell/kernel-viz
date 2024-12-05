# kernalized_perception.py

from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


def kernelized_perceptron(
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
    max_iterations: int = 100,
) -> Tuple[ArrayLike, ArrayLike]:

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    n_samples = x_arr.shape[0]
    alpha = np.zeros(n_samples)

    for iter in range(max_iterations):
        for i in range(n_samples):
            prediction = np.sum(
                [
                    alpha[i] * y_arr[j] * kernel(x_arr[j], x_arr[i])
                    for j in range(n_samples)
                ],
            )

        if y_arr[i] * prediction <= 0:
            alpha[i] += 1.0

    return alpha, x_arr


def predict(
    alpha: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
) -> Union[float, ArrayLike]:

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    alpha_arr = np.asarray(alpha)

    n_samples = x_arr.shape[0]
    predictions = np.zeros(n_samples)

    for i in range(n_samples):
        predictions[i] = np.sum(
            [
                alpha_arr[j] * y_arr[j] * kernel(x_arr[j], x_arr[i])
                for j in range(n_samples)
            ],
        )

    return np.sign(predictions)
