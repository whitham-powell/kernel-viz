# kernalized_perceptron.py

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class PerceptronLogger:
    def __init__(self) -> None:
        self.logs: Dict[str, Any] = {
            "misclassification_count": [],
            "alphas": [],
            "kernel": None,
            "kernel_params": None,
            "feature_space": None,
        }

    def log_misclassification_count(self, count: int) -> None:
        self.logs["misclassification_count"].append(count)

    def log_alphas(self, epoch_step: int, alphas: NDArray[np.float64]) -> None:
        self.logs["alphas"].append({"iteration": epoch_step, "alphas": alphas.copy()})

    def log_kernel(
        self,
        kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logs["kernel"] = kernel
        self.logs["kernel_params"] = kernel_params

    def log_feature_space(self, feature_space: NDArray[np.float64]) -> None:
        self.logs["feature_space"] = feature_space.copy()

    def get_logs(self) -> Dict[str, Any]:
        return self.logs


def kernelized_perceptron(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
    kernel_params: Optional[Dict[str, Any]] = None,
    max_iter: int = 100,
    logger: Optional[PerceptronLogger] = None,
) -> NDArray[np.float64]:

    n_samples = len(xs)
    if n_samples == 0:
        return np.array([])

    alphas = np.zeros(n_samples)
    alphas[0] = ys[0]

    if logger:
        logger.log_kernel(kernel, kernel_params)
        logger.log_feature_space(xs)

    for epoc_step in range(max_iter):
        misclassified = 0
        for i in range(1, n_samples):
            xi = xs[i]
            yi = ys[i]
            f0_xi = np.sum(
                [
                    alphas[j] * kernel(xs[j], xi, **(kernel_params or {}))
                    for j in range(n_samples)
                ],
            )
            if yi * f0_xi < 0:
                alphas[i] += yi
                misclassified += 1

        if logger:
            logger.log_alphas(epoc_step, alphas)
            print(f"Epoch {epoc_step}: Misclassified {misclassified}")
            logger.log_misclassification_count(misclassified)
    return alphas


def predict(
    xs: NDArray[np.float64],
    alphas: NDArray[np.float64],
    x_new: ArrayLike,
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
    kernel_params: Optional[Dict[str, Any]] = None,
) -> int:
    f_x_new = np.sum(
        [
            alphas[i] * kernel(xs[i], x_new, **(kernel_params or {}))
            for i in range(len(xs))
        ],
    )
    return 1 if f_x_new > 0 else -1
