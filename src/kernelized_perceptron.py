# kernelized_perceptron.py

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
            "kernel_matrix": None,
        }

    def log_misclassification_count(self, count: int) -> None:
        """Log the misclassification count at a given epoch step. Should be logged at each epoch step."""
        if not isinstance(count, (int, np.integer)):
            raise TypeError(
                f"misclassification count must be an integer, got {type(count)}",
            )
        self.logs["misclassification_count"].append(count)

    def log_alphas(self, epoch_step: int, alphas: NDArray[np.float64]) -> None:
        """Log the alphas at a given epoch step. Should log the alphas at each epoch."""
        if not isinstance(epoch_step, int):
            raise TypeError(f"epoch_step must be an integer, got {type(epoch_step)}")
        self.logs["alphas"].append({"iteration": epoch_step, "alphas": alphas.copy()})

    def log_kernel(
        self,
        kernel_func: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the kernel function and its parameters. Kernel function must be a callable."""
        if not callable(kernel_func):
            raise TypeError(f"kernel_func must be callable, got {type(kernel_func)}")

        self.logs["kernel"] = kernel_func
        self.logs["kernel_params"] = kernel_params or {}

    def log_feature_space(self, feature_space: NDArray[np.float64]) -> None:
        """Log the feature space. Should only be logged once per training."""
        self.logs["feature_space"] = feature_space.copy()

    def log_kernel_matrix(self, xs: NDArray[np.float64]) -> None:
        kernel_matrix = self.compute_kernel_matrix(xs)
        self.logs["kernel_matrix"] = kernel_matrix

    def compute_kernel_matrix(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the kernel matrix for the given feature space."""
        n_samples = len(xs)
        kernel_matrix = np.zeros((n_samples, n_samples))

        kernel_func = self.logs.get("kernel")
        kernel_params = self.logs.get("kernel_params")

        if kernel_func is None:
            raise ValueError("Kernel function is not set.")

        if not callable(kernel_func):
            raise TypeError(
                f"kernel_func: ({kernel_func}) must be callable, got {type(kernel_func)}",
            )

        for i in range(n_samples):
            for j in range(i, n_samples):
                try:
                    value = kernel_func(xs[i], xs[j], **(kernel_params or {}))
                    kernel_matrix[i, j] = value
                    kernel_matrix[j, i] = value
                except Exception as e:
                    raise RuntimeError(
                        f"Error computing kernel matrix for indices (i={i}, j={j}): {e}",
                    ) from e

        assert np.allclose(
            kernel_matrix,
            kernel_matrix.T,
        ), "Kernel matrix is not symmetric"
        return kernel_matrix

    def get_logs(self) -> Dict[str, Any]:
        """Return a copy of the logs dictionary."""
        return self.logs.copy()


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

    for epoch_step in range(max_iter):
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
            logger.log_alphas(epoch_step, alphas)
            print(f"Epoch {epoch_step + 1}: Misclassified {misclassified}")
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
