# test_kernelized_perceptron.py

import numpy as np
import pytest

from src.kernelized_perceptron import kernelized_perceptron, predict
from src.kernels import (
    affine_kernel,
    exponential_kernel,
    kernel_func,
    laplacian_kernel,
    polynomial_kernel,
    quadratic_kernel,
    rbf_gaussian_kernel,
)


@pytest.fixture(params=["linear", "xor", "single_point", "empty"])
def test_dataset(request):
    """
    Provides different datasets for testing the kernelized perceptron.
    """
    if request.param == "linear":
        xs = np.array([[1, 2], [2, 3], [3, 1], [5, 2]], dtype=np.float64)
        ys = np.array([1, 1, -1, -1], dtype=np.float64)
    elif request.param == "xor":
        xs = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float64)
        ys = np.array([1, 1, -1, -1], dtype=np.float64)
    elif request.param == "single_point":
        xs = np.array([[1, 1]], dtype=np.float64)
        ys = np.array([1], dtype=np.float64)
    elif request.param == "empty":
        xs = np.empty((0, 2), dtype=np.float64)
        ys = np.empty(0, dtype=np.float64)
    else:
        raise ValueError(f"Unknown dataset type: {request.param}")

    return xs, ys, request.param


def generate_id_test_kernels(param):
    kernel_func_name = ""
    kwargs_str = ""
    print(f"Debug param: {param} (type: {type(param)})")
    if callable(param):
        kernel_func_name = param.__name__
        return f"{kernel_func_name}"
    if isinstance(param, dict):
        kwargs_str = ", ".join(f"{k}={v}" for k, v in param.items())
        return f"with_{kwargs_str or 'no_args'}"


test_kernels = [
    (kernel_func, {}),
    (affine_kernel, {"c": 1.0}),
    (quadratic_kernel, {}),
    (polynomial_kernel, {"degree": 3, "c": 1.0}),
    (rbf_gaussian_kernel, {"sigma": 1.0}),
    (exponential_kernel, {"sigma": 1.0}),
    (laplacian_kernel, {"gamma": 1.0}),
]


@pytest.mark.parametrize(
    "kernel_func, kwargs",
    test_kernels,
    ids=generate_id_test_kernels,
)
class TestKernelizedPerceptron:

    def test_linear_seperability(
        self,
        kernel_func,
        kwargs,
    ):
        xs = np.array([[1, 2], [2, 3], [3, 1], [5, 2]], dtype=np.float64)
        ys = np.array([1, 1, -1, -1], dtype=np.float64)

        alphas = kernelized_perceptron(xs, ys, kernel_func, kwargs)

        for x, y in zip(xs, ys):
            prediction = predict(xs, alphas, x, kernel_func, kwargs)
            assert (
                prediction == y
            ), f"Failed to classify {x} correctly. Expected {y}, got {prediction}."

    def test_XOR_handling(
        self,
        kernel_func,
        kwargs,
    ):
        # TODO: This should be split into 2 or more tests

        xs = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float64)
        ys = np.array([1, 1, -1, -1], dtype=np.float64)
        alphas = kernelized_perceptron(xs, ys, kernel_func, kwargs, max_iter=10)

        # Assert that some updates occurred
        assert np.any(alphas != 0), (
            f"Expected updates to alphas for XOR dataset, but no updates occurred. "
            f"Got alphas: {alphas}."
        )

        # Check if kernel is expected to separate XOR
        kernel_name = kernel_func.__name__
        non_linear_kernels = {
            "rbf_gaussian_kernel",
            "exponential_kernel",
            "laplacian_kernel",
        }

        # Check for misclassifications
        misclassified = sum(
            predict(xs, alphas, x, kernel_func, kwargs) != y for x, y in zip(xs, ys)
        )

        if kernel_name in non_linear_kernels:
            # Allow non-linear kernels to succeed
            assert misclassified == 0, (
                f"Kernel {kernel_name} should classify XOR correctly but did not. "
                f"Got alphas: {alphas}."
            )
        else:
            # Expect linear kernels to fail
            assert misclassified > 0, (
                f"Kernel {kernel_name} incorrectly classified XOR data fully. "
                f"Got alphas: {alphas}."
            )

    def test_empty_dataset(
        self,
        kernel_func,
        kwargs,
    ):
        xs = np.empty((0, 2))
        ys = np.empty(0)
        alphas = kernelized_perceptron(xs, ys, kernel_func)
        assert (
            alphas.size == 0
        ), "Expected no alphas to be returned for an empty dataset."

    def test_single_sample(
        self,
        kernel_func,
        kwargs,
    ):
        xs = np.array([[1, 1]], dtype=np.float64)
        ys = np.array([1], dtype=np.float64)
        alphas = kernelized_perceptron(xs, ys, kernel_func)
        assert (
            alphas.size == 1
        ), "Expected a single alpha to be returned for a single sample."

    def test_alphas_update(
        self,
        kernel_func,
        kwargs,
    ):
        xs = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float64)
        ys = np.array([1, 1, -1], dtype=np.float64)
        alphas = kernelized_perceptron(xs, ys, kernel_func, kwargs)
        assert np.any(
            alphas > 0,
        ), "Alpha coefficients should update for misclassified samples."

    def test_alphas_length(
        self,
        kernel_func,
        kwargs,
    ):
        xs = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float64)
        ys = np.array([1, 1, -1], dtype=np.float64)
        alphas = kernelized_perceptron(xs, ys, kernel_func, kwargs)
        assert len(alphas) == len(
            xs,
        ), "Alpha vector length must equal the number of training samples."
