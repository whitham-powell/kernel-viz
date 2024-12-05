# test_kernelized_perceptron.py
from dataclasses import dataclass

import numpy as np
import pytest

from src.kernelized_perceptron import kernelized_perceptron, predict
from src.kernels import (
    affine_kernel,
    exponential_kernel,
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    quadratic_kernel,
    rbf_gaussian_kernel,
)


@dataclass
class NPArrTestData:
    x_train: np.ndarray
    y_train: np.ndarray


@pytest.fixture
def np_arr_test_data():
    x_train = np.array([[1, 1], [2, 2], [3, 3]])
    y_train = np.array([1, -1, 1])
    return NPArrTestData(x_train, y_train)


class TestKernelizedPerceptron:
    # x_train = np.array([[1, 1], [2, 2], [3, 3]])
    # y_train = np.array([1, -1, 1])

    def generate_id(param):
        kernel_func_name = ""
        kwargs_str = ""
        print(f"Debug param: {param} (type: {type(param)})")
        if callable(param):
            kernel_func_name = param.__name__
            return f"{kernel_func_name}"
        if isinstance(param, dict):
            kwargs_str = ", ".join(f"{k}={v}" for k, v in param.items())
            return f"with_{kwargs_str or 'no_args'}"

    @pytest.mark.parametrize(
        "kernel_func, kwargs",
        [
            (linear_kernel, {}),
            (affine_kernel, {"c": 1.0}),
            (quadratic_kernel, {}),
            (polynomial_kernel, {"degree": 3, "c": 1.0}),
            (rbf_gaussian_kernel, {"sigma": 1.0}),
            (exponential_kernel, {"sigma": 1.0}),
            (laplacian_kernel, {"gamma": 1.0}),
        ],
        ids=generate_id,
    )
    def test_kernelized_perceptron(self, np_arr_test_data, kernel_func, kwargs):
        x_train = np_arr_test_data.x_train
        y_train = np_arr_test_data.y_train

        alpha, x_arr = kernelized_perceptron(x_train, y_train, kernel_func)
        assert alpha.shape == (3,)
        assert x_arr.shape == (3, 2)

    def test_alpha_updates(self, np_arr_test_data):

        x_train = np_arr_test_data.x_train
        y_train = np_arr_test_data.y_train
        alpha, _ = kernelized_perceptron(
            x_train,
            y_train,
            kernel=linear_kernel,
            max_iterations=10,
        )
        # Check that alpha is updated for misclassified samples
        assert np.any(
            alpha > 0,
        ), "Alpha coefficients should update for misclassified samples."
        # Check that the length of alpha matches the number of samples
        assert len(alpha) == len(
            x_train,
        ), "Alpha vector length must equal the number of training samples."

    def test_predict(self, np_arr_test_data):
        x_train = np_arr_test_data.x_train
        y_train = np_arr_test_data.y_train

        alpha, x_arr = kernelized_perceptron(x_train, y_train, linear_kernel)
        predictions = predict(alpha, x_arr, y_train, linear_kernel)
        print(f"Debug predictions: {predictions}")

    #     assert predictions.shape == (3,)
    #     assert np.all(predictions == y_train)
