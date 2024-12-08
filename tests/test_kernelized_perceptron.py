# test_kernelized_perceptron.py

import numpy as np
import pytest

from src.kernelized_perceptron import PerceptronLogger, kernelized_perceptron, predict
from src.kernels import (
    affine_kernel,
    exponential_kernel,
    laplacian_kernel,
    linear_kernel,
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
    (linear_kernel, {}),
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
                f"Kernel {kernel_name} incorrectly classified XOR data fully as expected. "
                f"Got alphas: {alphas}."
            )

    def test_empty_dataset(
        self,
        kernel_func,
        kwargs,
    ):
        xs = np.empty((0, 2))
        ys = np.empty(0)
        alphas = kernelized_perceptron(xs, ys, kernel_func, kwargs)
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
        alphas = kernelized_perceptron(xs, ys, kernel_func, kwargs)
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
        len_alphas = len(alphas)
        len_xs = len(xs)
        assert (
            len_alphas == len_xs
        ), f"Alpha vector length must equal the number of training samples. Got {len_alphas} alphas for {len_xs} samples."


class TestPerceptronLogger:

    def test_log_misclassification_count(self):
        logger = PerceptronLogger()
        logger.log_misclassification_count(5)
        assert logger.get_logs()["misclassification_count"] == [
            5,
        ], "Expected a single log entry for misclassification count."

    def test_log_misclassification_count_raises_on_non_int(self):
        logger = PerceptronLogger()
        with pytest.raises(TypeError):
            logger.log_misclassification_count(5.0)

    def test_log_alphas(self):
        logger = PerceptronLogger()
        alphas = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        logger.log_alphas(1, alphas)

        logs = logger.get_logs()
        assert len(logs["alphas"]) == 1
        logged_entry = logs["alphas"][0]

        # Check the structure and values
        assert isinstance(logged_entry, dict)
        assert "iteration" in logged_entry
        assert "alphas" in logged_entry
        assert logged_entry["iteration"] == 1, "Iteration mismatch"
        assert np.array_equal(logged_entry["alphas"], alphas), "Alphas mismatch"

    def test_log_alphas_raises_on_non_int(self):
        logger = PerceptronLogger()
        alphas = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        with pytest.raises(TypeError):
            logger.log_alphas(1.0, alphas)

    @pytest.mark.parametrize(
        "kernel_func, kernel_kwargs",
        test_kernels,
        ids=generate_id_test_kernels,
    )
    def test_log_kernel(self, kernel_func, kernel_kwargs):
        logger = PerceptronLogger()
        kernel = kernel_func
        kernel_params = kernel_kwargs
        logger.log_kernel(kernel, kernel_params)
        assert logger.get_logs()["kernel"] == kernel
        assert logger.get_logs()["kernel_params"] == kernel_params

    def test_log_kernel_raises_on_non_callable_kernel_func(self):
        logger = PerceptronLogger()
        with pytest.raises(TypeError):
            logger.log_kernel(1, {})

    def test_log_feature_space(self):
        logger = PerceptronLogger()
        feature_space = np.array([[1, 2], [3, 4]], dtype=np.float64)
        logger.log_feature_space(feature_space)
        assert np.array_equal(logger.get_logs()["feature_space"], feature_space)

    @pytest.mark.parametrize(
        "kernel_func, kernel_kwargs",
        test_kernels,
        ids=generate_id_test_kernels,
    )
    def test_log_kernel_matrix(self, kernel_func, kernel_kwargs):
        logger = PerceptronLogger()
        test_xs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)

        logger.log_kernel(kernel_func, kernel_kwargs)
        logger.log_kernel_matrix(test_xs)
        logged_kernel_matrix = logger.get_logs()["kernel_matrix"]
        logged_kernel = logger.get_logs()["kernel"]
        logged_kernel_params = logger.get_logs()["kernel_params"]

        expected_kernel_matrix = np.array(
            [
                [
                    kernel_func(test_xs[0], test_xs[0], **kernel_kwargs),
                    kernel_func(test_xs[0], test_xs[1], **kernel_kwargs),
                    kernel_func(test_xs[0], test_xs[2], **kernel_kwargs),
                ],
                [
                    kernel_func(test_xs[1], test_xs[0], **kernel_kwargs),
                    kernel_func(test_xs[1], test_xs[1], **kernel_kwargs),
                    kernel_func(test_xs[1], test_xs[2], **kernel_kwargs),
                ],
                [
                    kernel_func(test_xs[2], test_xs[0], **kernel_kwargs),
                    kernel_func(test_xs[2], test_xs[1], **kernel_kwargs),
                    kernel_func(test_xs[2], test_xs[2], **kernel_kwargs),
                ],
            ],
        )
        assert logged_kernel is kernel_func, "Logged kernel function does not match"
        assert (
            logged_kernel_params == kernel_kwargs
        ), "Logged kernel parameters do not match"
        assert np.allclose(
            logged_kernel_matrix,
            expected_kernel_matrix,
        ), "Kernel matrix logging failed"

    @pytest.mark.parametrize(
        "kernel_func, kernel_kwargs",
        test_kernels,
        ids=generate_id_test_kernels,
    )
    def test_compute_kernel_matrix(self, kernel_func, kernel_kwargs):
        test_xs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        logger = PerceptronLogger()
        logger.log_kernel(kernel_func, kernel_kwargs)

        computed_kernel_matrix = logger.compute_kernel_matrix(test_xs)
        expected_kernel_matrix = np.array(
            [
                [
                    kernel_func(test_xs[0], test_xs[0], **kernel_kwargs),
                    kernel_func(test_xs[0], test_xs[1], **kernel_kwargs),
                    kernel_func(test_xs[0], test_xs[2], **kernel_kwargs),
                ],
                [
                    kernel_func(test_xs[1], test_xs[0], **kernel_kwargs),
                    kernel_func(test_xs[1], test_xs[1], **kernel_kwargs),
                    kernel_func(test_xs[1], test_xs[2], **kernel_kwargs),
                ],
                [
                    kernel_func(test_xs[2], test_xs[0], **kernel_kwargs),
                    kernel_func(test_xs[2], test_xs[1], **kernel_kwargs),
                    kernel_func(test_xs[2], test_xs[2], **kernel_kwargs),
                ],
            ],
        )
        # Assert correctness
        assert np.allclose(
            computed_kernel_matrix,
            expected_kernel_matrix,
        ), "Computed kernel matrix does not match expected kernel matrix"

        # Assert symmetry
        assert np.allclose(
            computed_kernel_matrix,
            computed_kernel_matrix.T,
        ), "Computed kernel matrix is not symmetric"

    def test_compute_kernel_matrix_no_kernel_set(self):
        logger = PerceptronLogger()
        test_xs = np.array([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="Kernel function is not set."):
            logger.compute_kernel_matrix(test_xs)

    def test_compute_kernel_matrix_non_callable_kernel(self):
        logger = PerceptronLogger()

        logger.logs["kernel"] = "not a function"
        test_xs = np.array([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(TypeError, match="kernel_func: .* must be callable"):
            logger.compute_kernel_matrix(test_xs)

    def test_get_logs(self):
        logger = PerceptronLogger()
        logs = logger.get_logs()
        assert isinstance(logs, dict)
        assert "misclassification_count" in logs
        assert "alphas" in logs
        assert "kernel" in logs
        assert "kernel_params" in logs
        assert "feature_space" in logs


@pytest.mark.parametrize(
    "kernel_func, kernel_kwargs",
    test_kernels,
    ids=generate_id_test_kernels,
)
class TestKernelizedPerceptronIntegration:
    def test_kernelized_perceptron_logging(
        self,
        kernel_func,
        kernel_kwargs,
    ):
        # Setup test data
        xs = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.float64)
        ys = np.array([1, -1, -1, 1], dtype=np.float64)
        test_max_iterations = 2
        logger = PerceptronLogger()
        _ = kernelized_perceptron(
            xs,
            ys,
            kernel_func,
            kernel_kwargs,
            max_iter=test_max_iterations,
            logger=logger,
        )

        logs = logger.get_logs()

        assert len(logs["misclassification_count"]) > 0
        expected_alphas_log_len = len(logs["alphas"])
        assert (
            expected_alphas_log_len == test_max_iterations
        ), f"Expected {test_max_iterations} alphas, got {expected_alphas_log_len}"
        assert (
            logs["kernel"] is not None
        ), f"Kernel function not logged for {kernel_func}"
        assert (
            logs["feature_space"] is not None
        ), f"Feature space not logged for {kernel_func}"
        assert logs["alphas"][0]["alphas"].shape[0] == len(xs), "Alphas shape mismatch"
        assert logs["feature_space"].shape == xs.shape, "Feature space shape mismatch"

    def test_kernelized_perceptron_logging_with_different_kernels(
        self,
        kernel_func,
        kernel_kwargs,
    ):
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.float64)
        logger = PerceptronLogger()
        test_max_iterations = 2
        _ = kernelized_perceptron(
            xs,
            ys,
            kernel_func,
            kernel_kwargs,
            max_iter=test_max_iterations,
            logger=logger,
        )

        logs = logger.get_logs()

        assert logs["kernel"] == kernel_func, "Kernel function mismatch"
        assert logs["kernel_params"] == kernel_kwargs, "Kernel parameters mismatch"

    def test_kernelized_perceptron_logging_iterations(self, kernel_func, kernel_kwargs):
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.float64)
        logger = PerceptronLogger()
        test_max_iterations = 4
        _ = kernelized_perceptron(
            xs,
            ys,
            kernel_func,
            kernel_kwargs,
            max_iter=test_max_iterations,
            logger=logger,
        )

        logs = logger.get_logs()

        assert len(logs["misclassification_count"]) <= test_max_iterations
        assert len(logs["alphas"]) <= test_max_iterations

        iterations = [alpha_dict["iteration"] for alpha_dict in logs["alphas"]]
        assert iterations == list(range(len(iterations)))
