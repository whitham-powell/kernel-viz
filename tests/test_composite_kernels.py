# test_composite_kernels.py

import numpy as np
import pytest

from src.kernels import (
    additive_kernel,
    linear_kernel,
    multiplicative_kernel,
    polynomial_kernel,
    rbf_gaussian_kernel,
)


class TestAdditiveKernel:
    """Tests for the additive kernel combination function."""

    def test_additive_kernel_basic(self):
        """Test basic additive kernel functionality."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]
        result = additive_kernel(x, y, kernels)

        # Manually compute expected result
        expected = linear_kernel(x, y) + rbf_gaussian_kernel(x, y, sigma=1.0)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_additive_kernel_single_kernel(self):
        """Test additive kernel with single kernel."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        kernels = [linear_kernel]
        result = additive_kernel(x, y, kernels)
        expected = linear_kernel(x, y)

        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_additive_kernel_multiple_kernels(self):
        """Test additive kernel with multiple different kernels."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        kernels = [
            linear_kernel,
            lambda a, b: polynomial_kernel(a, b, degree=2, c=1.0),
            lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0),
        ]
        result = additive_kernel(x, y, kernels)

        # Manually compute expected result
        expected = (
            linear_kernel(x, y)
            + polynomial_kernel(x, y, degree=2, c=1.0)
            + rbf_gaussian_kernel(x, y, sigma=1.0)
        )
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_additive_kernel_empty_list_raises_error(self):
        """Test that empty kernel list raises ValueError."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        with pytest.raises(ValueError, match="The kernels list cannot be empty"):
            additive_kernel(x, y, [])

    def test_additive_kernel_symmetry(self):
        """Test that additive kernel maintains symmetry."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]

        result_xy = additive_kernel(x, y, kernels)
        result_yx = additive_kernel(y, x, kernels)

        assert np.isclose(
            result_xy,
            result_yx,
        ), f"Symmetry violated: {result_xy} != {result_yx}"

    def test_additive_kernel_identity_case(self):
        """Test additive kernel where x == y."""
        x = np.array([1, 2])

        kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]
        result = additive_kernel(x, x, kernels)

        # When x == y, RBF kernel should return 1.0
        expected = linear_kernel(x, x) + 1.0
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_additive_kernel_with_numpy_arrays(self):
        """Test additive kernel with numpy array inputs."""
        x = np.array([1.0, 2.0], dtype=np.float64)
        y = np.array([3.0, 4.0], dtype=np.float64)

        kernels = [linear_kernel]
        result = additive_kernel(x, y, kernels)
        expected = linear_kernel(x, y)

        assert np.isclose(result, expected), f"Expected {expected}, got {result}"
        assert isinstance(
            result,
            (float, np.floating),
        ), f"Result should be float, got {type(result)}"


class TestMultiplicativeKernel:
    """Tests for the multiplicative kernel combination function."""

    def test_multiplicative_kernel_basic(self):
        """Test basic multiplicative kernel functionality."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]
        result = multiplicative_kernel(x, y, kernels)

        # Manually compute expected result
        expected = linear_kernel(x, y) * rbf_gaussian_kernel(x, y, sigma=1.0)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_multiplicative_kernel_single_kernel(self):
        """Test multiplicative kernel with single kernel."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        kernels = [linear_kernel]
        result = multiplicative_kernel(x, y, kernels)
        expected = linear_kernel(x, y)

        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_multiplicative_kernel_multiple_kernels(self):
        """Test multiplicative kernel with multiple different kernels."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        kernels = [
            linear_kernel,
            lambda a, b: polynomial_kernel(a, b, degree=2, c=1.0),
            lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0),
        ]
        result = multiplicative_kernel(x, y, kernels)

        # Manually compute expected result
        expected = (
            linear_kernel(x, y)
            * polynomial_kernel(x, y, degree=2, c=1.0)
            * rbf_gaussian_kernel(x, y, sigma=1.0)
        )
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_multiplicative_kernel_empty_list(self):
        """Test multiplicative kernel with empty list returns 1.0."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        result = multiplicative_kernel(x, y, [])
        expected = 1.0

        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_multiplicative_kernel_symmetry(self):
        """Test that multiplicative kernel maintains symmetry."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]

        result_xy = multiplicative_kernel(x, y, kernels)
        result_yx = multiplicative_kernel(y, x, kernels)

        assert np.isclose(
            result_xy,
            result_yx,
        ), f"Symmetry violated: {result_xy} != {result_yx}"

    def test_multiplicative_kernel_identity_case(self):
        """Test multiplicative kernel where x == y."""
        x = np.array([1, 2])

        kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]
        result = multiplicative_kernel(x, x, kernels)

        # When x == y, RBF kernel should return 1.0
        expected = linear_kernel(x, x) * 1.0
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_multiplicative_kernel_zero_result(self):
        """Test multiplicative kernel that should result in zero."""
        x = np.array([0, 0])
        y = np.array([1, 1])

        # Linear kernel with zero vector should be 0
        kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]
        result = multiplicative_kernel(x, y, kernels)

        # Since linear_kernel([0,0], [1,1]) = 0, result should be 0
        assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"

    def test_multiplicative_kernel_with_numpy_arrays(self):
        """Test multiplicative kernel with numpy array inputs."""
        x = np.array([1.0, 2.0], dtype=np.float64)
        y = np.array([3.0, 4.0], dtype=np.float64)

        kernels = [linear_kernel]
        result = multiplicative_kernel(x, y, kernels)
        expected = linear_kernel(x, y)

        assert np.isclose(result, expected), f"Expected {expected}, got {result}"
        assert isinstance(
            result,
            (float, np.floating),
        ), f"Result should be float, got {type(result)}"

    def test_multiplicative_kernel_very_small_values(self):
        """Test multiplicative kernel with very small kernel values."""
        x = np.array([0.001, 0.001])
        y = np.array([100, 100])

        # RBF with small sigma should give very small values
        kernels = [
            lambda a, b: rbf_gaussian_kernel(a, b, sigma=0.01),
            lambda a, b: rbf_gaussian_kernel(a, b, sigma=0.01),
        ]
        result = multiplicative_kernel(x, y, kernels)

        # Result should be very small but positive
        assert result >= 0, f"Result should be non-negative, got {result}"
        assert result < 1e-6, f"Result should be very small, got {result}"

    def test_multiplicative_kernel_large_values(self):
        """Test multiplicative kernel with large kernel values."""
        x = np.array([1, 1])
        y = np.array([1, 1])

        # Polynomial kernels with same point should give large values
        kernels = [
            lambda a, b: polynomial_kernel(a, b, degree=3, c=1.0),
            lambda a, b: polynomial_kernel(a, b, degree=2, c=1.0),
        ]
        result = multiplicative_kernel(x, y, kernels)

        # Manually compute: (2+1)^3 * (2+1)^2 = 27 * 9 = 243
        expected = 27 * 9
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"


class TestCompositeKernelIntegration:
    """Integration tests for composite kernels with other components."""

    def test_additive_kernel_in_prediction_context(self):
        """Test additive kernel can be used in prediction scenarios."""
        from src.kernelized_perceptron import predict

        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        alphas = np.array([1.0, 1.0], dtype=np.float64)
        x_new = np.array([2, 2])

        # Create a composite kernel function
        def composite_kernel(x, y):
            kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]
            return additive_kernel(x, y, kernels)

        prediction = predict(xs, alphas, x_new, composite_kernel)
        assert prediction in [-1, 1], f"Invalid prediction: {prediction}"

    def test_multiplicative_kernel_in_prediction_context(self):
        """Test multiplicative kernel can be used in prediction scenarios."""
        from src.kernelized_perceptron import predict

        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        alphas = np.array([1.0, 1.0], dtype=np.float64)
        x_new = np.array([2, 2])

        # Create a composite kernel function
        def composite_kernel(x, y):
            kernels = [lambda a, b: polynomial_kernel(a, b, degree=2, c=1.0)]
            return multiplicative_kernel(x, y, kernels)

        prediction = predict(xs, alphas, x_new, composite_kernel)
        assert prediction in [-1, 1], f"Invalid prediction: {prediction}"

    @pytest.mark.parametrize(
        "kernel_combination",
        [
            "additive",
            "multiplicative",
        ],
    )
    def test_composite_kernels_positive_definiteness_property(self, kernel_combination):
        """Test that composite kernels maintain positive definiteness properties."""
        x = np.array([1, 2])

        kernels = [linear_kernel, lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0)]

        if kernel_combination == "additive":
            result = additive_kernel(x, x, kernels)
        else:  # multiplicative
            result = multiplicative_kernel(x, x, kernels)

        # For positive definite kernels, K(x,x) should be positive
        assert result > 0, f"K(x,x) should be positive, got {result}"

    def test_nested_composite_kernels(self):
        """Test composition of composite kernels."""
        x = np.array([1, 2])
        y = np.array([3, 4])

        # Create a kernel that uses additive inside multiplicative
        def nested_kernel(a, b):
            inner_kernels = [
                linear_kernel,
                lambda u, v: rbf_gaussian_kernel(u, v, sigma=1.0),
            ]
            additive_result = additive_kernel(a, b, inner_kernels)
            return additive_result

        outer_kernels = [nested_kernel, linear_kernel]
        result = multiplicative_kernel(x, y, outer_kernels)

        # Should be a valid finite number
        assert np.isfinite(result), f"Result should be finite, got {result}"
        assert result >= 0, f"Result should be non-negative, got {result}"
