"""Comprehensive tests for kernel functions covering additional edge cases and properties."""

import numpy as np
import pytest

from kernel_viz.kernels import (
    affine_kernel,
    exponential_kernel,
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    quadratic_kernel,
    rbf_gaussian_kernel,
)


class TestKernelProperties:
    """Test mathematical properties that kernels should satisfy."""

    @pytest.fixture
    def sample_vectors(self):
        """Generate sample test vectors."""
        return {
            "x1": np.array([1.0, 2.0, 3.0]),
            "x2": np.array([4.0, 5.0, 6.0]),
            "x3": np.array([7.0, 8.0, 9.0]),
            "zero": np.array([0.0, 0.0, 0.0]),
            "unit": np.array([1.0, 0.0, 0.0]),
        }

    @pytest.mark.parametrize(
        "kernel_func,kernel_params",
        [
            (linear_kernel, {}),
            (affine_kernel, {"c": 1.0}),
            (quadratic_kernel, {"c": 1.0}),
            (polynomial_kernel, {"degree": 3, "c": 1.0}),
            (rbf_gaussian_kernel, {"sigma": 1.0}),
            (exponential_kernel, {"sigma": 1.0}),
            (laplacian_kernel, {"gamma": 1.0}),
        ],
    )
    def test_kernel_symmetry(self, kernel_func, kernel_params, sample_vectors):
        """Test that k(x, y) = k(y, x) for all kernels."""
        x1 = sample_vectors["x1"]
        x2 = sample_vectors["x2"]

        k_xy = kernel_func(x1, x2, **kernel_params)
        k_yx = kernel_func(x2, x1, **kernel_params)

        assert np.allclose(k_xy, k_yx), (
            f"{kernel_func.__name__} is not symmetric: "
            f"k(x,y)={k_xy} != k(y,x)={k_yx}"
        )

    @pytest.mark.parametrize(
        "kernel_func,kernel_params",
        [
            (linear_kernel, {}),
            (affine_kernel, {"c": 1.0}),
            (quadratic_kernel, {"c": 1.0}),
            (polynomial_kernel, {"degree": 3, "c": 1.0}),
            (rbf_gaussian_kernel, {"sigma": 1.0}),
            (exponential_kernel, {"sigma": 1.0}),
            (laplacian_kernel, {"gamma": 1.0}),
        ],
    )
    def test_kernel_positive_semidefinite(self, kernel_func, kernel_params):
        """Test that kernel matrix is positive semi-definite."""
        # Generate random data
        np.random.seed(42)
        X = np.random.randn(10, 3)

        # Compute kernel matrix
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel_func(X[i], X[j], **kernel_params)

        # Check positive semi-definiteness via eigenvalues
        eigenvalues = np.linalg.eigvals(K)
        min_eigenvalue = np.min(eigenvalues.real)

        # Allow small negative values due to numerical precision
        assert min_eigenvalue >= -1e-10, (
            f"{kernel_func.__name__} produces non-PSD kernel matrix. "
            f"Min eigenvalue: {min_eigenvalue}"
        )

    def test_linear_kernel_with_zero_vector(self, sample_vectors):
        """Test linear kernel behavior with zero vector."""
        x = sample_vectors["x1"]
        zero = sample_vectors["zero"]

        assert linear_kernel(x, zero) == 0.0
        assert linear_kernel(zero, x) == 0.0
        assert linear_kernel(zero, zero) == 0.0

    def test_rbf_kernel_with_identical_vectors(self, sample_vectors):
        """Test RBF kernel returns 1 for identical vectors."""
        x = sample_vectors["x1"]

        for sigma in [0.1, 1.0, 10.0]:
            assert rbf_gaussian_kernel(x, x, sigma=sigma) == 1.0

    def test_rbf_kernel_range(self, sample_vectors):
        """Test RBF kernel output is in [0, 1]."""
        x1 = sample_vectors["x1"]
        x2 = sample_vectors["x2"]

        for sigma in [0.1, 1.0, 10.0]:
            k_val = rbf_gaussian_kernel(x1, x2, sigma=sigma)
            assert 0 <= k_val <= 1, f"RBF kernel value {k_val} not in [0, 1]"


class TestKernelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_kernels_with_1d_vectors(self):
        """Test all kernels work with 1D vectors."""
        x = np.array([3.0])
        y = np.array([4.0])

        # Test each kernel
        assert linear_kernel(x, y) == 12.0
        assert affine_kernel(x, y, c=1.0) == 13.0
        assert quadratic_kernel(x, y, c=1.0) == 169.0  # (12 + 1)^2
        assert polynomial_kernel(x, y, degree=3, c=1.0) == 2197.0  # (12 + 1)^3

        # RBF and exponential should be in (0, 1]
        assert 0 < rbf_gaussian_kernel(x, y, sigma=1.0) < 1
        assert 0 < exponential_kernel(x, y, sigma=1.0) < 1
        assert laplacian_kernel(x, y, gamma=1.0) > 0

    def test_kernels_with_high_dimensional_vectors(self):
        """Test kernels with high-dimensional vectors."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        # All kernels should work without error
        linear_result = linear_kernel(x, y)
        affine_result = affine_kernel(x, y)
        quad_result = quadratic_kernel(x, y)
        poly_result = polynomial_kernel(x, y)
        rbf_result = rbf_gaussian_kernel(x, y)
        exp_result = exponential_kernel(x, y)
        lap_result = laplacian_kernel(x, y, gamma=1.0)

        # Basic sanity checks
        assert isinstance(linear_result, float)
        assert isinstance(affine_result, float)
        assert quad_result > 0  # quadratic of affine is always positive
        assert poly_result != 0  # very unlikely to be exactly 0
        assert 0 < rbf_result < 1
        assert 0 < exp_result < 1
        assert lap_result > 0

    def test_polynomial_kernel_edge_cases(self):
        """Test polynomial kernel with edge case parameters."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        # Degree 0 should return 1
        assert polynomial_kernel(x, y, degree=0, c=0) == 1.0

        # Large degree should not overflow for reasonable inputs
        result = polynomial_kernel(x, y, degree=10, c=0.1)
        assert np.isfinite(result)

        # Negative c is allowed
        result = polynomial_kernel(x, y, degree=2, c=-5.0)
        assert np.isfinite(result)

    def test_rbf_kernel_extreme_sigma(self):
        """Test RBF kernel with extreme sigma values."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        # Very small sigma - should approach 0 for different vectors
        result_small = rbf_gaussian_kernel(x, y, sigma=0.01)
        assert result_small < 1e-10

        # Very large sigma - should approach 1
        result_large = rbf_gaussian_kernel(x, y, sigma=1000.0)
        assert result_large > 0.99

    def test_laplacian_kernel_extreme_gamma(self):
        """Test Laplacian kernel with extreme gamma values."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        # Very small gamma - should approach 1
        result_small = laplacian_kernel(x, y, gamma=0.001)
        assert result_small > 0.99

        # Very large gamma - should approach 0
        result_large = laplacian_kernel(x, y, gamma=100.0)
        assert result_large < 0.01


class TestKernelNumericalStability:
    """Test numerical stability of kernel computations."""

    def test_rbf_kernel_with_very_close_vectors(self):
        """Test RBF kernel with nearly identical vectors."""
        x = np.array([1.0, 2.0, 3.0])
        y = x + 1e-15  # Add tiny perturbation

        # Should still return approximately 1
        result = rbf_gaussian_kernel(x, y, sigma=1.0)
        assert np.allclose(result, 1.0, rtol=1e-10)

    def test_kernels_with_large_magnitude_vectors(self):
        """Test kernels with vectors of large magnitude."""
        x = np.array([1e6, 2e6, 3e6])
        y = np.array([4e6, 5e6, 6e6])

        # Linear kernel should handle large values
        linear_result = linear_kernel(x, y)
        assert np.isfinite(linear_result)

        # RBF should still be in [0, 1]
        rbf_result = rbf_gaussian_kernel(x, y, sigma=1e6)
        assert 0 <= rbf_result <= 1

    def test_kernels_with_mixed_magnitude_vectors(self):
        """Test kernels with vectors containing mixed magnitudes."""
        x = np.array([1e-10, 1.0, 1e10])
        y = np.array([1e-10, 1.0, 1e10])

        # All kernels should produce finite results
        assert np.isfinite(linear_kernel(x, y))
        assert np.isfinite(affine_kernel(x, y))
        assert np.isfinite(quadratic_kernel(x, y))
        assert np.isfinite(rbf_gaussian_kernel(x, y))
        assert np.isfinite(exponential_kernel(x, y))
        assert np.isfinite(laplacian_kernel(x, y, gamma=1.0))


class TestKernelBatchOperations:
    """Test kernel operations on batches of vectors."""

    def test_kernel_matrix_computation(self):
        """Test efficient computation of kernel matrices."""
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(20, 3)

        # Compute kernel matrix for each kernel type
        kernels_to_test = [
            (linear_kernel, {}),
            (polynomial_kernel, {"degree": 2}),
            (rbf_gaussian_kernel, {"sigma": 1.0}),
        ]

        for kernel_func, params in kernels_to_test:
            K = np.zeros((len(X), len(X)))
            for i in range(len(X)):
                for j in range(len(X)):
                    K[i, j] = kernel_func(X[i], X[j], **params)

            # Check symmetry
            assert np.allclose(K, K.T), f"{kernel_func.__name__} matrix not symmetric"

            # Check diagonal for RBF (should be all 1s)
            if kernel_func == rbf_gaussian_kernel:
                assert np.allclose(np.diag(K), 1.0)


class TestKernelParameterValidation:
    """Test parameter validation for kernels."""

    def test_polynomial_kernel_invalid_degree(self):
        """Test polynomial kernel with invalid degree."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        # Non-integer degree should raise error
        with pytest.raises(ValueError, match="degree must be an integer"):
            polynomial_kernel(x, y, degree=2.5)

    def test_rbf_kernel_negative_sigma(self):
        """Test RBF kernel behavior with negative sigma."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        # Negative sigma should work (squared in formula)
        result = rbf_gaussian_kernel(x, y, sigma=-1.0)
        assert result == rbf_gaussian_kernel(x, y, sigma=1.0)

    def test_kernel_mismatched_dimensions(self):
        """Test kernels with mismatched vector dimensions."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0, 5.0])

        # All kernels should raise appropriate errors
        kernels = [
            (linear_kernel, {}),
            (affine_kernel, {}),
            (polynomial_kernel, {}),
            (rbf_gaussian_kernel, {}),
        ]

        for kernel_func, params in kernels:
            # numpy 2.0 removed AxisError, use Exception instead
            with pytest.raises(
                Exception,
            ):  # Will catch ValueError or other dimension errors
                kernel_func(x, y, **params)


if __name__ == "__main__":
    pytest.main([__file__])
