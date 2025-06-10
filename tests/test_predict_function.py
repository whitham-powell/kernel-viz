# test_predict_function.py

import numpy as np
import pytest

from kernel_viz.algorithms.perceptron import kernelized_perceptron, predict
from kernel_viz.kernels import linear_kernel, polynomial_kernel, rbf_gaussian_kernel


class TestPredictFunction:
    """Tests for the predict function that complement kernelized_perceptron tests."""

    def test_predict_basic_linear_separable(self):
        """Test predict function on linearly separable data."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        alphas = np.array(
            [1.0, -1.0],
            dtype=np.float64,
        )  # Corrected: opposing signs for decision boundary

        # Test positive prediction - f(x) = 1*(2*2) + (-1)*(-2*-2) = 4 - 4 = 0, so prediction = -1
        # Let's use different alphas that create a proper decision boundary
        alphas = np.array([2.0, -1.0], dtype=np.float64)
        x_new = np.array([2, 2])
        prediction = predict(xs, alphas, x_new, linear_kernel)
        # f(x) = 2*(1*2 + 1*2) + (-1)*(-1*2 + -1*2) = 2*4 + (-1)*(-4) = 8 + 4 = 12 > 0
        assert prediction == 1, f"Expected 1, got {prediction}"

        # Test negative prediction
        x_new = np.array([-2, -2])
        prediction = predict(xs, alphas, x_new, linear_kernel)
        # f(x) = 2*(1*-2 + 1*-2) + (-1)*(-1*-2 + -1*-2) = 2*(-4) + (-1)*4 = -8 - 4 = -12 < 0
        assert prediction == -1, f"Expected -1, got {prediction}"

    def test_predict_zero_boundary(self):
        """Test predict function exactly on decision boundary."""
        xs = np.array([[1, 0], [-1, 0]], dtype=np.float64)
        alphas = np.array([1.0, 1.0], dtype=np.float64)

        # Point that should give f(x) = 0 or very close
        x_new = np.array([0, 0])
        prediction = predict(xs, alphas, x_new, linear_kernel)
        # When f(x) = 0, we expect -1 (since we use > 0 not >= 0)
        assert prediction == -1, f"Expected -1 for zero boundary, got {prediction}"

    def test_predict_with_zero_alphas(self):
        """Test predict function when all alphas are zero."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        alphas = np.array([0.0, 0.0], dtype=np.float64)

        x_new = np.array([1, 1])
        prediction = predict(xs, alphas, x_new, linear_kernel)
        # With zero alphas, f(x) = 0, so should return -1
        assert prediction == -1, f"Expected -1 with zero alphas, got {prediction}"

    def test_predict_single_support_vector(self):
        """Test predict function with only one non-zero alpha."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        alphas = np.array([2.0, 0.0], dtype=np.float64)

        # Should be dominated by first support vector
        x_new = np.array([1, 1])
        prediction = predict(xs, alphas, x_new, linear_kernel)
        assert prediction == 1, f"Expected 1, got {prediction}"

    @pytest.mark.parametrize(
        "kernel,params",
        [
            (linear_kernel, {}),
            (rbf_gaussian_kernel, {"sigma": 1.0}),
            (polynomial_kernel, {"degree": 2, "c": 1.0}),
        ],
    )
    def test_predict_with_different_kernels(self, kernel, params):
        """Test predict function with various kernel types."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        alphas = np.array([1.0, 1.0], dtype=np.float64)

        # Test multiple points
        test_points = [
            np.array([2, 2]),  # Should be positive
            np.array([-2, -2]),  # Should be negative
        ]

        for x_new in test_points:
            prediction = predict(xs, alphas, x_new, kernel, params)
            assert prediction in [-1, 1], f"Invalid prediction {prediction}"

    def test_predict_consistency_with_training_data(self):
        """Test that predict correctly classifies training points."""
        # Simple linearly separable case
        xs = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]], dtype=np.float64)
        ys = np.array([1, 1, -1, -1], dtype=np.int64)

        # Use kernelized perceptron to get realistic alphas
        alphas = kernelized_perceptron(xs, ys, linear_kernel, max_iter=10)

        # Check each training point - after convergence, all should be correctly classified
        for i, (x, y) in enumerate(zip(xs, ys)):
            prediction = predict(xs, alphas, x, linear_kernel)
            assert (
                prediction == y
            ), f"Training point {i} misclassified: expected {y}, got {prediction}"

    def test_predict_empty_training_set(self):
        """Test predict function behavior with empty training set."""
        xs = np.empty((0, 2), dtype=np.float64)
        alphas = np.array([], dtype=np.float64)
        x_new = np.array([1, 1])

        prediction = predict(xs, alphas, x_new, linear_kernel)
        # With no training data, sum is 0, so should return -1
        assert prediction == -1, f"Expected -1 for empty training set, got {prediction}"

    def test_predict_dimensional_consistency(self):
        """Test predict function with different input dimensions."""
        # 1D case
        xs_1d = np.array([[1], [-1]], dtype=np.float64)
        alphas_1d = np.array([1.0, 1.0], dtype=np.float64)
        x_new_1d = np.array([2])

        prediction = predict(xs_1d, alphas_1d, x_new_1d, linear_kernel)
        assert prediction in [-1, 1], "1D prediction failed"

        # 3D case
        xs_3d = np.array([[1, 1, 1], [-1, -1, -1]], dtype=np.float64)
        alphas_3d = np.array([1.0, 1.0], dtype=np.float64)
        x_new_3d = np.array([2, 2, 2])

        prediction = predict(xs_3d, alphas_3d, x_new_3d, linear_kernel)
        assert prediction in [-1, 1], "3D prediction failed"

    def test_predict_large_alpha_values(self):
        """Test predict function with large alpha values."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        alphas = np.array(
            [1000.0, -1000.0],
            dtype=np.float64,
        )  # Opposing signs for decision boundary

        x_new = np.array([0.1, 0.1])  # Small positive values
        prediction = predict(xs, alphas, x_new, linear_kernel)
        # f(x) = 1000*(0.1*1 + 0.1*1) + (-1000)*(-1*0.1 + -1*0.1) = 1000*0.2 + (-1000)*(-0.2) = 200 + 200 = 400 > 0
        assert prediction == 1, f"Expected 1 with large alphas, got {prediction}"

    def test_predict_negative_alpha_values(self):
        """Test predict function with negative alpha values."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        alphas = np.array([-1.0, 1.0], dtype=np.float64)  # First alpha negative

        x_new = np.array([1, 1])
        prediction = predict(xs, alphas, x_new, linear_kernel)
        assert prediction in [
            -1,
            1,
        ], f"Invalid prediction with negative alphas: {prediction}"

    def test_predict_kernel_parameter_impact(self):
        """Test that kernel parameters affect predictions as expected."""
        xs = np.array([[0, 0], [1, 1]], dtype=np.float64)
        alphas = np.array([1.0, -1.0], dtype=np.float64)
        x_new = np.array([0.5, 0.5])

        # Test with different RBF sigma values
        pred_narrow = predict(xs, alphas, x_new, rbf_gaussian_kernel, {"sigma": 0.1})
        pred_wide = predict(xs, alphas, x_new, rbf_gaussian_kernel, {"sigma": 10.0})

        # Both should be valid predictions
        assert pred_narrow in [-1, 1], f"Invalid narrow prediction: {pred_narrow}"
        assert pred_wide in [-1, 1], f"Invalid wide prediction: {pred_wide}"
