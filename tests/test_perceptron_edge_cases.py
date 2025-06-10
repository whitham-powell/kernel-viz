# test_perceptron_edge_cases.py

from unittest.mock import patch

import numpy as np

from src.kernelized_perceptron import PerceptronLogger, kernelized_perceptron
from src.kernels import linear_kernel, rbf_gaussian_kernel


class TestKernelizedPerceptronEdgeCases:
    """Tests for edge cases and error conditions in kernelized perceptron."""

    def test_early_convergence(self):
        """Test that algorithm terminates early when no misclassifications occur."""
        # Linearly separable data that should converge quickly
        xs = np.array([[2, 2], [3, 3], [-2, -2], [-3, -3]], dtype=np.float64)
        ys = np.array([1, 1, -1, -1], dtype=np.int64)

        logger = PerceptronLogger()
        alphas = kernelized_perceptron(
            xs,
            ys,
            linear_kernel,
            max_iter=10,
            logger=logger,
        )

        # Should converge before max_iter
        logs = logger.get_logs()
        final_misclassifications = logs["misclassification_count"][-1]
        assert (
            final_misclassifications == 0
        ), "Should converge to zero misclassifications"

        # Verify all training points are correctly classified
        from src.kernelized_perceptron import predict

        for i, (x, y) in enumerate(zip(xs, ys)):
            prediction = predict(xs, alphas, x, linear_kernel)
            assert (
                prediction == y
            ), f"Training point {i} misclassified after convergence"

    def test_max_iterations_reached(self):
        """Test behavior when max iterations is reached without convergence."""
        # XOR-like data that's hard to separate with linear kernel
        xs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.float64)
        ys = np.array([1, 1, -1, -1], dtype=np.int64)

        logger = PerceptronLogger()
        kernelized_perceptron(xs, ys, linear_kernel, max_iter=3, logger=logger)

        # Should have exactly 3 iterations logged
        logs = logger.get_logs()
        assert (
            len(logs["alphas"]) == 3
        ), f"Expected 3 iterations, got {len(logs['alphas'])}"
        assert (
            len(logs["misclassification_count"]) == 3
        ), "Should have 3 misclassification counts"

    def test_single_point_dataset(self):
        """Test with dataset containing only one point."""
        xs = np.array([[1, 1]], dtype=np.float64)
        ys = np.array([1], dtype=np.int64)

        logger = PerceptronLogger()
        alphas = kernelized_perceptron(xs, ys, linear_kernel, logger=logger)

        # Should have alpha[0] = y[0] = 1
        assert len(alphas) == 1, "Should have exactly one alpha"
        assert alphas[0] == 1, f"Expected alpha[0] = 1, got {alphas[0]}"

        # Should complete one iteration with 0 misclassifications (no other points to classify)
        logs = logger.get_logs()
        assert (
            len(logs["misclassification_count"]) > 0
        ), "Should have logged misclassification counts"

    def test_identical_points_different_labels(self):
        """Test with identical points having different labels."""
        xs = np.array([[1, 1], [1, 1]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.int64)

        logger = PerceptronLogger()
        alphas = kernelized_perceptron(xs, ys, linear_kernel, max_iter=5, logger=logger)

        # The algorithm initializes alphas[0] = ys[0] = 1, and only updates alphas[1] when misclassified
        # For identical points, the decision is based on the net contribution of both alphas
        # This test checks that the algorithm handles this edge case without crashing
        logs = logger.get_logs()
        assert (
            len(logs["misclassification_count"]) == 5
        ), "Should complete all iterations"
        assert len(alphas) == 2, "Should have alphas for both points"

    def test_zero_kernel_values(self):
        """Test behavior when kernel returns zero values."""

        def zero_kernel(x, y):
            return 0.0

        xs = np.array([[1, 1], [2, 2]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.int64)

        alphas = kernelized_perceptron(xs, ys, zero_kernel, max_iter=3)

        # With zero kernel, all points should be misclassified after first
        # (since f(x) = 0 for all x, and we predict -1 when f(x) <= 0)
        assert len(alphas) == 2, "Should have two alphas"

    def test_very_large_kernel_values(self):
        """Test behavior with very large kernel values."""

        def large_kernel(x, y):
            return 1e10 * linear_kernel(x, y)

        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.int64)

        alphas = kernelized_perceptron(xs, ys, large_kernel, max_iter=3)

        # Should still work, just with large intermediate values
        assert len(alphas) == 2, "Should have two alphas"
        assert all(np.isfinite(alphas)), "All alphas should be finite"

    def test_kernel_parameter_none_handling(self):
        """Test handling of None kernel parameters."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.int64)

        # Should handle None parameters gracefully
        alphas = kernelized_perceptron(xs, ys, linear_kernel, kernel_params=None)
        assert len(alphas) == 2, "Should handle None parameters"

    def test_misclassification_counting_accuracy(self):
        """Test that misclassification counting is accurate."""
        # Known case where we can predict misclassifications
        xs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
        ys = np.array([1, 1, -1, -1], dtype=np.int64)

        logger = PerceptronLogger()
        kernelized_perceptron(xs, ys, linear_kernel, max_iter=1, logger=logger)

        logs = logger.get_logs()
        misclassification_counts = logs["misclassification_count"]

        # Verify counts are non-negative integers
        for count in misclassification_counts:
            assert isinstance(
                count,
                (int, np.integer),
            ), f"Count should be integer, got {type(count)}"
            assert count >= 0, f"Count should be non-negative, got {count}"
            assert (
                count <= len(xs) - 1
            ), "Count should not exceed number of points being classified"

    def test_alpha_evolution_consistency(self):
        """Test that alpha evolution is consistent across iterations."""
        xs = np.array([[1, 1], [2, 2], [-1, -1]], dtype=np.float64)
        ys = np.array([1, 1, -1], dtype=np.int64)

        logger = PerceptronLogger()
        kernelized_perceptron(
            xs,
            ys,
            linear_kernel,
            max_iter=3,
            logger=logger,
        )

        logs = logger.get_logs()
        alpha_evolution = logs["alphas"]

        # Check that alphas only increase (never decrease) since we only add ys[i]
        for i in range(len(alpha_evolution) - 1):
            current_alphas = alpha_evolution[i]["alphas"]
            next_alphas = alpha_evolution[i + 1]["alphas"]

            # Alphas should only change in the direction of the labels
            diff = next_alphas - current_alphas
            for j, d in enumerate(diff):
                if d != 0:
                    # If alpha changed, it should be in direction of ys[j]
                    assert (
                        d * ys[j] > 0
                    ), f"Alpha change inconsistent with label at iteration {i}, point {j}"

    def test_logger_capture_during_training(self):
        """Test that logger captures all expected information during training."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.int64)

        logger = PerceptronLogger()
        kernelized_perceptron(
            xs,
            ys,
            rbf_gaussian_kernel,
            kernel_params={"sigma": 1.0},
            max_iter=2,
            logger=logger,
        )

        logs = logger.get_logs()

        # Check all expected data is logged
        assert (
            logs["kernel"] is rbf_gaussian_kernel
        ), "Kernel function not logged correctly"
        assert logs["kernel_params"] == {
            "sigma": 1.0,
        }, "Kernel parameters not logged correctly"
        assert np.array_equal(
            logs["feature_space"],
            xs,
        ), "Feature space not logged correctly"
        assert np.array_equal(
            logs["true_labels"],
            ys,
        ), "True labels not logged correctly"

        # Check iteration-specific data
        assert len(logs["alphas"]) == 2, "Should have 2 iterations of alphas"
        assert (
            len(logs["misclassification_count"]) == 2
        ), "Should have 2 misclassification counts"

        for i, alpha_entry in enumerate(logs["alphas"]):
            assert alpha_entry["iteration"] == i, f"Iteration number mismatch at {i}"
            assert len(alpha_entry["alphas"]) == len(xs), "Alpha vector length mismatch"

    @patch("builtins.print")
    def test_training_output_suppression(self, mock_print):
        """Test that print statements during training work as expected."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.int64)

        logger = PerceptronLogger()
        kernelized_perceptron(xs, ys, linear_kernel, max_iter=1, logger=logger)

        # Verify print was called (training output)
        mock_print.assert_called()

        # Check that print was called with expected format
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any(
            "Epoch" in str(call) and "Misclassified" in str(call)
            for call in print_calls
        ), "Expected training output not found"

    def test_numerical_stability_small_differences(self):
        """Test numerical stability with very small differences between points."""
        eps = 1e-10
        xs = np.array([[1, 1], [1 + eps, 1 + eps]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.int64)

        alphas = kernelized_perceptron(xs, ys, linear_kernel, max_iter=3)

        # Should handle small differences without numerical issues
        assert all(
            np.isfinite(alphas),
        ), "Alphas should remain finite with small differences"
        assert len(alphas) == 2, "Should have correct number of alphas"

    def test_algorithm_determinism(self):
        """Test that algorithm is deterministic for same inputs."""
        xs = np.array([[1, 1], [2, 2], [-1, -1]], dtype=np.float64)
        ys = np.array([1, 1, -1], dtype=np.int64)

        # Run twice with same parameters
        alphas1 = kernelized_perceptron(xs, ys, linear_kernel, max_iter=3)
        alphas2 = kernelized_perceptron(xs, ys, linear_kernel, max_iter=3)

        assert np.array_equal(alphas1, alphas2), "Algorithm should be deterministic"

    def test_memory_efficiency_large_iteration_count(self):
        """Test memory usage with large iteration count."""
        xs = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        ys = np.array([1, -1], dtype=np.int64)

        logger = PerceptronLogger()
        # Use a large iteration count
        alphas = kernelized_perceptron(
            xs,
            ys,
            linear_kernel,
            max_iter=1000,
            logger=logger,
        )

        logs = logger.get_logs()

        # Should not store more data than necessary
        assert len(logs["alphas"]) <= 1000, "Should not exceed max iterations"
        assert (
            len(logs["misclassification_count"]) <= 1000
        ), "Should not exceed max iterations"

        # Final alphas should be reasonable
        assert all(np.isfinite(alphas)), "Final alphas should be finite"
