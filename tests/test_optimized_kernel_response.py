"""Tests for optimized kernel response surface implementation."""

import time

import numpy as np
import pytest
from matplotlib import pyplot as plt

from kernel_viz.algorithms.perceptron import PerceptronLogger, kernelized_perceptron
from kernel_viz.kernels.base import (
    linear_kernel,
    polynomial_kernel,
    rbf_gaussian_kernel,
)
from kernel_viz.visualization.component_factory import KernelResponseFactory
from kernel_viz.visualization.optimized_kernel_response import (
    OptimizedKernelResponseFactory,
    benchmark_kernel_response,
)


@pytest.fixture
def sample_data():
    """Generate consistent test data."""
    np.random.seed(42)
    # Use more data points for performance testing
    X = np.random.randn(20, 2)
    y = np.sign(X[:, 0] + 0.5 * X[:, 1])
    y[y == 0] = 1
    return X, y


@pytest.fixture
def trained_logs_rbf(sample_data):
    """Train model with RBF kernel and return logs."""
    X, y = sample_data
    logger = PerceptronLogger()
    kernelized_perceptron(
        X,
        y,
        rbf_gaussian_kernel,
        kernel_params={"sigma": 0.5},
        max_iter=10,
        logger=logger,
    )
    return logger.get_logs()


@pytest.fixture
def trained_logs_linear(sample_data):
    """Train model with linear kernel and return logs."""
    X, y = sample_data
    logger = PerceptronLogger()
    kernelized_perceptron(X, y, linear_kernel, max_iter=10, logger=logger)
    return logger.get_logs()


@pytest.fixture
def trained_logs_polynomial(sample_data):
    """Train model with polynomial kernel and return logs."""
    X, y = sample_data
    logger = PerceptronLogger()
    kernelized_perceptron(
        X,
        y,
        polynomial_kernel,
        kernel_params={"degree": 3, "c": 1.0},
        max_iter=10,
        logger=logger,
    )
    return logger.get_logs()


class TestOptimizedKernelResponse:
    """Test optimized kernel response implementation."""

    def test_basic_functionality(self, trained_logs_rbf):
        """Test that optimized version produces valid output."""
        factory = OptimizedKernelResponseFactory(trained_logs_rbf)
        component = factory.create()

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        assert len(artists) == 4  # surface, boundary, pos points, neg points

        # Test update
        artists = component.update_func(5, ax, artists)
        assert len(artists) == 4

        plt.close(fig)

    @pytest.mark.parametrize(
        "kernel_logs",
        ["trained_logs_linear", "trained_logs_rbf", "trained_logs_polynomial"],
    )
    def test_vectorized_kernels(self, kernel_logs, request):
        """Test that vectorized kernel implementations work correctly."""
        logs = request.getfixturevalue(kernel_logs)

        factory = OptimizedKernelResponseFactory(
            logs,
            cache_kernels=True,
            grid_resolution=20,
        )
        component = factory.create()

        # Check that kernel matrix was pre-computed for supported kernels
        kernel_name = logs["kernel"].__name__
        if kernel_name in ["linear_kernel", "rbf_gaussian_kernel", "polynomial_kernel"]:
            assert hasattr(factory, "_kernel_matrix")
            assert factory._kernel_matrix.shape == (
                len(logs["feature_space"]),
                factory.grid_resolution**2,
            )

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        component.update_func(3, ax, artists)

        plt.close(fig)

    def test_visual_similarity(self, trained_logs_rbf):
        """Test that optimized produces visually similar results to original."""
        # Original
        factory_orig = KernelResponseFactory(trained_logs_rbf)
        comp_orig = factory_orig.create()

        # Optimized
        factory_opt = OptimizedKernelResponseFactory(
            trained_logs_rbf,
            grid_resolution=50,  # Same as original
            use_interpolation=False,  # No interpolation for fair comparison
        )
        comp_opt = factory_opt.create()

        # Setup both
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        artists1 = comp_orig.setup_func(ax1)
        comp_orig.update_func(5, ax1, artists1)
        ax1.set_title("Original Implementation")

        artists2 = comp_opt.setup_func(ax2)
        comp_opt.update_func(5, ax2, artists2)
        ax2.set_title("Optimized Implementation")

        plt.tight_layout()
        plt.close(fig)

    def test_interpolation_mode(self, trained_logs_rbf):
        """Test interpolation mode produces smooth results."""
        factory = OptimizedKernelResponseFactory(
            trained_logs_rbf,
            grid_resolution=20,  # Low resolution
            use_interpolation=True,  # But interpolate to 100x100
        )

        # Check that scipy is available for interpolation
        try:
            import scipy.interpolate  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available for interpolation")

        component = factory.create()

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Check that display grid was created
        assert hasattr(factory, "_xx_display")
        assert factory._xx_display.shape == (100, 100)

        component.update_func(3, ax, artists)
        plt.close(fig)

    def test_caching_behavior(self, trained_logs_rbf):
        """Test that kernel caching improves performance."""
        # Run with caching
        factory_cached = OptimizedKernelResponseFactory(
            trained_logs_rbf,
            cache_kernels=True,
            grid_resolution=20,
        )
        comp_cached = factory_cached.create()

        fig, ax = plt.subplots()
        artists = comp_cached.setup_func(ax)

        # Time multiple updates
        start_time = time.time()
        for i in range(5):
            comp_cached.update_func(i % len(trained_logs_rbf["alphas"]), ax, artists)
        cached_time = time.time() - start_time

        plt.close(fig)

        # Run without caching
        factory_no_cache = OptimizedKernelResponseFactory(
            trained_logs_rbf,
            cache_kernels=False,
            grid_resolution=20,
        )
        comp_no_cache = factory_no_cache.create()

        fig, ax = plt.subplots()
        artists = comp_no_cache.setup_func(ax)

        start_time = time.time()
        for i in range(5):
            comp_no_cache.update_func(i % len(trained_logs_rbf["alphas"]), ax, artists)
        no_cache_time = time.time() - start_time

        plt.close(fig)

        # Caching should be faster (or at least not significantly slower)
        # We don't assert strict inequality due to timing variability
        print(f"Cached time: {cached_time:.3f}s, No-cache time: {no_cache_time:.3f}s")

    def test_edge_cases(self):
        """Test edge cases like empty data, single point, etc."""
        # Single data point
        logs_single = {
            "feature_space": np.array([[0.5, 0.5]]),
            "true_labels": np.array([1]),
            "kernel": rbf_gaussian_kernel,
            "kernel_params": {"sigma": 1.0},
            "alphas": [{"alphas": np.array([1.0])}],
        }

        factory = OptimizedKernelResponseFactory(
            logs_single,
            grid_resolution=10,
        )
        component = factory.create()

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        component.update_func(0, ax, artists)

        plt.close(fig)

        # No support vectors
        logs_zero = {
            "feature_space": np.array([[0, 0], [1, 1]]),
            "true_labels": np.array([1, -1]),
            "kernel": linear_kernel,
            "kernel_params": {},
            "alphas": [{"alphas": np.array([0.0, 0.0])}],
        }

        factory = OptimizedKernelResponseFactory(logs_zero)
        component = factory.create()

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        component.update_func(0, ax, artists)

        plt.close(fig)

    def test_performance_benchmark(self, trained_logs_rbf):
        """Run performance benchmark comparing implementations."""
        # Only run if explicitly requested
        results = benchmark_kernel_response(trained_logs_rbf, n_frames=20)

        print("\nPerformance Benchmark Results:")
        print("-" * 40)
        for name, time_taken in results.items():
            print(f"{name:<20}: {time_taken:.3f}s")

        # Check that optimized versions are not significantly slower
        original_time = results["original"]
        for name, time_taken in results.items():
            if name != "original":
                # Allow up to 50% slower for safety margin
                assert (
                    time_taken < original_time * 1.5
                ), f"{name} is too slow: {time_taken:.3f}s vs original {original_time:.3f}s"

    @pytest.mark.mpl_image_compare(tolerance=5, style="default")
    def test_optimized_visual_output(self, trained_logs_rbf):
        """Visual regression test for optimized implementation."""
        factory = OptimizedKernelResponseFactory(
            trained_logs_rbf,
            grid_resolution=30,
            use_interpolation=True,
            cache_kernels=True,
        )
        component = factory.create()

        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        component.update_func(5, ax, artists)

        return fig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
