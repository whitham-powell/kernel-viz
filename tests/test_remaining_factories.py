"""Tests for the remaining factory implementations."""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from kernel_viz.algorithms.perceptron import PerceptronLogger, kernelized_perceptron
from kernel_viz.kernels.base import linear_kernel, rbf_gaussian_kernel
from kernel_viz.visualization.component_factory import (
    KernelMatrixFactory,
    KernelResponseFactory,
    MisclassificationTrackerFactory,
)
from kernel_viz.visualization.core import (
    create_kernel_matrix_component,
    create_kernel_response_component,
    create_misclassification_tracker_component,
)
from kernel_viz.visualization.visualizer import PerceptronVisualizer


@pytest.fixture
def sample_data():
    """Generate consistent test data."""
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([1, -1, -1, 1], dtype=np.float64)
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
        max_iter=5,
        logger=logger,
    )
    return logger.get_logs()


@pytest.fixture
def trained_logs_linear(sample_data):
    """Train model with linear kernel and return logs."""
    X, y = sample_data
    logger = PerceptronLogger()
    kernelized_perceptron(X, y, linear_kernel, max_iter=5, logger=logger)
    return logger.get_logs()


class TestRemainingFactoryEquivalence:
    """Test exact equivalence between factory and original implementations."""

    def test_kernel_response_factory_setup(self, trained_logs_rbf):
        """Test kernel response factory creates correct component."""
        viz = PerceptronVisualizer()

        # Create using both methods
        comp_orig = create_kernel_response_component(viz, trained_logs_rbf)
        factory = KernelResponseFactory(trained_logs_rbf, debug_mode=viz.debug_mode)
        comp_factory = factory.create(
            subplot_params={"gridspec": (0, 1), "colspan": 2},
            name="kernel_response",
        )

        # Test setup produces same number of artists
        fig, (ax1, ax2) = plt.subplots(1, 2)

        artists1 = comp_orig.setup_func(ax1)
        artists2 = comp_factory.setup_func(ax2)

        assert len(artists1) == len(artists2)
        assert len(artists1) == 4  # surface, boundary, points_pos, points_neg

        plt.close(fig)

    def test_kernel_matrix_factory_setup(self, trained_logs_rbf):
        """Test kernel matrix factory creates correct component."""
        viz = PerceptronVisualizer()

        # Create using both methods
        comp_orig = create_kernel_matrix_component(viz, trained_logs_rbf)
        factory = KernelMatrixFactory(trained_logs_rbf, debug_mode=viz.debug_mode)
        comp_factory = factory.create(
            subplot_params={"gridspec": (1, 1)},
            name="kernel_matrix",
        )

        # Test setup produces same number of artists
        fig, (ax1, ax2) = plt.subplots(1, 2)

        artists1 = comp_orig.setup_func(ax1)
        artists2 = comp_factory.setup_func(ax2)

        assert len(artists1) == len(artists2)
        assert len(artists1) == 5  # heatmap + 4 alpha indicators

        # Test update
        artists1 = comp_orig.update_func(2, ax1, artists1)
        artists2 = comp_factory.update_func(2, ax2, artists2)

        # Check titles match
        assert ax1.get_title() == ax2.get_title()

        plt.close(fig)

    def test_misclassification_tracker_factory_setup(self, trained_logs_rbf):
        """Test misclassification tracker factory creates correct component."""
        viz = PerceptronVisualizer()

        # Create using both methods
        comp_orig = create_misclassification_tracker_component(viz, trained_logs_rbf)
        factory = MisclassificationTrackerFactory(
            trained_logs_rbf,
            debug_mode=viz.debug_mode,
        )
        comp_factory = factory.create(
            subplot_params={"gridspec": (0, 0)},
            name="misclassification_tracker",
        )

        # Test setup produces same number of artists
        fig, (ax1, ax2) = plt.subplots(1, 2)

        artists1 = comp_orig.setup_func(ax1)
        artists2 = comp_factory.setup_func(ax2)

        assert len(artists1) == len(artists2)
        assert len(artists1) == 1  # scatter plot

        # Test update
        artists1 = comp_orig.update_func(2, ax1, artists1)
        artists2 = comp_factory.update_func(2, ax2, artists2)

        # Check titles match
        assert ax1.get_title() == ax2.get_title()

        plt.close(fig)

    @pytest.mark.mpl_image_compare(tolerance=1, style="default")
    def test_kernel_response_visual_match(self, trained_logs_rbf):
        """Visual test: Kernel response factory matches original."""
        viz = PerceptronVisualizer()

        # Use factory
        factory = KernelResponseFactory(trained_logs_rbf, debug_mode=viz.debug_mode)
        component = factory.create(
            subplot_params={"gridspec": (0, 1), "colspan": 2},
            name="kernel_response",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        component.update_func(3, ax, artists)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=1, style="default")
    def test_kernel_matrix_visual_match(self, trained_logs_rbf):
        """Visual test: Kernel matrix factory matches original."""
        viz = PerceptronVisualizer()

        # Use factory
        factory = KernelMatrixFactory(trained_logs_rbf, debug_mode=viz.debug_mode)
        component = factory.create(
            subplot_params={"gridspec": (1, 1)},
            name="kernel_matrix",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        component.update_func(2, ax, artists)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=1, style="default")
    def test_misclassification_visual_match(self, trained_logs_rbf):
        """Visual test: Misclassification tracker factory matches original."""
        viz = PerceptronVisualizer()

        # Use factory
        factory = MisclassificationTrackerFactory(
            trained_logs_rbf,
            debug_mode=viz.debug_mode,
        )
        component = factory.create(
            subplot_params={"gridspec": (0, 0)},
            name="misclassification_tracker",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        component.update_func(1, ax, artists)

        return fig

    def test_kernel_response_edge_cases(self):
        """Test kernel response factory handles edge cases."""
        # Test with empty kernel params
        logs = {
            "feature_space": np.array([[0, 0], [1, 1]]),
            "true_labels": np.array([1, -1]),
            "kernel": linear_kernel,
            "kernel_params": None,
            "alphas": [{"alphas": np.array([0.1, -0.1])}],
        }

        factory = KernelResponseFactory(logs)
        assert factory.kernel_params == {}

        # Test with single sample
        logs_single = {
            "feature_space": np.array([[0.5, 0.5]]),
            "true_labels": np.array([1]),
            "kernel": rbf_gaussian_kernel,
            "kernel_params": {"sigma": 1.0},
            "alphas": [{"alphas": np.array([1.0])}],
        }

        factory = KernelResponseFactory(logs_single)
        component = factory.create()

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        assert len(artists) == 4  # Should still have all components
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
