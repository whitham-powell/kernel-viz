"""Visual regression tests to ensure factory implementations match original output."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kernel_viz.algorithms.perceptron import PerceptronLogger, kernelized_perceptron
from kernel_viz.kernels.base import linear_kernel, rbf_gaussian_kernel
from kernel_viz.visualization.component_factory import (
    AlphaEvolutionFactory,
    DecisionBoundaryFactory,
    KernelMatrixHeatmapFactory,
)


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


class TestFactoryVisualRegression:
    """Visual regression tests comparing factory to original implementations."""

    @pytest.mark.mpl_image_compare(tolerance=1, style="default")
    def test_decision_boundary_rbf_visual(self, trained_logs_rbf):
        """Visual test: Decision boundary with RBF kernel."""
        factory = DecisionBoundaryFactory(trained_logs_rbf)
        component = factory.create()

        fig, ax = plt.subplots(figsize=(6, 6))
        artists = component.setup_func(ax)
        # Test frame 4 (after training stabilizes)
        component.update_func(4, ax, artists)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=1, style="default")
    def test_decision_boundary_linear_visual(self, trained_logs_linear):
        """Visual test: Decision boundary with linear kernel."""
        factory = DecisionBoundaryFactory(trained_logs_linear)
        component = factory.create()

        fig, ax = plt.subplots(figsize=(6, 6))
        artists = component.setup_func(ax)
        component.update_func(4, ax, artists)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=1, style="default")
    def test_alpha_evolution_visual(self, trained_logs_rbf):
        """Visual test: Alpha evolution component."""
        factory = AlphaEvolutionFactory(trained_logs_rbf)
        factory.total_frames = 5
        component = factory.create()

        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        component.update_func(4, ax, artists)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=1, style="default")
    def test_kernel_matrix_heatmap_visual(self, trained_logs_rbf):
        """Visual test: Kernel matrix heatmap component."""
        factory = KernelMatrixHeatmapFactory(trained_logs_rbf)
        component = factory.create()

        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        component.update_func(2, ax, artists)

        return fig

    def test_decision_boundary_fixed_dims(self, trained_logs_rbf):
        """Test decision boundary with fixed dimensions parameter."""
        # Modify data to be 3D
        logs_3d = trained_logs_rbf.copy()
        X_3d = np.random.randn(5, 3)
        logs_3d["feature_space"] = X_3d
        logs_3d["true_labels"] = np.array([1, -1, 1, -1, 1])  # Match the 5 samples
        logs_3d["alphas"] = [{"alphas": np.random.randn(5)} for _ in range(3)]

        factory = DecisionBoundaryFactory(logs_3d, fixed_dims={2: 0.5})
        component = factory.create()

        # Should not raise error
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        component.update_func(0, ax, artists)
        plt.close(fig)

        assert factory.fixed_dims == {2: 0.5}

    def test_alpha_evolution_with_custom_frames(self, trained_logs_rbf):
        """Test alpha evolution with custom total_frames."""
        factory = AlphaEvolutionFactory(trained_logs_rbf)
        factory.total_frames = 10  # Custom frame count
        component = factory.create()

        fig, ax = plt.subplots()
        component.setup_func(ax)

        # Check x-axis limits match total_frames
        xlim = ax.get_xlim()
        assert xlim[1] == 10

        plt.close(fig)


class TestFactoryNumericalEquivalence:
    """Test numerical equivalence between factory and original implementations."""

    def test_decision_boundary_numerical_output(self, trained_logs_rbf):
        """Test that decision boundary computation is numerically identical."""
        from kernel_viz.visualization.core import compute_decision_boundary

        # Get same frame data
        alphas = trained_logs_rbf["alphas"][2]["alphas"]
        xs = trained_logs_rbf["feature_space"]
        kernel = trained_logs_rbf["kernel"]
        kernel_params = trained_logs_rbf["kernel_params"]

        # Compute decision boundary
        xx1, yy1, zz1 = compute_decision_boundary(xs, alphas, kernel, kernel_params)
        xx2, yy2, zz2 = compute_decision_boundary(xs, alphas, kernel, kernel_params)

        # Should be identical
        assert np.allclose(xx1, xx2)
        assert np.allclose(yy1, yy2)
        assert np.allclose(zz1, zz2)

    def test_factory_preserves_parameters(self, trained_logs_rbf):
        """Test that factory preserves all initialization parameters."""
        # Test DecisionBoundaryFactory
        db_factory = DecisionBoundaryFactory(
            trained_logs_rbf,
            plot_type="line",
            fixed_dims={2: 0.5},
        )
        assert db_factory.plot_type == "line"
        assert db_factory.fixed_dims == {2: 0.5}

        # Test AlphaEvolutionFactory
        ae_factory = AlphaEvolutionFactory(trained_logs_rbf, debug_mode=True)
        assert ae_factory.debug_mode is True

        # Test custom total_frames
        ae_factory.total_frames = 20
        assert ae_factory.total_frames == 20

    def test_component_subplot_params(self, trained_logs_rbf):
        """Test that components can have custom subplot parameters."""
        factory = DecisionBoundaryFactory(trained_logs_rbf)
        custom_params = {"gridspec": (1, 1), "colspan": 2}
        component = factory.create(subplot_params=custom_params, name="test_component")

        assert component.subplot_params == custom_params
        assert component.name == "test_component"


class TestFactoryEdgeCases:
    """Test factory implementations with edge cases."""

    def test_empty_kernel_params(self):
        """Test factories handle None kernel_params."""
        logs = {
            "feature_space": np.array([[0, 0], [1, 1]]),
            "true_labels": np.array([1, -1]),
            "kernel": linear_kernel,
            "kernel_params": None,
            "alphas": [{"alphas": np.array([0.1, -0.1])}],
        }

        factory = DecisionBoundaryFactory(logs)
        assert factory.kernel_params == {}

    def test_single_sample(self):
        """Test with single training sample."""
        logs = {
            "feature_space": np.array([[0.5, 0.5]]),
            "true_labels": np.array([1]),
            "kernel": rbf_gaussian_kernel,
            "kernel_params": {"sigma": 1.0},
            "alphas": [{"alphas": np.array([1.0])}],
        }

        # Should not crash
        factory = AlphaEvolutionFactory(logs)
        component = factory.create()

        fig, ax = plt.subplots()
        setup_artists = component.setup_func(ax)
        assert len(setup_artists) == 1  # One line for one sample
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
