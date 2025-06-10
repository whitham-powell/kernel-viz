"""Tests for kernel matrix heatmap animation component using factory pattern."""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from kernel_viz.algorithms.perceptron import PerceptronLogger
from kernel_viz.kernels import linear_kernel, rbf_gaussian_kernel
from kernel_viz.visualization.base import AnimationComponent
from kernel_viz.visualization.component_factory import (
    KernelMatrixHeatmapFactory,
    create_component,
)
from kernel_viz.visualization.core import PerceptronVisualizer


@pytest.fixture
def sample_logs():
    """Create sample training data and logs with kernel matrix."""
    X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.float64)
    y = np.array([1, -1, -1, 1], dtype=np.float64)
    logger = PerceptronLogger()
    
    # Log basic data
    logger.log_feature_space(X)
    logger.log_true_labels(y)
    logger.log_kernel(rbf_gaussian_kernel, {"sigma": 1.0})
    
    # Compute and log kernel matrix
    n_samples = len(X)
    kernel_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = rbf_gaussian_kernel(X[i], X[j], sigma=1.0)
    logger.logs["kernel_matrix"] = kernel_matrix
    
    # Log three iterations of training
    alphas = [
        np.array([0.5, 0, 0, 0]),
        np.array([0.5, 0.3, 0, 0]),
        np.array([0.5, 0.3, 0.2, 0]),
    ]
    for i, alpha in enumerate(alphas):
        logger.log_alphas(i, alpha)
        logger.log_misclassification_count(3 - i)
    
    return logger.get_logs()


@pytest.fixture
def linear_logs():
    """Create sample logs with linear kernel for testing."""
    X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
    y = np.array([1, 1, -1, -1], dtype=np.float64)
    logger = PerceptronLogger()
    
    logger.log_feature_space(X)
    logger.log_true_labels(y)
    logger.log_kernel(linear_kernel, {})
    
    # Log iterations
    alphas = [
        np.array([0.2, 0, 0, 0]),
        np.array([0.2, 0.1, 0, 0]),
        np.array([0.2, 0.1, 0, 0.1]),
    ]
    for i, alpha in enumerate(alphas):
        logger.log_alphas(i, alpha)
    
    return logger.get_logs()


class TestKernelMatrixHeatmapFactory:
    """Tests for KernelMatrixHeatmapFactory class."""
    
    def test_factory_initialization(self, sample_logs):
        """Test factory initialization and data extraction."""
        factory = KernelMatrixHeatmapFactory(sample_logs)
        
        assert factory.kernel_matrix is not None
        assert factory.kernel_matrix.shape == (4, 4)
        assert factory.n_samples == 4
        assert len(factory.alphas_history) == 3
    
    def test_factory_computes_kernel_matrix_if_missing(self, linear_logs):
        """Test that factory computes kernel matrix if not provided."""
        # Remove kernel matrix from logs
        linear_logs.pop("kernel_matrix", None)
        
        factory = KernelMatrixHeatmapFactory(linear_logs)
        
        assert factory.kernel_matrix is not None
        assert factory.kernel_matrix.shape == (4, 4)
        # Check it's computed correctly for linear kernel
        X = linear_logs["feature_space"]
        expected = X @ X.T
        np.testing.assert_allclose(factory.kernel_matrix, expected)
    
    def test_factory_create_method(self, sample_logs):
        """Test factory create method returns proper component."""
        factory = KernelMatrixHeatmapFactory(sample_logs)
        component = factory.create(
            subplot_params={"gridspec": (0, 0)},
            name="test_heatmap"
        )
        
        assert isinstance(component, AnimationComponent)
        assert component.name == "test_heatmap"
        assert component.subplot_params == {"gridspec": (0, 0)}
        assert callable(component.setup_func)
        assert callable(component.update_func)


class TestKernelMatrixHeatmapVisualization:
    """Tests for kernel matrix heatmap visualization."""
    
    def test_setup_creates_heatmap(self, sample_logs):
        """Test that setup creates a heatmap image."""
        factory = KernelMatrixHeatmapFactory(sample_logs)
        fig, ax = plt.subplots()
        
        artists = factory.setup(ax)
        
        # Check that we have an image as first artist
        assert len(artists) > 0
        assert isinstance(artists[0], AxesImage)
        
        # Check colorbar was added
        assert len(fig.axes) == 2  # Main axes + colorbar axes
        
        # Check labels
        assert ax.get_xlabel() == "Sample Index"
        assert ax.get_ylabel() == "Sample Index"
        assert "Kernel Matrix" in ax.get_title()
        
        plt.close(fig)
    
    def test_support_vector_markers_created(self, sample_logs):
        """Test that support vector markers are created."""
        factory = KernelMatrixHeatmapFactory(sample_logs)
        fig, ax = plt.subplots()
        
        artists = factory.setup(ax)
        
        # Should have image + markers
        assert len(artists) > 1
        
        # Check diagonal markers
        assert len(factory.sv_markers_diag) == factory.n_samples
        for marker in factory.sv_markers_diag:
            assert isinstance(marker, Line2D)
            assert marker.get_markersize() == 0  # Initially hidden
        
        plt.close(fig)
    
    def test_update_shows_support_vectors(self, sample_logs):
        """Test that update correctly shows support vectors."""
        factory = KernelMatrixHeatmapFactory(sample_logs)
        fig, ax = plt.subplots()
        
        artists = factory.setup(ax)
        
        # Update to frame 1 (two support vectors)
        factory.update(1, ax, artists)
        
        # Check that first two diagonal markers are visible
        assert factory.sv_markers_diag[0].get_markersize() > 0
        assert factory.sv_markers_diag[1].get_markersize() > 0
        assert factory.sv_markers_diag[2].get_markersize() == 0
        assert factory.sv_markers_diag[3].get_markersize() == 0
        
        # Check title updated
        assert "Iteration 2" in ax.get_title()
        assert "Support Vectors: 2/4" in ax.get_title()
        
        plt.close(fig)
    
    def test_marker_colors_based_on_alpha_sign(self, sample_logs):
        """Test that marker colors reflect alpha sign."""
        # Modify alphas to have negative values
        sample_logs["alphas"][2]["alphas"] = np.array([0.5, -0.3, 0.2, 0])
        
        factory = KernelMatrixHeatmapFactory(sample_logs)
        fig, ax = plt.subplots()
        
        artists = factory.setup(ax)
        factory.update(2, ax, artists)
        
        # Check marker colors
        assert factory.sv_markers_diag[0].get_color() == "yellow"  # positive
        assert factory.sv_markers_diag[1].get_color() == "cyan"    # negative
        assert factory.sv_markers_diag[2].get_color() == "yellow"  # positive
        
        plt.close(fig)


class TestFactoryIntegration:
    """Test integration with create_component factory method."""
    
    def test_create_component_factory_method(self, sample_logs):
        """Test creating component through factory method."""
        component = create_component(
            "kernel_matrix_heatmap",
            sample_logs,
            debug_mode=False
        )
        
        assert isinstance(component, AnimationComponent)
        assert callable(component.setup_func)
        assert callable(component.update_func)
    
    def test_unknown_component_raises_error(self, sample_logs):
        """Test that unknown component type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown component type"):
            create_component("unknown_component", sample_logs)


class TestVisualizerIntegration:
    """Test integration with PerceptronVisualizer."""
    
    def test_visualizer_method_creates_component(self, sample_logs):
        """Test that visualizer can create kernel matrix heatmap component."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_matrix_heatmap_component(sample_logs)
        
        assert isinstance(component, AnimationComponent)
        assert component.name == "kernel_matrix_heatmap"
        assert component.subplot_params == {"gridspec": (1, 1)}
    
    def test_visualizer_debug_mode_propagates(self, sample_logs):
        """Test that debug mode propagates to factory."""
        visualizer = PerceptronVisualizer()
        visualizer.set_debug_mode(True)
        
        # Capture debug output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        component = visualizer.create_kernel_matrix_heatmap_component(sample_logs)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should see debug output from factory
        assert "Kernel matrix shape:" in output
        assert "Kernel matrix range:" in output
    
    @pytest.mark.mpl_image_compare(tolerance=10, style="default")
    def test_kernel_matrix_visualization_snapshot(self, sample_logs):
        """Visual regression test for kernel matrix heatmap."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_matrix_heatmap_component(sample_logs)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        
        # Update to show some support vectors
        component.update_func(1, ax, artists)
        
        return fig