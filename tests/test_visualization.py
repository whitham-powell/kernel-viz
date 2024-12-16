# test_visualization.py
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.animation import Animation
from matplotlib.testing.decorators import check_figures_equal

from src.kernel_visualizer import PerceptronVisualizer
from src.kernelized_perceptron import PerceptronLogger
from src.kernels import rbf_gaussian_kernel


class TestData:
    """Test data generator for consistent test cases."""

    @staticmethod
    def create_xor_dataset():
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([1, -1, -1, 1], dtype=np.float64)
        return X, y

    @staticmethod
    def create_simple_dataset():
        X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.float64)
        y = np.array([1, -1, -1, 1], dtype=np.float64)
        return X, y


@pytest.fixture
def sample_logs():
    """Create sample training data and logs."""
    X, y = TestData.create_xor_dataset()
    logger = PerceptronLogger()

    # Log basic data
    logger.log_feature_space(X)
    logger.log_true_labels(y)
    logger.log_kernel(rbf_gaussian_kernel, {"sigma": 1.0})

    # Log three iterations of training
    alphas = [np.array([1, 0, 0, 0]), np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])]
    for i, alpha in enumerate(alphas):
        logger.log_alphas(i, alpha)
        logger.log_misclassification_count(3 - i)

    return logger.get_logs()


class TestDecisionBoundaryComponent:
    """Test the decision boundary visualization component."""

    @check_figures_equal()
    def test_initial_setup(self, fig_test, fig_ref, sample_logs):
        """Test initial visualization state."""
        # Setup test component
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)
        ax_test = fig_test.add_subplot(111)
        artists = component.setup_func(ax_test)

        # Create reference plot
        ax_ref = fig_ref.add_subplot(111)
        X = sample_logs["feature_space"]
        y = sample_logs["true_labels"]
        ax_ref.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", zorder=2)

        assert len(artists) > 0

    def test_visual_properties(self, sample_logs):
        """Test specific visual properties of the component."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        scatter = artists[0]

        # Check scatter plot properties
        assert scatter.get_edgecolors().shape[1] == 4  # RGBA colors
        assert scatter.get_zorder() == 2  # Points above contour
        assert ax.get_xlabel() == "Feature 1"
        assert ax.get_ylabel() == "Feature 2"

        plt.close(fig)

    @pytest.mark.parametrize("frame", [0, 1, 2])
    def test_frame_transitions(self, sample_logs, frame):
        """Test visualization at specific frames."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Update to specific frame
        updated_artists = component.update_func(frame, ax, artists)

        # Verify frame-specific state
        assert len(updated_artists) >= len(artists)
        assert ax.get_title() == "Decision Boundary"

        plt.close(fig)


class TestAlphaEvolutionComponent:
    """Test the alpha evolution visualization component."""

    def test_setup_properties(self, sample_logs):
        """Test initial setup and properties."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_alpha_evolution_component(sample_logs)
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Check basic properties
        assert len(artists) == len(sample_logs["feature_space"])
        assert ax.get_title() == "Alpha Values Evolution"
        assert ax.get_xlabel() == "Training Iteration"
        assert ax.get_ylabel() == "Alpha Value"

        # Check line properties
        for line in artists:
            assert line.get_linestyle() == "-"
            assert line.get_alpha() in (0.7, 0.1)  # Active or inactive

        plt.close(fig)

    @pytest.mark.parametrize("frame", [0, 1, 2])
    def test_alpha_transitions(self, sample_logs, frame):
        """Test alpha value transitions between frames."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_alpha_evolution_component(sample_logs)
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Update to frame
        updated = component.update_func(frame, ax, artists)

        # Check frame-specific state
        alphas = sample_logs["alphas"][frame]["alphas"]
        active_count = np.sum(np.abs(alphas) > 1e-10)

        # Count red (active) lines
        red_lines = sum(
            1
            for line in updated
            if line.get_color() == "red" and line.get_alpha() == 0.7
        )

        assert red_lines == active_count

        plt.close(fig)


class TestVisualizerIntegration:
    """Test overall visualizer behavior."""

    def test_component_registration(self, sample_logs):
        """Test component addition and management."""
        visualizer = PerceptronVisualizer()

        # Add components
        components = [
            visualizer.create_decision_boundary_component(sample_logs),
            visualizer.create_alpha_evolution_component(sample_logs),
        ]

        for component in components:
            visualizer.add_component(component)

        assert len(visualizer.components) == len(components)

        # Check grid layout
        for component in visualizer.components:
            assert "gridspec" in component.subplot_params

    def test_animation_creation(self, sample_logs):
        """Test animation creation and properties."""
        visualizer = PerceptronVisualizer()

        # Add components
        visualizer.add_component(
            visualizer.create_decision_boundary_component(sample_logs),
        )
        visualizer.add_component(
            visualizer.create_alpha_evolution_component(sample_logs),
        )

        # Create animation
        animation = visualizer.animate(sample_logs)

        assert isinstance(animation, Animation)
        assert animation.save_count == len(sample_logs["alphas"])

        plt.close("all")

    def test_grid_layout(self, sample_logs):
        """Test grid layout calculation and updates."""
        visualizer = PerceptronVisualizer()

        # Test single component
        visualizer.add_component(
            visualizer.create_decision_boundary_component(sample_logs),
        )
        rows, cols = visualizer._calculate_grid_dimensions()
        assert (rows, cols) == (1, 1)

        # Test two components
        visualizer.add_component(
            visualizer.create_alpha_evolution_component(sample_logs),
        )
        rows, cols = visualizer._calculate_grid_dimensions()
        assert (rows, cols) == (1, 2)

        # Verify component positions
        positions = [c.subplot_params["gridspec"] for c in visualizer.components]
        assert positions[0] != positions[1]


def test_animation_save(sample_logs, tmp_path):
    """Test animation saving functionality."""
    visualizer = PerceptronVisualizer()
    visualizer.add_component(visualizer.create_decision_boundary_component(sample_logs))

    # Test GIF saving
    gif_path = tmp_path / "animation.gif"
    animation = visualizer.animate(sample_logs, save_path=str(gif_path))  # noqa : F841
    assert gif_path.exists()

    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__])
