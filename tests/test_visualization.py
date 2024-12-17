# test_visualization.py
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.animation import Animation
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.contour import QuadContourSet
from matplotlib.testing.decorators import check_figures_equal

from src.kernel_visualizer import PerceptronVisualizer
from src.kernelized_perceptron import PerceptronLogger
from src.kernels import linear_kernel, polynomial_kernel, rbf_gaussian_kernel


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

        expected_title = f"Decision Boundary - Iteration {frame + 1}"
        assert (
            ax.get_title() == expected_title
        ), f"Expected title '{expected_title}' but got '{ax.get_title()}'"

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
            assert line.get_alpha() == 0.3  # Initial alpha value

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
        total_lines = len(alphas)

        # Count active (red) lines
        red_lines = sum(
            1
            for line in updated
            if line.get_color() == "red" and line.get_alpha() == 0.7
        )

        # Count inactive (gray) lines
        gray_lines = sum(
            1
            for line in updated
            if line.get_color() == "gray" and line.get_alpha() == 0.1
        )

        assert red_lines == active_count, "Incorrect number of active lines"
        assert gray_lines == (
            total_lines - active_count
        ), "Incorrect number of inactive lines"
        assert red_lines + gray_lines == total_lines, "Total line count mismatch"

        plt.close(fig)


class TestKernelResponseComponent:
    """Tests for the Kernel Response Surface visualization component."""

    @check_figures_equal()
    def test_initial_setup(self, fig_test, fig_ref, sample_logs):
        """Test initial setup of the Kernel Response Surface component."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        # Test figure
        ax_test = fig_test.add_subplot(111)
        artists = component.setup_func(ax_test)

        # Reference figure
        ax_ref = fig_ref.add_subplot(111)
        ax_ref.set_title("Kernel Response Surface")

        # Verify initial setup
        assert len(artists) >= 3  # Surface, contours, and points
        assert any(isinstance(artist, QuadContourSet) for artist in artists)
        assert any(isinstance(artist, PathCollection) for artist in artists)

        # Check axis labels and title
        assert ax_test.get_xlabel() == "Feature 1"
        assert ax_test.get_ylabel() == "Feature 2"
        assert "Kernel Response Surface" in ax_test.get_title()

    @pytest.mark.parametrize("frame", [0, 1, 2])
    def test_frame_updates(self, frame, sample_logs):
        """Test frame-by-frame updates of the kernel response surface."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        fig, ax = plt.subplots()
        initial_artists = component.setup_func(ax)
        updated_artists = component.update_func(frame, ax, initial_artists)

        # Verify frame updates
        assert len(updated_artists) >= len(initial_artists)
        assert ax.get_title() == f"Kernel Response Surface - Iteration {frame + 1}"

        # Check surface exists and has correct properties
        surface = next(
            artist for artist in updated_artists if isinstance(artist, QuadContourSet)
        )
        assert surface.get_array() is not None

        plt.close(fig)

    @pytest.mark.parametrize(
        "kernel,params",
        [
            (linear_kernel, {}),
            (rbf_gaussian_kernel, {"sigma": 1.0}),
            (polynomial_kernel, {"degree": 2, "c": 1.0}),
        ],
    )
    def test_different_kernels(self, kernel, params, sample_logs):
        """Test kernel response visualization with different kernel functions."""
        # Modify logs for different kernel
        modified_logs = sample_logs.copy()
        modified_logs["kernel"] = kernel
        modified_logs["kernel_params"] = params

        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(modified_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        updated_artists = component.update_func(0, ax, artists)

        # Verify surface properties for different kernels
        surface = next(
            artist for artist in updated_artists if isinstance(artist, QuadContourSet)
        )
        response_values = surface.get_array()

        # Check response range is normalized
        assert np.all(response_values >= 0) and np.all(response_values <= 1)

        plt.close(fig)

    def test_response_surface_normalization(self, sample_logs):
        """Test that kernel response values are properly normalized."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Check normalization across multiple frames
        for frame in range(len(sample_logs["alphas"])):
            updated_artists = component.update_func(frame, ax, artists)
            surface = next(
                artist
                for artist in updated_artists
                if isinstance(artist, QuadContourSet)
            )
            response_values = surface.get_array()

            # Verify normalization
            assert np.all(response_values >= 0) and np.all(response_values <= 1)
            # Check for reasonable range
            assert np.ptp(response_values) > 0.1  # At least some variation

        plt.close(fig)

    def test_active_points_highlighting(self, sample_logs):
        """Test that support vectors are properly highlighted."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Check point colors for active/inactive status
        for frame in range(len(sample_logs["alphas"])):
            updated_artists = component.update_func(frame, ax, artists)
            points = next(
                artist
                for artist in updated_artists
                if isinstance(artist, PathCollection)
            )

            alphas = sample_logs["alphas"][frame]["alphas"]
            active_points = np.abs(alphas) > 1e-10

            # Get point colors
            colors = points.get_facecolor()

            # Verify active points are highlighted
            assert len(colors) == len(active_points)
            for color, is_active in zip(colors, active_points):
                if is_active:
                    assert np.array_equal(color[:3], [1, 0, 0])  # Red for active
                else:
                    assert np.array_equal(
                        color[:3],
                        [0.5, 0.5, 0.5],
                    )  # Gray for inactive

        plt.close(fig)

    def test_contour_levels(self, sample_logs):
        """Test that contour levels are appropriate and consistent."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        updated_artists = component.update_func(0, ax, artists)

        # Get contour set
        contours = next(
            artist for artist in updated_artists if isinstance(artist, QuadContourSet)
        )

        # Verify contour properties
        assert len(contours.levels) >= 10  # Reasonable number of levels
        assert contours.levels[0] < 0  # Should include negative values
        assert contours.levels[-1] > 0  # Should include positive values
        assert 0 in contours.levels  # Should include decision boundary

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

    def test_basic_grid_dimensions(self, sample_logs):
        """Test basic grid layout with 1-2 components."""
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

    @pytest.mark.parametrize(
        "n_components,expected_layout",
        [
            (1, (1, 1)),  # Single component: 1x1
            (2, (1, 2)),  # Two components: 1x2
            (3, (2, 2)),  # Three components: 2x2 (with bottom spanning)
            (4, (2, 2)),  # Four components: 2x2
        ],
    )
    def test_multicomponent_grid_layout(
        self,
        sample_logs,
        n_components,
        expected_layout,
    ):
        """Test grid layout with different numbers of components."""
        visualizer = PerceptronVisualizer()

        # Add requested number of components
        for _ in range(n_components):
            visualizer.add_component(
                visualizer.create_decision_boundary_component(sample_logs),
            )

        rows, cols = visualizer._calculate_grid_dimensions()
        assert (rows, cols) == expected_layout

        # For three components, verify bottom component spans columns
        if n_components == 3:
            # Check that last component's gridspec spans columns
            last_component = visualizer.components[-1]
            gridspec = last_component.subplot_params["gridspec"]
            assert isinstance(
                gridspec[1],
                slice,
            ), "Bottom component should span columns"
            assert gridspec[1].start == 0 and gridspec[1].stop == 2

    def test_grid_layout_during_component_addition(self, sample_logs):
        """Test grid layout updates when components are added dynamically."""
        visualizer = PerceptronVisualizer()
        decision_boundary = visualizer.create_decision_boundary_component(sample_logs)
        alpha_evolution = visualizer.create_alpha_evolution_component(sample_logs)

        visualizer.add_component(decision_boundary)
        rows, cols = visualizer._calculate_grid_dimensions()
        assert (rows, cols) == (1, 1)

        visualizer.add_component(alpha_evolution)
        rows, cols = visualizer._calculate_grid_dimensions()
        assert (rows, cols) == (1, 2)

    def test_animation_creation_and_frames(self, sample_logs):
        """Test animation creation and frame count validation."""
        visualizer = PerceptronVisualizer()
        visualizer.add_component(
            visualizer.create_decision_boundary_component(sample_logs),
        )
        visualizer.add_component(
            visualizer.create_alpha_evolution_component(sample_logs),
        )

        animation = visualizer.animate(sample_logs)

        assert isinstance(animation, Animation)
        expected_frames = len(sample_logs["alphas"])
        assert visualizer.total_frames == expected_frames
        plt.close("all")

    @pytest.mark.parametrize("n_components", [1, 2, 3, 4])
    def test_animation_save(self, sample_logs, tmp_path, n_components):
        """Test animation saving with different component counts."""
        visualizer = PerceptronVisualizer()

        # Add requested number of components
        for i in range(n_components):
            if i % 2 == 0:
                visualizer.add_component(
                    visualizer.create_decision_boundary_component(sample_logs),
                )
            else:
                visualizer.add_component(
                    visualizer.create_alpha_evolution_component(sample_logs),
                )

        # Test GIF saving
        gif_path = tmp_path / f"animation_{n_components}_components.gif"
        animation = visualizer.animate(sample_logs, save_path=str(gif_path))

        assert gif_path.exists()
        assert isinstance(animation, Animation)
        plt.close("all")

    def test_animate_no_components(self, sample_logs):
        """Test animate raises error when no components are added."""
        visualizer = PerceptronVisualizer()
        with pytest.raises(ValueError, match="No components added to visualizer"):
            visualizer.animate(sample_logs)


def test_invalid_logs_in_component():
    """Test component creation with invalid logs."""
    visualizer = PerceptronVisualizer()
    invalid_logs = {"feature_space": None}  # Missing keys

    with pytest.raises(KeyError):
        visualizer.create_decision_boundary_component(invalid_logs)


class TestComponentManagement:
    """Tests for component lifecycle, state, and error handling."""

    def test_component_state_and_layout(self, sample_logs):
        """Test component addition and state independence."""
        visualizer = PerceptronVisualizer()

        # Create identical components
        component1 = visualizer.create_decision_boundary_component(sample_logs)
        component2 = visualizer.create_decision_boundary_component(sample_logs)

        # Test addition and position independence
        visualizer.add_component(component1)
        visualizer.add_component(component2)

        positions = [c.subplot_params["gridspec"] for c in visualizer.components]
        assert (
            positions[0] != positions[1]
        ), "Components should have different positions"

        # Test state independence
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        artists1 = component1.setup_func(ax1)
        artists2 = component2.setup_func(ax2)

        # Update to different frames to verify independent state
        updated1 = component1.update_func(0, ax1, artists1)
        updated2 = component2.update_func(1, ax2, artists2)

        assert isinstance(
            updated1[0],
            Artist,
        ), "Component 1 did not return valid artist on update"
        assert isinstance(
            updated2[0],
            Artist,
        ), "Component 2 did not return valid artist on update"

        assert len(updated1) > 0, "Component 1 update returned empty list of Artists"
        assert all(
            artist.axes == ax1 for artist in updated1
        ), "Artists from Component 1 are not linked to the correct axes (ax1)."
        assert all(
            artist.figure == fig1 for artist in updated1
        ), "Artists from Component 1 are not linked to the correct figure (fig1)."

        assert len(updated2) > 0, "Component 2 update returned empty list of Artists"
        assert all(
            artist.axes == ax2 for artist in updated2
        ), "Artists from Component 2 are not linked to the correct axes (ax2)."
        assert all(
            artist.figure == fig2 for artist in updated2
        ), "Artists from Component 2 are not linked to the correct figure (fig2)."

        assert (
            ax1.get_title() != ax2.get_title()
        ), "Titles for Component 1 and Component 2 axes should differ, indicating state independence."

        plt.close("all")

    def test_component_validation(self, sample_logs):
        """Test component validation and error handling."""
        visualizer = PerceptronVisualizer()

        # Test invalid logs
        invalid_logs = {"feature_space": None}
        with pytest.raises(KeyError):
            visualizer.create_decision_boundary_component(invalid_logs)

        # Test partial logs
        partial_logs = {
            "feature_space": sample_logs["feature_space"],
            "true_labels": sample_logs["true_labels"],
            "kernel": sample_logs["kernel"],
            "kernel_params": sample_logs["kernel_params"],
            "alphas": sample_logs["alphas"][:1],
        }

        component = visualizer.create_decision_boundary_component(partial_logs)
        visualizer.add_component(component)
        animation = visualizer.animate(partial_logs)
        assert animation is not None
        assert isinstance(animation, Animation)
        assert (
            len(list(animation.new_frame_seq())) == visualizer.total_frames
        )  # Frame sequence length
        assert visualizer.total_frames == 1
        plt.close("all")

    @pytest.mark.xfail(reason="remove_component not fully implemented")
    def test_component_removal(self, sample_logs):
        """Test component removal when implemented."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)
        visualizer.add_component(component)

        with pytest.raises(NotImplementedError):
            visualizer.remove_component(component)
        plt.close("all")


class TestPerformanceAndResourceManagement:
    """Tests for performance and resource management."""

    def test_figure_cleanup(self, sample_logs):
        """Test proper cleanup of matplotlib figures."""
        visualizer = PerceptronVisualizer()
        initial_figures = len(plt.get_fignums())

        # Run animation
        component = visualizer.create_decision_boundary_component(sample_logs)
        visualizer.add_component(component)
        animation = visualizer.animate(sample_logs)
        assert animation is not None
        assert isinstance(animation, Animation)
        assert (
            len(list(animation.new_frame_seq())) == visualizer.total_frames
        )  # Frame sequence length

        # Check figure cleanup
        plt.close("all")
        final_figures = len(plt.get_fignums())
        assert final_figures == initial_figures, "Figure cleanup failed"

    @pytest.mark.parametrize("n_frames", [1, 10, 100])
    def test_large_frame_count(self, sample_logs, n_frames):
        """Test handling of different numbers of frames."""
        modified_logs = sample_logs.copy()
        modified_logs["alphas"] = [
            {"iteration": i, "alphas": np.zeros(4)} for i in range(n_frames)
        ]

        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(modified_logs)
        visualizer.add_component(component)

        animation = visualizer.animate(modified_logs)

        assert animation is not None
        assert isinstance(animation, Animation)
        assert (
            len(list(animation.new_frame_seq())) == visualizer.total_frames
        )  # Frame sequence length
        assert visualizer.total_frames == n_frames
        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__])
