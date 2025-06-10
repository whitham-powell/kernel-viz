# test_visualization.py

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.animation import Animation
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D

from kernel_viz.algorithms.perceptron import PerceptronLogger
from kernel_viz.kernels import linear_kernel, polynomial_kernel, rbf_gaussian_kernel
from kernel_viz.visualization.core import PerceptronVisualizer


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
    """Tests for the decision boundary visualization component."""

    def test_component_setup(self, sample_logs):
        """Test component creation and basic attributes."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)

        # Test required attributes exist
        assert callable(component.setup_func)
        assert callable(component.update_func)
        assert "gridspec" in component.subplot_params

        # Test subplot params are correct format
        assert isinstance(component.subplot_params["gridspec"], tuple)

    def test_initial_visualization(self, sample_logs):
        """Test initial visualization state and scatter plot setup."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)

        # Create test figure and set up the component
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Verify basic visual elements
        assert len(artists) > 0, "Expected at least one artist in setup"
        scatter = next(
            (artist for artist in artists if isinstance(artist, PathCollection)),
            None,
        )
        assert scatter is not None, "Scatter plot missing from initial visualization"

        # Check axis labels and title
        assert ax.get_xlabel() == "Feature 1", "X-axis label mismatch"
        assert ax.get_ylabel() == "Feature 2", "Y-axis label mismatch"
        assert "Decision Boundary - Iteration 1" in ax.get_title(), "Title mismatch"

        plt.close(fig)

    def test_visual_properties(self, sample_logs):
        """Test visual properties of the component."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Check axis labels and title
        assert ax.get_xlabel() == "Feature 1"
        assert ax.get_ylabel() == "Feature 2"
        assert "Decision Boundary" in ax.get_title()

        # Check scatter plot exists with correct properties
        scatter = artists[0]
        assert scatter.get_edgecolors().shape[1] == 4  # RGBA colors
        assert scatter.get_zorder() == 2  # Points above contour

        plt.close(fig)

    @pytest.mark.parametrize("frame", [0, 1, 2])
    def test_frame_updates(self, frame, sample_logs):
        """Test visualization updates at specific frames."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)

        fig, ax = plt.subplots()
        initial_artists = component.setup_func(ax)
        updated_artists = component.update_func(frame, ax, initial_artists)

        # Verify frame update results
        assert len(updated_artists) >= len(initial_artists)

        # Verify frame-specific title
        expected_title = f"Decision Boundary - Iteration {frame + 1}"
        assert ax.get_title() == expected_title

        plt.close(fig)

    def test_boundary_consistency(self, sample_logs):
        """Test that decision boundary updates are smooth between frames."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(sample_logs)
        fig, ax = plt.subplots()

        initial_artists = component.setup_func(ax)
        previous_artists = None

        for frame in range(3):
            updated_artists = component.update_func(frame, ax, initial_artists)

            # Verify artists are updated
            assert len(updated_artists) >= len(initial_artists)

            # Verify consistent number of artists between frames
            if previous_artists is not None:
                assert len(updated_artists) == len(previous_artists)

            previous_artists = updated_artists

        plt.close(fig)


class TestAlphaEvolutionComponent:
    """Tests for the alpha evolution visualization component."""

    def test_component_setup(self, sample_logs):
        """Test component creation and basic attributes."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_alpha_evolution_component(sample_logs)

        # Test required attributes exist
        assert callable(component.setup_func)
        assert callable(component.update_func)
        assert "gridspec" in component.subplot_params

        # Test subplot params are correct format
        assert isinstance(component.subplot_params["gridspec"], tuple)

    def test_initial_visualization(self, sample_logs):
        """Test initial visualization state and alpha line setup."""

        # Manually create the reference figure
        fig_ref, ax_ref = plt.subplots()
        n_samples = len(sample_logs["feature_space"])
        for i in range(n_samples):
            ax_ref.plot(
                [],
                [],
                label=f"$\\alpha_{{{i}}}$",
                alpha=0.3,
                linewidth=0.5,
                color="gray",
            )
        ax_ref.set_title("Alpha Values Evolution")
        ax_ref.set_xlabel("Training Iteration")
        ax_ref.set_ylabel("Alpha Value")
        ax_ref.grid(True, linestyle="--", alpha=0.7)

        # Generate the test figure using the visualizer
        visualizer = PerceptronVisualizer()
        component = visualizer.create_alpha_evolution_component(sample_logs)
        fig_test, ax_test = plt.subplots()
        artists = component.setup_func(ax_test)

        assert artists is not None

        # Compare axes properties
        assert ax_ref.get_title() == ax_test.get_title(), "Titles do not match"
        assert ax_ref.get_xlabel() == ax_test.get_xlabel(), "X-axis labels do not match"
        assert ax_ref.get_ylabel() == ax_test.get_ylabel(), "Y-axis labels do not match"

        # Verify grid properties
        assert (
            ax_ref.xaxis.get_gridlines()[0].get_linestyle()
            == ax_test.xaxis.get_gridlines()[0].get_linestyle()
        ), "Grid line styles do not match"
        assert (
            ax_ref.xaxis.get_gridlines()[0].get_alpha()
            == ax_test.xaxis.get_gridlines()[0].get_alpha()
        ), "Grid line alphas do not match"

        # Verify lines
        ref_lines = [line for line in ax_ref.get_lines()]
        test_lines = [line for line in ax_test.get_lines()]
        assert len(ref_lines) == len(test_lines), "Number of lines does not match"
        for ref_line, test_line in zip(ref_lines, test_lines):
            assert isinstance(ref_line, Line2D), "Reference artist is not a Line2D"
            assert isinstance(test_line, Line2D), "Test artist is not a Line2D"
            assert (
                ref_line.get_alpha() == test_line.get_alpha()
            ), "Line alphas do not match"
            assert (
                ref_line.get_linewidth() == test_line.get_linewidth()
            ), "Line widths do not match"

        # Cleanup
        plt.close(fig_ref)
        plt.close(fig_test)

    def test_visual_properties(self, sample_logs):
        """Test visual properties of the component."""
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
    def test_frame_updates(self, frame, sample_logs):
        """Test general frame update behavior."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_alpha_evolution_component(sample_logs)

        fig, ax = plt.subplots()
        initial_artists = component.setup_func(ax)
        updated_artists = component.update_func(frame, ax, initial_artists)

        # Basic update checks
        assert len(updated_artists) == len(initial_artists)
        assert all(
            isinstance(artist, plt.matplotlib.lines.Line2D)
            for artist in updated_artists
        )
        assert ax.get_legend() is not None

        plt.close(fig)

    @pytest.mark.parametrize("frame", [0, 1, 2])
    def test_alpha_transitions(self, sample_logs, frame):
        """Test specific alpha value visualization behavior."""
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
    """Tests for the kernel response visualization component."""

    def test_component_setup(self, sample_logs):
        """Test component creation and basic attributes."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        # Test required attributes exist
        assert callable(component.setup_func)
        assert callable(component.update_func)
        assert "gridspec" in component.subplot_params

        # Test subplot params are correct format
        assert isinstance(component.subplot_params["gridspec"], tuple)

    def test_initial_visualization(self, sample_logs):
        """Test that the kernel response component is properly configured."""

        # Create the visualization component
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        # Validate the component setup behavior
        # Ensure that the component is prepared to handle the required elements
        assert (
            "contourf" in component.setup_func.__code__.co_names
        ), "setup_func does not define contourf for surface plot"
        assert (
            "scatter" in component.setup_func.__code__.co_names
        ), "setup_func does not define scatter for points"
        assert (
            "colorbar" in component.setup_func.__code__.co_names
        ), "setup_func does not define colorbar addition"

    @pytest.mark.parametrize("frame", [0, 1, 2])
    def test_frame_updates(self, frame, sample_logs):
        """Test general frame update behavior."""

        # Initialize the visualizer and create the component
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        # Create figure and set up the initial visualization
        fig, ax = plt.subplots()
        initial_artists = component.setup_func(ax)

        # Verify the initial setup created the correct number of artists
        assert (
            len(initial_artists) >= 3
        ), "Expected at least 3 visual elements initially"

        # Perform the frame update
        updated_artists = component.update_func(frame, ax, initial_artists)

        # Verify the updated title reflects the current frame
        assert (
            ax.get_title() == f"Kernel Response Surface - Iteration {frame + 1}"
        ), "Frame title mismatch"

        # Verify the number of artists remains consistent after the update
        assert len(updated_artists) == len(
            initial_artists,
        ), "Number of artists should remain consistent after update"

        # Check the surface and contour updates
        updated_surface = updated_artists[0]
        assert isinstance(
            updated_surface,
            QuadContourSet,
        ), "Updated surface should be a QuadContourSet"
        assert updated_surface.collections, "Updated surface should have collections"

        # Extract positive and negative scatter points
        points_pos, points_neg = updated_artists[1], updated_artists[2]

        # Validate number of scatter points
        true_labels = sample_logs["true_labels"]
        positive_count = np.sum(true_labels == 1)
        negative_count = np.sum(true_labels == -1)

        assert len(points_pos.get_offsets()) == positive_count, (
            f"Mismatch in number of positive scatter points: "
            f"expected {positive_count}, got {len(points_pos.get_offsets())}"
        )
        assert len(points_neg.get_offsets()) == negative_count, (
            f"Mismatch in number of negative scatter points: "
            f"expected {negative_count}, got {len(points_neg.get_offsets())}"
        )

        # Validate colors reflect active/inactive states based on alphas
        alphas = sample_logs["alphas"][frame]["alphas"]

        # Active/inactive colors for positive points
        expected_positive_colors = [
            "red" if abs(alpha) > 1e-10 else "gray"
            for alpha in alphas[true_labels == 1]
        ]
        positive_colors = points_pos.get_facecolors()
        assert len(positive_colors) == positive_count, "Positive color count mismatch"
        for actual_color, expected_color in zip(
            positive_colors,
            expected_positive_colors,
        ):
            target_color = [1, 0, 0] if expected_color == "red" else [0.5, 0.5, 0.5]
            assert np.allclose(
                actual_color[:3],
                target_color,
                atol=1e-2,
            ), f"Positive scatter point color mismatch: expected {target_color}, got {actual_color[:3]}"

        # Active/inactive colors for negative points
        expected_negative_colors = [
            "red" if abs(alpha) > 1e-10 else "gray"
            for alpha in alphas[true_labels == -1]
        ]
        negative_colors = points_neg.get_facecolors()
        assert len(negative_colors) == negative_count, "Negative color count mismatch"
        for actual_color, expected_color in zip(
            negative_colors,
            expected_negative_colors,
        ):
            target_color = [1, 0, 0] if expected_color == "red" else [0.5, 0.5, 0.5]
            assert np.allclose(
                actual_color[:3],
                target_color,
                atol=1e-2,
            ), f"Negative scatter point color mismatch: expected {target_color}, got {actual_color[:3]}"

        plt.close(fig)

    # Kernel Response specific tests
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
        true_labels = sample_logs["true_labels"]
        positive_indices = np.where(true_labels == 1)[0]
        negative_indices = np.where(true_labels == -1)[0]

        for frame in range(len(sample_logs["alphas"])):
            updated_artists = component.update_func(frame, ax, artists)

            # Get scatter points for positive and negative labels
            points_pos, points_neg = updated_artists[1], updated_artists[2]

            # Extract active/inactive states from alphas
            alphas = sample_logs["alphas"][frame]["alphas"]
            active_positive = np.abs(alphas[positive_indices]) > 1e-10
            active_negative = np.abs(alphas[negative_indices]) > 1e-10

            # Get colors for positive and negative points
            positive_colors = points_pos.get_facecolor()
            negative_colors = points_neg.get_facecolor()

            # Ensure number of colors matches number of points
            assert len(positive_colors) == len(positive_indices), (
                f"Mismatch in number of positive scatter points: "
                f"expected {len(positive_indices)}, got {len(positive_colors)}"
            )
            assert len(negative_colors) == len(negative_indices), (
                f"Mismatch in number of negative scatter points: "
                f"expected {len(negative_indices)}, got {len(negative_colors)}"
            )

            # Validate colors based on active/inactive states
            for color, is_active in zip(positive_colors, active_positive):
                if is_active:
                    assert np.allclose(
                        color[:3],
                        [1, 0, 0],
                        atol=1e-2,
                    ), f"Expected red for active positive point, got {color[:3]}"
                else:
                    assert np.allclose(
                        color[:3],
                        [0.5, 0.5, 0.5],
                        atol=1e-2,
                    ), f"Expected gray for inactive positive point, got {color[:3]}"

            for color, is_active in zip(negative_colors, active_negative):
                if is_active:
                    assert np.allclose(
                        color[:3],
                        [1, 0, 0],
                        atol=1e-2,
                    ), f"Expected red for active negative point, got {color[:3]}"
                else:
                    assert np.allclose(
                        color[:3],
                        [0.5, 0.5, 0.5],
                        atol=1e-2,
                    ), f"Expected gray for inactive negative point, got {color[:3]}"

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
        assert len(contours.levels) >= 10, "Expected at least 10 contour levels"
        assert (
            0 <= contours.levels[0] <= 1
        ), "Expected lowest contour level to be in [0, 1]"
        assert (
            0 <= contours.levels[-1] <= 1
        ), "Expected highest contour level to be in [0, 1]"
        assert (
            0 in contours.levels
        ), "Expected decision boundary (level 0) to be present"

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
        atol = 1e-12  # Allow small tolerance for floating-point errors
        assert np.all(
            response_values >= -atol,
        ), "Response values contain unexpected negatives"
        assert np.all(response_values <= 1), "Response values exceed 1"

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
            "misclassification_count": sample_logs["misclassification_count"][:1],
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

        modified_logs["misclassification_count"] = [0] * n_frames

        visualizer = PerceptronVisualizer()
        component = visualizer.create_decision_boundary_component(modified_logs)
        visualizer.add_component(component)

        animation = visualizer.animate(modified_logs)

        assert animation is not None
        assert isinstance(animation, Animation)

        assert len(list(animation.new_frame_seq())) == visualizer.total_frames
        assert (
            visualizer.total_frames == n_frames
        ), f"Expected {n_frames} frames, but got {visualizer.total_frames}"

        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__])
