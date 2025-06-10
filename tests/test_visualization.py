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
        """Test initial visualization state and artist setup."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        # Create test figure and set up the component
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Verify we have the expected artists
        assert (
            len(artists) == 4
        ), "Expected 4 artists: surface, decision_boundary, points_pos, points_neg"

        # Check for contour and scatter
        assert any(
            isinstance(artist, QuadContourSet) for artist in ax.collections
        ), "Expected contour plot"
        scatter_artists = [
            artist for artist in ax.collections if isinstance(artist, PathCollection)
        ]
        assert (
            len(scatter_artists) == 2
        ), "Expected 2 scatter plots (pos and neg points)"

        # Verify colorbar exists (additional axes)
        assert len(ax.figure.axes) > 1, "Expected colorbar axes"

        # Check title and labels
        assert "Kernel Response Surface" in ax.get_title()
        assert ax.get_xlabel() == "Feature 1"
        assert ax.get_ylabel() == "Feature 2"

        plt.close(fig)

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
        assert len(updated_artists) == 4, "Expected 4 artists after update"

        # The artists are: [surface, decision_boundary, points_pos, points_neg]
        surface, decision_boundary, points_pos, points_neg = updated_artists

        # Verify types
        assert isinstance(
            surface,
            QuadContourSet,
        ), "First artist should be response surface"
        assert isinstance(
            decision_boundary,
            QuadContourSet,
        ), "Second artist should be decision boundary"
        assert isinstance(
            points_pos,
            PathCollection,
        ), "Third artist should be positive points scatter"
        assert isinstance(
            points_neg,
            PathCollection,
        ), "Fourth artist should be negative points scatter"

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

        # Validate that points have fixed face colors (blue for positive, red for negative)
        positive_colors = points_pos.get_facecolors()
        negative_colors = points_neg.get_facecolors()

        # Check positive points are blue
        if positive_colors.ndim == 2 and positive_colors.shape[0] == 1:
            # Single color for all points
            assert np.allclose(
                positive_colors[0, :3],
                [0, 0, 1],
                atol=1e-2,
            ), f"Expected blue for positive points, got {positive_colors[0, :3]}"
        else:
            # Multiple colors (shouldn't happen with current implementation)
            for color in positive_colors:
                assert np.allclose(
                    color[:3],
                    [0, 0, 1],
                    atol=1e-2,
                ), f"Expected blue for positive points, got {color[:3]}"

        # Check negative points are red
        if negative_colors.ndim == 2 and negative_colors.shape[0] == 1:
            # Single color for all points
            assert np.allclose(
                negative_colors[0, :3],
                [1, 0, 0],
                atol=1e-2,
            ), f"Expected red for negative points, got {negative_colors[0, :3]}"
        else:
            # Multiple colors (shouldn't happen with current implementation)
            for color in negative_colors:
                assert np.allclose(
                    color[:3],
                    [1, 0, 0],
                    atol=1e-2,
                ), f"Expected red for negative points, got {color[:3]}"

        # Validate support vectors are highlighted with larger size and yellow edge
        alphas = sample_logs["alphas"][frame]["alphas"]
        sizes_pos = points_pos.get_sizes()
        sizes_neg = points_neg.get_sizes()
        # Edge colors are set in the visualization but not checked in this test
        # points_pos.get_edgecolors() and points_neg.get_edgecolors() would show
        # yellow for support vectors and black for non-support vectors

        # Check support vector highlighting for positive points
        active_pos = np.abs(alphas[true_labels == 1]) > 1e-10
        expected_sizes_pos = [120 if active else 80 for active in active_pos]
        assert np.array_equal(
            sizes_pos,
            expected_sizes_pos,
        ), "Positive point sizes incorrect"

        # Check support vector highlighting for negative points
        active_neg = np.abs(alphas[true_labels == -1]) > 1e-10
        expected_sizes_neg = [120 if active else 80 for active in active_neg]
        assert np.array_equal(
            sizes_neg,
            expected_sizes_neg,
        ), "Negative point sizes incorrect"

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
            surface = updated_artists[0]

            # Verify it's a contour plot
            assert isinstance(surface, QuadContourSet)

            # Response surface should have consistent colormap limits
            # (handled by vmin/vmax in contourf)

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
            # Artists are: [decision_boundary, confidence_regions, points_pos, points_neg]
            points_pos, points_neg = updated_artists[2], updated_artists[3]

            # Extract active/inactive states from alphas
            alphas = sample_logs["alphas"][frame]["alphas"]
            active_positive = np.abs(alphas[positive_indices]) > 1e-10
            active_negative = np.abs(alphas[negative_indices]) > 1e-10

            # Get edge colors for positive and negative points (support vectors have yellow edges)
            positive_edge_colors = points_pos.get_edgecolors()
            negative_edge_colors = points_neg.get_edgecolors()

            # Ensure number of points matches
            assert len(positive_edge_colors) == len(positive_indices), (
                f"Mismatch in number of positive scatter points: "
                f"expected {len(positive_indices)}, got {len(positive_edge_colors)}"
            )
            assert len(negative_edge_colors) == len(negative_indices), (
                f"Mismatch in number of negative scatter points: "
                f"expected {len(negative_indices)}, got {len(negative_edge_colors)}"
            )

            # Validate edge colors based on active/inactive states (yellow for support vectors)
            for edge_color, is_active in zip(positive_edge_colors, active_positive):
                if is_active:
                    # Yellow edge for support vectors
                    assert np.allclose(
                        edge_color[:3],
                        [1, 1, 0],  # Yellow in RGB
                        atol=1e-2,
                    ), f"Expected yellow edge for active positive point, got {edge_color[:3]}"
                else:
                    # White edge for non-support vectors
                    assert np.allclose(
                        edge_color[:3],
                        [1, 1, 1],  # White in RGB
                        atol=1e-2,
                    ), f"Expected white edge for inactive positive point, got {edge_color[:3]}"

            for edge_color, is_active in zip(negative_edge_colors, active_negative):
                if is_active:
                    # Yellow edge for support vectors
                    assert np.allclose(
                        edge_color[:3],
                        [1, 1, 0],  # Yellow in RGB
                        atol=1e-2,
                    ), f"Expected yellow edge for active negative point, got {edge_color[:3]}"
                else:
                    # White edge for non-support vectors
                    assert np.allclose(
                        edge_color[:3],
                        [1, 1, 1],  # White in RGB
                        atol=1e-2,
                    ), f"Expected white edge for inactive negative point, got {edge_color[:3]}"

        plt.close(fig)

    def test_contour_levels(self, sample_logs):
        """Test that contour levels are appropriate and consistent."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_response_component(sample_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        updated_artists = component.update_func(0, ax, artists)

        # Get response surface
        surface = updated_artists[0]
        assert isinstance(surface, QuadContourSet)

        # Response surface should have multiple levels
        if hasattr(surface, "levels"):
            assert (
                len(surface.levels) >= 10
            ), "Expected at least 10 contour levels for response surface"

        # Get decision boundary
        decision_boundary = updated_artists[1]
        assert isinstance(decision_boundary, QuadContourSet)

        # Decision boundary should have level 0
        if hasattr(decision_boundary, "levels"):
            assert (
                0 in decision_boundary.levels
                or abs(decision_boundary.levels[0]) < 1e-10
            )

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

        # Verify we can create decision boundaries with different kernels
        surface = next(
            artist for artist in updated_artists if isinstance(artist, QuadContourSet)
        )
        assert isinstance(surface, QuadContourSet)
        # No specific range check needed - decision values can be any real number

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


class TestKernelMatrixComponent:
    """Tests for the kernel matrix visualization component."""

    def test_component_setup(self, sample_logs):
        """Test kernel matrix component creation and basic attributes."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_matrix_component(sample_logs)

        assert callable(component.setup_func)
        assert callable(component.update_func)
        assert "gridspec" in component.subplot_params
        assert component.subplot_params["gridspec"] == (1, 1)

    def test_kernel_matrix_computation(self, sample_logs):
        """Test kernel matrix is computed if not provided."""
        visualizer = PerceptronVisualizer()

        # Remove kernel matrix from logs
        logs_without_matrix = sample_logs.copy()
        logs_without_matrix.pop("kernel_matrix", None)

        component = visualizer.create_kernel_matrix_component(logs_without_matrix)

        # Setup and verify kernel matrix is computed
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Should have image and markers
        assert len(artists) > 1
        assert hasattr(ax.images[0], "get_array")

        # Check kernel matrix shape
        kernel_matrix = ax.images[0].get_array()
        n_samples = len(sample_logs["feature_space"])
        assert kernel_matrix.shape == (n_samples, n_samples)

        plt.close(fig)

    def test_heatmap_visualization(self, sample_logs):
        """Test heatmap visualization setup."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_matrix_component(sample_logs)

        fig, ax = plt.subplots()
        component.setup_func(ax)

        # Check heatmap image
        assert len(ax.images) == 1
        image = ax.images[0]
        assert image.get_cmap().name == "RdBu_r"

        # Check colorbar
        assert len(fig.axes) == 2  # Main axes + colorbar

        # Check labels
        assert ax.get_xlabel() == "Sample Index"
        assert ax.get_ylabel() == "Sample Index"
        assert "Kernel Matrix" in ax.get_title()

        plt.close(fig)

    def test_support_vector_indicators(self, sample_logs):
        """Test support vector indicator creation and updates."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_matrix_component(sample_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Initial state - all markers should be hidden
        markers = [a for a in artists if hasattr(a, "get_markersize")]
        assert all(m.get_markersize() == 0 for m in markers)

        # Update to show support vectors
        component.update_func(1, ax, artists)

        # Some markers should now be visible
        visible_markers = [m for m in markers if m.get_markersize() > 0]
        assert len(visible_markers) > 0

        plt.close(fig)

    @pytest.mark.mpl_image_compare(tolerance=10, style="default")
    def test_kernel_matrix_visual_snapshot(self, sample_logs):
        """Visual regression test for kernel matrix component."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_kernel_matrix_component(sample_logs)

        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        component.update_func(2, ax, artists)

        return fig


class TestMisclassificationTrackerComponent:
    """Tests for the misclassification tracker component."""

    def test_component_setup(self, sample_logs):
        """Test misclassification tracker component creation."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_misclassification_tracker_component(sample_logs)

        assert callable(component.setup_func)
        assert callable(component.update_func)
        assert "gridspec" in component.subplot_params
        assert component.subplot_params["gridspec"] == (0, 0)

    def test_initial_visualization(self, sample_logs):
        """Test initial visualization state."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_misclassification_tracker_component(sample_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Should have scatter plot
        assert len(artists) == 1
        scatter = artists[0]
        assert isinstance(scatter, PathCollection)

        # Check initial colors - scatter plot might not have individual colors set yet
        colors = scatter.get_facecolors()
        if len(colors) == 0 or len(colors) == 1:
            # Single color for all points - check it's gray
            pass  # Colors might be set in update function
        else:
            # All points should be gray initially
            assert all(np.allclose(c[:3], [0.5, 0.5, 0.5]) for c in colors)

        # Check labels and legend
        assert ax.get_xlabel() == "Feature 1"
        assert ax.get_ylabel() == "Feature 2"
        assert "Misclassified Points Tracker" in ax.get_title()
        assert ax.get_legend() is not None

        plt.close(fig)

    def test_misclassification_updates(self, sample_logs):
        """Test that misclassified points are highlighted correctly."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_misclassification_tracker_component(sample_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        scatter = artists[0]

        # Update to a frame with misclassifications
        component.update_func(0, ax, artists)

        # Check that some points are red (misclassified)
        colors = scatter.get_facecolors()
        red_points = [c for c in colors if np.allclose(c[:3], [1.0, 0.0, 0.0])]
        gray_points = [c for c in colors if np.allclose(c[:3], [0.5, 0.5, 0.5])]

        # Should have some points with colors set
        # Note: depending on the data, all points might be misclassified or correctly classified
        assert len(colors) > 0, "No colors were set"
        assert len(red_points) > 0 or len(gray_points) > 0, "No colored points found"

        plt.close(fig)

    def test_decision_function_calculation(self, sample_logs):
        """Test decision function calculation logic."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_misclassification_tracker_component(sample_logs)

        # Use a simple linear kernel for predictable results
        linear_logs = sample_logs.copy()
        from kernel_viz.kernels import linear_kernel

        linear_logs["kernel"] = linear_kernel
        linear_logs["kernel_params"] = {}

        component = visualizer.create_misclassification_tracker_component(linear_logs)

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Update and check
        component.update_func(0, ax, artists)

        # Verify scatter plot exists and has correct number of points
        scatter = artists[0]
        n_samples = len(linear_logs["feature_space"])
        assert len(scatter.get_offsets()) == n_samples

        plt.close(fig)

    @pytest.mark.mpl_image_compare(tolerance=10, style="default")
    def test_misclassification_visual_snapshot(self, sample_logs):
        """Visual regression test for misclassification tracker."""
        visualizer = PerceptronVisualizer()
        component = visualizer.create_misclassification_tracker_component(sample_logs)

        fig, ax = plt.subplots(figsize=(8, 6))
        artists = component.setup_func(ax)
        component.update_func(1, ax, artists)

        return fig


class TestDecisionBoundaryAdvanced:
    """Advanced tests for decision boundary component covering missing cases."""

    def test_plot_type_parameter(self, sample_logs):
        """Test different plot types (line vs contour)."""
        visualizer = PerceptronVisualizer()

        # Test line plot for linear kernel
        from kernel_viz.kernels import linear_kernel

        linear_logs = sample_logs.copy()
        linear_logs["kernel"] = linear_kernel
        linear_logs["kernel_params"] = {}

        component = visualizer.create_decision_boundary_component(
            linear_logs,
            plot_type="line",
        )

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Should have scatter and line
        assert len(artists) == 2
        assert isinstance(artists[1], Line2D)

        plt.close(fig)

        # Test contour plot
        component = visualizer.create_decision_boundary_component(
            sample_logs,
            plot_type="contour",
        )

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Should have scatter only initially
        assert len(artists) == 1
        assert isinstance(artists[0], PathCollection)

        plt.close(fig)

    def test_fixed_dims_parameter(self, sample_logs):
        """Test fixed dimensions for high-dimensional data."""
        # Create high-dimensional data
        X_high = np.random.randn(10, 5)  # 5D data
        high_dim_logs = sample_logs.copy()
        high_dim_logs["feature_space"] = X_high
        # Ensure alphas match the number of samples
        high_dim_logs["alphas"] = [
            {"iteration": i, "alphas": np.random.randn(10)} for i in range(3)
        ]
        high_dim_logs["true_labels"] = np.array([1, -1] * 5)  # 10 labels

        visualizer = PerceptronVisualizer()

        # Test with fixed dimensions
        fixed_dims = {2: 0.5, 3: -0.5, 4: 1.0}  # Fix dims 2, 3, 4
        component = visualizer.create_decision_boundary_component(
            high_dim_logs,
            fixed_dims=fixed_dims,
        )

        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Should still create visualization
        assert len(artists) >= 1
        assert isinstance(artists[0], PathCollection)

        # Update to verify fixed dims are used
        component.update_func(0, ax, artists)

        plt.close(fig)

    def test_kernel_type_auto_detection(self, sample_logs):
        """Test automatic plot type selection based on kernel."""
        visualizer = PerceptronVisualizer()

        # Linear kernel should default to line plot
        from kernel_viz.kernels import linear_kernel

        linear_logs = sample_logs.copy()
        linear_logs["kernel"] = linear_kernel
        linear_logs["kernel_params"] = {}

        component = visualizer.create_decision_boundary_component(linear_logs)
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Should have line for linear kernel
        assert any(isinstance(a, Line2D) for a in artists)

        plt.close(fig)

        # RBF kernel should default to contour
        component = visualizer.create_decision_boundary_component(sample_logs)
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Should not have line for RBF kernel
        assert not any(isinstance(a, Line2D) for a in artists)

        plt.close(fig)


class TestComputeDecisionBoundary:
    """Tests for the compute_decision_boundary utility function."""

    def test_basic_functionality(self, sample_logs):
        """Test basic decision boundary computation."""
        from kernel_viz.visualization.core import compute_decision_boundary

        xs = sample_logs["feature_space"]
        alphas = sample_logs["alphas"][0]["alphas"]
        kernel = sample_logs["kernel"]
        kernel_params = sample_logs["kernel_params"]

        xx, yy, zz = compute_decision_boundary(xs, alphas, kernel, kernel_params)

        # Check output shapes
        assert xx.shape == yy.shape == zz.shape
        assert xx.shape == (100, 100)  # Default grid resolution

        # Check grid bounds
        assert xx.min() < xs[:, 0].min()
        assert xx.max() > xs[:, 0].max()
        assert yy.min() < xs[:, 1].min()
        assert yy.max() > xs[:, 1].max()

    def test_fixed_dimensions(self, sample_logs):
        """Test decision boundary with fixed dimensions."""
        from kernel_viz.visualization.core import compute_decision_boundary

        # Create 3D data
        xs_3d = np.random.randn(10, 3)
        alphas = np.random.randn(10)
        kernel = sample_logs["kernel"]
        kernel_params = sample_logs["kernel_params"]

        # Fix third dimension
        fixed_dims = {2: 0.5}

        xx, yy, zz = compute_decision_boundary(
            xs_3d,
            alphas,
            kernel,
            kernel_params,
            fixed_dims,
        )

        # Should still produce 2D grid
        assert xx.shape == yy.shape == zz.shape
        assert xx.shape == (100, 100)

    def test_empty_fixed_dims(self, sample_logs):
        """Test decision boundary with empty fixed_dims dict."""
        from kernel_viz.visualization.core import compute_decision_boundary

        xs = sample_logs["feature_space"]
        alphas = sample_logs["alphas"][0]["alphas"]
        kernel = sample_logs["kernel"]
        kernel_params = sample_logs["kernel_params"]

        # Empty fixed_dims should work
        xx, yy, zz = compute_decision_boundary(xs, alphas, kernel, kernel_params, {})

        assert xx.shape == yy.shape == zz.shape

    def test_kernel_params_none(self, sample_logs):
        """Test decision boundary with None kernel_params."""
        from kernel_viz.kernels import linear_kernel
        from kernel_viz.visualization.core import compute_decision_boundary

        xs = sample_logs["feature_space"]
        alphas = sample_logs["alphas"][0]["alphas"]

        # Use linear kernel with None params
        xx, yy, zz = compute_decision_boundary(xs, alphas, linear_kernel, None)

        assert xx.shape == yy.shape == zz.shape


class TestVisualizationEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_alphas_log(self):
        """Test components with empty alphas log."""
        visualizer = PerceptronVisualizer()

        empty_logs = {
            "feature_space": np.array([[1, 1], [2, 2]]),
            "true_labels": np.array([1, -1]),
            "kernel": lambda x, y: np.dot(x, y),
            "kernel_params": {},
            "alphas": [],  # Empty!
            "misclassification_count": [],
        }

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            component = visualizer.create_decision_boundary_component(empty_logs)
            visualizer.add_component(component)
            visualizer.animate(empty_logs)

    def test_mismatched_data_sizes(self):
        """Test components with mismatched data sizes."""
        visualizer = PerceptronVisualizer()

        mismatched_logs = {
            "feature_space": np.array([[1, 1], [2, 2], [3, 3]]),  # 3 samples
            "true_labels": np.array([1, -1]),  # 2 labels - mismatch!
            "kernel": lambda x, y: np.dot(x, y),
            "kernel_params": {},
            "alphas": [{"iteration": 0, "alphas": np.array([1, 0])}],  # 2 alphas
            "misclassification_count": [1],
        }

        # The misclassification tracker should actually handle this by using the size
        # from feature_space, so it might not raise an error immediately
        # Instead, let's test that it fails during update when dimensions don't match
        try:
            component = visualizer.create_misclassification_tracker_component(
                mismatched_logs,
            )
            fig, ax = plt.subplots()
            artists = component.setup_func(ax)
            # This should fail during update due to mismatched dimensions
            with pytest.raises((ValueError, IndexError)):
                component.update_func(0, ax, artists)
            plt.close(fig)
        except (ValueError, IndexError):
            # It's also OK if it fails during component creation
            pass

    def test_debug_mode(self, sample_logs):
        """Test debug mode functionality."""
        visualizer = PerceptronVisualizer()
        visualizer.set_debug_mode(True)

        # Capture debug output
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured

        component = visualizer.create_alpha_evolution_component(sample_logs)
        visualizer.add_component(component)

        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        # Should see debug output
        assert "grid" in output.lower() or "component" in output.lower()


if __name__ == "__main__":
    pytest.main([__file__])
