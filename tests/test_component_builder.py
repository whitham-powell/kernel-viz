"""Tests for component builder pattern."""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from kernel_viz.kernels.base import linear_kernel, rbf_gaussian_kernel
from kernel_viz.visualization.base import AnimationComponent
from kernel_viz.visualization.component_builder import (
    AlphaEvolutionBuilder,
    ComponentBuilder,
    DecisionBoundaryBuilder,
)


class TestComponentBuilder:
    """Test the base ComponentBuilder class."""

    def test_builder_initialization(self):
        """Test builder starts with empty state."""
        builder = ComponentBuilder()

        assert builder._logs is None
        assert builder._setup_func is None
        assert builder._update_func is None
        assert builder._subplot_params == {"gridspec": (0, 0)}
        assert builder._name is None

    def test_builder_method_chaining(self):
        """Test builder methods return self for chaining."""
        builder = ComponentBuilder()
        logs = {"test": "data"}

        result = builder.with_logs(logs).with_name("TestComponent")

        assert result is builder
        assert builder._logs == logs
        assert builder._name == "TestComponent"

    def test_builder_reset(self):
        """Test reset clears all builder state."""
        builder = ComponentBuilder()
        builder.with_logs({"test": "data"}).with_name("Test")

        builder.reset()

        assert builder._logs is None
        assert builder._name is None

    def test_build_requires_functions(self):
        """Test build fails without setup and update functions."""
        builder = ComponentBuilder()

        with pytest.raises(ValueError, match="Setup and update functions are required"):
            builder.build()

    def test_build_creates_component(self):
        """Test build creates AnimationComponent correctly."""
        builder = ComponentBuilder()

        def dummy_setup(ax):
            return []

        def dummy_update(frame, ax, artists):
            return artists

        component = (
            builder.with_setup(dummy_setup)
            .with_update(dummy_update)
            .with_subplot_params(gridspec=(1, 1))
            .with_name("TestComponent")
            .build()
        )

        assert isinstance(component, AnimationComponent)
        assert component.setup_func == dummy_setup
        assert component.update_func == dummy_update
        assert component.subplot_params == {"gridspec": (1, 1)}
        assert component.name == "TestComponent"


class TestDecisionBoundaryBuilder:
    """Test DecisionBoundaryBuilder implementation."""

    @pytest.fixture
    def sample_logs(self):
        """Create sample logs for testing."""
        xs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        ys = np.array([1, -1, -1, 1])

        alphas_history = []
        for i in range(3):
            alphas = np.zeros(4)
            alphas[i % 4] = 0.1 * (i + 1)
            alphas_history.append({"iteration": i, "alphas": alphas})

        return {
            "feature_space": xs,
            "true_labels": ys,
            "kernel": linear_kernel,
            "kernel_params": {},
            "alphas": alphas_history,
        }

    def test_builder_defaults(self):
        """Test builder has correct defaults."""
        builder = DecisionBoundaryBuilder()

        assert builder._plot_type == "contour"
        assert builder._fixed_dims is None

    def test_with_plot_type(self):
        """Test setting plot type."""
        builder = DecisionBoundaryBuilder()

        builder.with_plot_type("line")
        assert builder._plot_type == "line"

        with pytest.raises(ValueError, match="Plot type must be"):
            builder.with_plot_type("invalid")

    def test_with_fixed_dims(self):
        """Test setting fixed dimensions."""
        builder = DecisionBoundaryBuilder()
        fixed_dims = {2: 0.0, 3: 1.0}

        builder.with_fixed_dims(fixed_dims)
        assert builder._fixed_dims == fixed_dims

    def test_auto_configure_requires_logs(self):
        """Test auto_configure requires logs to be set."""
        builder = DecisionBoundaryBuilder()

        with pytest.raises(ValueError, match="Logs must be set"):
            builder.auto_configure()

    def test_auto_configure_detects_linear_kernel(self, sample_logs):
        """Test auto_configure detects linear kernel and sets line plot."""
        builder = DecisionBoundaryBuilder()

        builder.with_logs(sample_logs).auto_configure()

        assert builder._plot_type == "line"
        assert builder._setup_func is not None
        assert builder._update_func is not None

    def test_auto_configure_keeps_contour_for_rbf(self, sample_logs):
        """Test auto_configure keeps contour plot for RBF kernel."""
        sample_logs["kernel"] = rbf_gaussian_kernel
        builder = DecisionBoundaryBuilder()

        builder.with_logs(sample_logs).auto_configure()

        assert builder._plot_type == "contour"

    def test_build_complete_component(self, sample_logs):
        """Test building complete decision boundary component."""
        component = (
            DecisionBoundaryBuilder()
            .with_logs(sample_logs)
            .with_subplot_params(gridspec=(0, 1))
            .with_name("DecisionBoundary")
            .auto_configure()
            .build()
        )

        assert isinstance(component, AnimationComponent)
        assert component.name == "DecisionBoundary"

        # Test that setup and update work
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        assert len(artists) == 2  # scatter + line for linear kernel

        updated = component.update_func(0, ax, artists)
        assert len(updated) == 2

        plt.close(fig)


class TestAlphaEvolutionBuilder:
    """Test AlphaEvolutionBuilder implementation."""

    @pytest.fixture
    def sample_logs(self):
        """Create sample logs for testing."""
        alphas_history = []
        for i in range(5):
            alphas = np.array([0.1 * i, -0.05 * i, 0.0, 0.2 * i])
            alphas_history.append({"iteration": i, "alphas": alphas})

        return {"alphas": alphas_history}

    def test_builder_defaults(self):
        """Test builder has correct defaults."""
        builder = AlphaEvolutionBuilder()

        assert builder._show_inactive is True
        assert builder._color_active == "red"
        assert builder._color_inactive == "gray"

    def test_color_customization(self):
        """Test setting custom colors."""
        builder = AlphaEvolutionBuilder()

        builder.with_active_color("blue").with_inactive_color("lightgray")

        assert builder._color_active == "blue"
        assert builder._color_inactive == "lightgray"

    def test_hide_inactive(self):
        """Test hiding inactive alpha values."""
        builder = AlphaEvolutionBuilder()

        builder.hide_inactive()
        assert builder._show_inactive is False

    def test_auto_configure(self, sample_logs):
        """Test auto configuration creates functions."""
        builder = AlphaEvolutionBuilder()

        builder.with_logs(sample_logs).auto_configure()

        assert builder._setup_func is not None
        assert builder._update_func is not None

    def test_build_complete_component(self, sample_logs):
        """Test building complete alpha evolution component."""
        component = (
            AlphaEvolutionBuilder()
            .with_logs(sample_logs)
            .with_active_color("green")
            .with_inactive_color("lightblue")
            .with_name("AlphaEvolution")
            .auto_configure()
            .build()
        )

        assert isinstance(component, AnimationComponent)
        assert component.name == "AlphaEvolution"

        # Test that setup and update work
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        assert len(artists) == 4  # One line per alpha

        updated = component.update_func(2, ax, artists)
        assert len(updated) == 4

        # Check that active lines have the custom color
        assert artists[0].get_color() == "green"  # Active (0.2)
        assert artists[2].get_color() == "lightblue"  # Inactive (0.0)

        plt.close(fig)

    def test_fluent_interface_example(self, sample_logs):
        """Test complete fluent interface usage."""
        component = (
            AlphaEvolutionBuilder()
            .with_logs(sample_logs)
            .hide_inactive()
            .with_active_color("purple")
            .with_subplot_params(gridspec=(1, 0))
            .auto_configure()
            .build()
        )

        assert isinstance(component, AnimationComponent)

        # Verify the hide_inactive worked
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)

        # Inactive lines should have alpha 0.0
        component.update_func(1, ax, artists)
        assert artists[2].get_alpha() == 0.0  # Inactive line

        plt.close(fig)
