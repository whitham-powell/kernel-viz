"""Test integration of factory pattern with existing code."""

import numpy as np
import pytest

from kernel_viz.kernels.base import linear_kernel
from kernel_viz.visualization import PerceptronVisualizer
from kernel_viz.visualization.base import AnimationComponent
from kernel_viz.visualization.component_adapter import (
    create_alpha_evolution_component_v2,
    create_decision_boundary_component_v2,
)


class TestPatternIntegration:
    """Test that factory pattern produces same results as existing implementation."""

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

    def test_decision_boundary_compatibility(self, sample_logs):
        """Test factory produces compatible decision boundary component."""
        visualizer = PerceptronVisualizer()

        # Create component using existing method
        component_v1 = visualizer.create_decision_boundary_component(sample_logs)

        # Create component using factory adapter
        component_v2 = create_decision_boundary_component_v2(visualizer, sample_logs)

        # Both should be AnimationComponent instances
        assert isinstance(component_v1, AnimationComponent)
        assert isinstance(component_v2, AnimationComponent)

        # Both should have setup and update functions
        assert component_v1.setup_func is not None
        assert component_v1.update_func is not None
        assert component_v2.setup_func is not None
        assert component_v2.update_func is not None

        # Test that both produce similar results
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Setup both components
        artists_v1 = component_v1.setup_func(ax1)
        artists_v2 = component_v2.setup_func(ax2)

        # Both should create same number of artists
        assert len(artists_v1) == len(artists_v2)

        # Update both to same frame
        updated_v1 = component_v1.update_func(0, ax1, artists_v1)
        updated_v2 = component_v2.update_func(0, ax2, artists_v2)

        assert len(updated_v1) == len(updated_v2)

        plt.close(fig)

    def test_alpha_evolution_compatibility(self, sample_logs):
        """Test factory produces compatible alpha evolution component."""
        visualizer = PerceptronVisualizer()
        visualizer.set_debug_mode(True)

        # Create component using existing method
        component_v1 = visualizer.create_alpha_evolution_component(sample_logs)

        # Create component using factory adapter
        component_v2 = create_alpha_evolution_component_v2(visualizer, sample_logs)

        # Both should be AnimationComponent instances
        assert isinstance(component_v1, AnimationComponent)
        assert isinstance(component_v2, AnimationComponent)

        # Test setup produces same number of lines
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        artists_v1 = component_v1.setup_func(ax1)
        artists_v2 = component_v2.setup_func(ax2)

        # Should have same number of lines (one per alpha)
        assert len(artists_v1) == len(artists_v2) == 4

        plt.close(fig)

    def test_factory_can_be_extended(self, sample_logs):
        """Test that factory pattern allows for easy extension."""
        from kernel_viz.visualization.component_factory import ComponentFactory

        # Define a custom factory
        class CustomFactory(ComponentFactory):
            def _extract_data(self):
                self.data = self.logs.get("custom_data", "default")

            def setup(self, ax):
                ax.text(0.5, 0.5, f"Custom: {self.data}")
                return []

            def update(self, frame, ax, artists):
                return artists

        # Create custom component
        custom_logs = {"custom_data": "test value"}
        factory = CustomFactory(custom_logs)
        component = factory.create(name="CustomComponent")

        assert component.name == "CustomComponent"
        assert component.setup_func is not None

        # Test it works
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        factory.setup(ax)

        # Check that text was added
        assert len(ax.texts) == 1
        assert "Custom: test value" in ax.texts[0].get_text()

        plt.close(fig)

    def test_direct_factory_usage(self, sample_logs):
        """Test using factories directly without adapter."""
        from kernel_viz.visualization.component_factory import DecisionBoundaryFactory

        # Direct factory usage
        factory = DecisionBoundaryFactory(
            sample_logs,
            plot_type="contour",  # Force contour even for linear kernel
        )
        component = factory.create()

        # Verify configuration was applied
        assert factory.plot_type == "contour"
        assert isinstance(component, AnimationComponent)

    def test_factory_with_custom_params(self, sample_logs):
        """Test factory with custom parameters."""
        from kernel_viz.visualization.component_factory import create_component

        # Create with custom subplot params
        component = create_component(
            "decision_boundary",
            sample_logs,
            plot_type="line",
            fixed_dims={2: 0.0, 3: 1.0},
        )

        # Component should be created successfully
        assert isinstance(component, AnimationComponent)
