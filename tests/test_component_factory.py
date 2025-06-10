"""Tests for component factory pattern."""
import numpy as np
import pytest
from matplotlib import pyplot as plt

from kernel_viz.kernels.base import linear_kernel, rbf_gaussian_kernel
from kernel_viz.visualization.base import AnimationComponent
from kernel_viz.visualization.component_factory import (
    AlphaEvolutionFactory,
    ComponentFactory,
    DecisionBoundaryFactory,
    create_component,
)


class TestComponentFactory:
    """Test the abstract ComponentFactory base class."""
    
    def test_abstract_factory_cannot_be_instantiated(self):
        """Test that abstract factory cannot be instantiated directly."""
        logs = {"test": "data"}
        with pytest.raises(TypeError):
            ComponentFactory(logs)
    
    def test_factory_requires_abstract_methods(self):
        """Test that concrete factories must implement abstract methods."""
        class IncompleteFactory(ComponentFactory):
            def _extract_data(self):
                pass
            # Missing setup and update methods
        
        logs = {"test": "data"}
        with pytest.raises(TypeError):
            IncompleteFactory(logs)


class TestDecisionBoundaryFactory:
    """Test DecisionBoundaryFactory implementation."""
    
    @pytest.fixture
    def sample_logs(self):
        """Create sample logs for testing."""
        np.random.seed(42)
        xs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        ys = np.array([1, -1, -1, 1])
        
        # Create alpha history
        alphas_history = []
        for i in range(5):
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
    
    def test_factory_initialization(self, sample_logs):
        """Test factory initialization extracts data correctly."""
        factory = DecisionBoundaryFactory(sample_logs)
        
        assert factory.xs is not None
        assert factory.ys is not None
        assert factory.kernel == linear_kernel
        assert factory.plot_type == "line"  # linear kernel should use line plot
    
    def test_factory_with_rbf_kernel(self, sample_logs):
        """Test factory with RBF kernel defaults to contour plot."""
        sample_logs["kernel"] = rbf_gaussian_kernel
        sample_logs["kernel_params"] = {"sigma": 1.0}
        
        factory = DecisionBoundaryFactory(sample_logs)
        assert factory.plot_type == "contour"
    
    def test_factory_create_component(self, sample_logs):
        """Test factory creates AnimationComponent correctly."""
        factory = DecisionBoundaryFactory(sample_logs)
        component = factory.create()
        
        assert isinstance(component, AnimationComponent)
        assert component.setup_func == factory.setup
        assert component.update_func == factory.update
        assert component.subplot_params == {"gridspec": (0, 0)}
    
    def test_setup_creates_artists(self, sample_logs):
        """Test setup method creates appropriate artists."""
        factory = DecisionBoundaryFactory(sample_logs)
        
        fig, ax = plt.subplots()
        artists = factory.setup(ax)
        
        assert len(artists) == 2  # scatter + line for linear kernel
        assert ax.get_title() == "Decision Boundary - Iteration 1"
        assert ax.get_xlabel() == "Feature 1"
        assert ax.get_ylabel() == "Feature 2"
        
        plt.close(fig)
    
    def test_update_modifies_artists(self, sample_logs):
        """Test update method modifies artists correctly."""
        factory = DecisionBoundaryFactory(sample_logs)
        
        fig, ax = plt.subplots()
        artists = factory.setup(ax)
        
        # Update to frame 1
        updated_artists = factory.update(1, ax, artists)
        
        assert len(updated_artists) == len(artists)
        plt.close(fig)
    
    def test_custom_subplot_params(self, sample_logs):
        """Test custom subplot params are passed through."""
        factory = DecisionBoundaryFactory(sample_logs)
        component = factory.create(
            subplot_params={"gridspec": (1, 1)},
            name="CustomDecisionBoundary"
        )
        
        assert component.subplot_params == {"gridspec": (1, 1)}
        assert component.name == "CustomDecisionBoundary"


class TestAlphaEvolutionFactory:
    """Test AlphaEvolutionFactory implementation."""
    
    @pytest.fixture
    def sample_logs(self):
        """Create sample logs for testing."""
        # Create alpha history with 4 samples, 5 iterations
        alphas_history = []
        for i in range(5):
            alphas = np.array([0.1 * i, -0.05 * i, 0.0, 0.15 * i])
            alphas_history.append({"iteration": i, "alphas": alphas})
        
        return {"alphas": alphas_history}
    
    def test_factory_initialization(self, sample_logs):
        """Test factory initialization extracts data correctly."""
        factory = AlphaEvolutionFactory(sample_logs)
        
        assert factory.n_samples == 4
        assert factory.all_alphas.shape == (5, 4)
        assert factory.y_min < 0  # Should include negative alphas
        assert factory.y_max > 0  # Should include positive alphas
    
    def test_setup_creates_lines(self, sample_logs):
        """Test setup creates one line per sample."""
        factory = AlphaEvolutionFactory(sample_logs)
        
        fig, ax = plt.subplots()
        artists = factory.setup(ax)
        
        assert len(artists) == 4  # One line per sample
        assert ax.get_title() == "Alpha Values Evolution"
        assert ax.get_xlabel() == "Training Iteration"
        assert ax.get_ylabel() == "Alpha Value"
        
        plt.close(fig)
    
    def test_update_changes_line_properties(self, sample_logs):
        """Test update changes line colors based on alpha values."""
        factory = AlphaEvolutionFactory(sample_logs)
        
        fig, ax = plt.subplots()
        artists = factory.setup(ax)
        
        # Update to frame 2
        factory.update(2, ax, artists)
        
        # Check that lines with non-zero alphas are red
        assert artists[0].get_color() == "red"  # alpha = 0.2
        assert artists[1].get_color() == "red"  # alpha = -0.1
        assert artists[2].get_color() == "gray"  # alpha = 0.0
        assert artists[3].get_color() == "red"  # alpha = 0.3
        
        plt.close(fig)
    
    def test_debug_mode(self, sample_logs):
        """Test debug mode initialization."""
        factory = AlphaEvolutionFactory(sample_logs, debug_mode=True)
        assert factory.debug_mode is True


class TestCreateComponent:
    """Test the factory method create_component."""
    
    @pytest.fixture
    def sample_logs(self):
        """Create minimal sample logs."""
        xs = np.array([[0, 0], [1, 1]])
        ys = np.array([1, -1])
        alphas_history = [{"iteration": 0, "alphas": np.array([0.1, -0.1])}]
        
        return {
            "feature_space": xs,
            "true_labels": ys,
            "kernel": linear_kernel,
            "kernel_params": {},
            "alphas": alphas_history,
        }
    
    def test_create_decision_boundary_component(self, sample_logs):
        """Test creating decision boundary component."""
        component = create_component("decision_boundary", sample_logs)
        
        assert isinstance(component, AnimationComponent)
        assert component.setup_func is not None
        assert component.update_func is not None
    
    def test_create_alpha_evolution_component(self, sample_logs):
        """Test creating alpha evolution component."""
        component = create_component("alpha_evolution", sample_logs)
        
        assert isinstance(component, AnimationComponent)
        assert component.setup_func is not None
        assert component.update_func is not None
    
    def test_unknown_component_type_raises_error(self, sample_logs):
        """Test unknown component type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown component type"):
            create_component("unknown_type", sample_logs)
    
    def test_kwargs_passed_to_factory(self, sample_logs):
        """Test kwargs are passed to factory constructor."""
        component = create_component(
            "decision_boundary", 
            sample_logs,
            plot_type="contour",
            fixed_dims={2: 0.0}
        )
        
        # Create factory to check if kwargs were used
        factory = DecisionBoundaryFactory(
            sample_logs, 
            plot_type="contour",
            fixed_dims={2: 0.0}
        )
        assert factory.plot_type == "contour"
        assert factory.fixed_dims == {2: 0.0}