"""Comprehensive equivalence tests between factory and original implementations."""

import hashlib
import pickle
from typing import List, Tuple

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.artist import Artist

from kernel_viz.algorithms.perceptron import PerceptronLogger, kernelized_perceptron
from kernel_viz.kernels.base import (
    linear_kernel,
    polynomial_kernel,
    rbf_gaussian_kernel,
)
from kernel_viz.visualization.component_factory import (
    AlphaEvolutionFactory,
    DecisionBoundaryFactory,
)
from kernel_viz.visualization.core import (
    create_alpha_evolution_component,
    create_decision_boundary_component,
)
from kernel_viz.visualization.visualizer import PerceptronVisualizer


def serialize_artist_state(artist: Artist) -> bytes:
    """Serialize artist state for comparison."""
    state = {}

    # Get common properties
    if hasattr(artist, "get_offsets"):
        state["offsets"] = artist.get_offsets().data
    if hasattr(artist, "get_data"):
        state["data"] = artist.get_data()
    if hasattr(artist, "get_facecolors"):
        state["facecolors"] = artist.get_facecolors()
    if hasattr(artist, "get_edgecolors"):
        state["edgecolors"] = artist.get_edgecolors()
    if hasattr(artist, "get_alpha"):
        state["alpha"] = artist.get_alpha()
    if hasattr(artist, "get_linewidth"):
        state["linewidth"] = artist.get_linewidth()
    if hasattr(artist, "get_linestyle"):
        state["linestyle"] = artist.get_linestyle()
    if hasattr(artist, "get_marker"):
        state["marker"] = artist.get_marker()
    if hasattr(artist, "get_markersize"):
        state["markersize"] = artist.get_markersize()
    if hasattr(artist, "get_color"):
        state["color"] = artist.get_color()
    if hasattr(artist, "get_zorder"):
        state["zorder"] = artist.get_zorder()

    return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)


def compare_artist_lists(
    artists1: List[Artist],
    artists2: List[Artist],
) -> Tuple[bool, List[str]]:
    """Compare two lists of artists for equivalence."""
    differences = []

    if len(artists1) != len(artists2):
        differences.append(
            f"Different number of artists: {len(artists1)} vs {len(artists2)}",
        )
        return False, differences

    for i, (a1, a2) in enumerate(zip(artists1, artists2)):
        # Compare types
        if type(a1) is not type(a2):
            differences.append(
                f"Artist {i}: Different types: {type(a1).__name__} vs {type(a2).__name__}",
            )
            continue

        # Compare serialized states
        try:
            state1 = serialize_artist_state(a1)
            state2 = serialize_artist_state(a2)

            # Use hash for quick comparison
            hash1 = hashlib.md5(state1).hexdigest()
            hash2 = hashlib.md5(state2).hexdigest()

            if hash1 != hash2:
                # Deserialize to find specific differences
                s1 = pickle.loads(state1)
                s2 = pickle.loads(state2)

                for key in set(s1.keys()) | set(s2.keys()):
                    if key not in s1:
                        differences.append(
                            f"Artist {i}: Missing key '{key}' in original",
                        )
                    elif key not in s2:
                        differences.append(
                            f"Artist {i}: Missing key '{key}' in factory",
                        )
                    elif isinstance(s1[key], np.ndarray) and isinstance(
                        s2[key],
                        np.ndarray,
                    ):
                        if not np.array_equal(s1[key], s2[key]):
                            max_diff = np.max(np.abs(s1[key] - s2[key]))
                            differences.append(
                                f"Artist {i}.{key}: Arrays differ (max diff: {max_diff})",
                            )
                    elif s1[key] != s2[key]:
                        differences.append(f"Artist {i}.{key}: {s1[key]} != {s2[key]}")
        except Exception as e:
            differences.append(f"Artist {i}: Error comparing states: {e}")

    return len(differences) == 0, differences


class TestFactoryEquivalence:
    """Test exact equivalence between factory and original implementations."""

    @pytest.fixture
    def trained_model_data(self):
        """Generate consistent training data."""
        np.random.seed(42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([1, -1, -1, 1], dtype=np.float64)

        # Train with different kernels
        models = {}

        for kernel_name, kernel, params in [
            ("rbf", rbf_gaussian_kernel, {"sigma": 0.5}),
            ("linear", linear_kernel, {}),
            ("polynomial", polynomial_kernel, {"degree": 3}),
        ]:
            logger = PerceptronLogger()
            kernelized_perceptron(
                X,
                y,
                kernel,
                kernel_params=params,
                max_iter=5,
                logger=logger,
            )
            models[kernel_name] = logger.get_logs()

        return models

    @pytest.mark.parametrize("kernel_type", ["rbf", "linear", "polynomial"])
    @pytest.mark.parametrize("frame", [0, 2, 4])
    def test_decision_boundary_equivalence(
        self,
        trained_model_data,
        kernel_type,
        frame,
    ):
        """Test decision boundary component equivalence."""
        logs = trained_model_data[kernel_type]
        viz = PerceptronVisualizer()

        # Create components
        comp_orig = create_decision_boundary_component(viz, logs)
        factory = DecisionBoundaryFactory(logs)
        comp_factory = factory.create()

        # Setup and update
        fig, (ax1, ax2) = plt.subplots(1, 2)

        artists1 = comp_orig.setup_func(ax1)
        artists1 = comp_orig.update_func(frame, ax1, artists1)

        artists2 = comp_factory.setup_func(ax2)
        artists2 = comp_factory.update_func(frame, ax2, artists2)

        # Compare
        match, differences = compare_artist_lists(artists1, artists2)

        plt.close(fig)

        if not match:
            pytest.fail("Factory implementation differs:\n" + "\n".join(differences))

    @pytest.mark.parametrize("kernel_type", ["rbf", "linear"])
    def test_alpha_evolution_equivalence(self, trained_model_data, kernel_type):
        """Test alpha evolution component equivalence."""
        logs = trained_model_data[kernel_type]
        viz = PerceptronVisualizer()
        viz.total_frames = 5

        # Create components
        comp_orig = create_alpha_evolution_component(viz, logs)
        factory = AlphaEvolutionFactory(logs)
        factory.total_frames = 5
        comp_factory = factory.create()

        # Test setup
        fig, (ax1, ax2) = plt.subplots(1, 2)

        artists1 = comp_orig.setup_func(ax1)
        artists2 = comp_factory.setup_func(ax2)

        match, differences = compare_artist_lists(artists1, artists2)

        if not match:
            plt.close(fig)
            pytest.fail("Setup differs:\n" + "\n".join(differences))

        # Test updates
        for frame in [0, 2, 4]:
            artists1 = comp_orig.update_func(frame, ax1, artists1)
            artists2 = comp_factory.update_func(frame, ax2, artists2)

            match, differences = compare_artist_lists(artists1, artists2)

            if not match:
                plt.close(fig)
                pytest.fail(f"Frame {frame} differs:\n" + "\n".join(differences))

        plt.close(fig)

    def test_parameter_preservation(self, trained_model_data):
        """Test that factory preserves all parameters correctly."""
        logs = trained_model_data["rbf"]

        # Test DecisionBoundaryFactory with custom parameters
        factory = DecisionBoundaryFactory(logs, plot_type="line", fixed_dims={2: 0.5})
        assert factory.plot_type == "line"
        assert factory.fixed_dims == {2: 0.5}

        # Test that parameters affect behavior
        factory_default = DecisionBoundaryFactory(logs)
        assert factory_default.plot_type == "contour"  # RBF should default to contour

        # Test AlphaEvolutionFactory
        factory_alpha = AlphaEvolutionFactory(logs, debug_mode=True)
        assert factory_alpha.debug_mode is True

    def test_axes_properties_preserved(self, trained_model_data):
        """Test that axes properties (labels, titles, etc.) are preserved."""
        logs = trained_model_data["linear"]
        viz = PerceptronVisualizer()

        # Decision boundary
        comp_orig = create_decision_boundary_component(viz, logs)
        factory = DecisionBoundaryFactory(logs)
        comp_factory = factory.create()

        fig, (ax1, ax2) = plt.subplots(1, 2)

        comp_orig.setup_func(ax1)
        comp_orig.update_func(2, ax1, comp_orig.setup_func(ax1))

        comp_factory.setup_func(ax2)
        comp_factory.update_func(2, ax2, comp_factory.setup_func(ax2))

        # Compare axes properties
        assert ax1.get_xlabel() == ax2.get_xlabel()
        assert ax1.get_ylabel() == ax2.get_ylabel()
        assert ax1.get_title() == ax2.get_title()
        assert ax1.get_xlim() == ax2.get_xlim()
        assert ax1.get_ylim() == ax2.get_ylim()

        plt.close(fig)


class TestFactoryRegressionPrevention:
    """Tests to prevent regressions in factory implementations."""

    def test_factory_handles_edge_cases(self):
        """Test factory implementations handle edge cases properly."""
        # Single sample
        logs_single = {
            "feature_space": np.array([[0.5, 0.5]]),
            "true_labels": np.array([1]),
            "kernel": linear_kernel,
            "kernel_params": {},
            "alphas": [{"alphas": np.array([1.0])}],
        }

        factory = DecisionBoundaryFactory(logs_single)
        component = factory.create()

        # Should not crash
        fig, ax = plt.subplots()
        artists = component.setup_func(ax)
        component.update_func(0, ax, artists)
        plt.close(fig)

    def test_factory_bug_fixes_preserved(self):
        """Test that bug fixes in factory are preserved."""
        logs = {
            "feature_space": np.array([[0, 0], [1, 1]]),
            "true_labels": np.array([1, -1]),
            "kernel": linear_kernel,
            "kernel_params": {},
            "alphas": [{"alphas": np.array([0.1, -0.1])}],
        }

        # The original had a bug with kernel name check
        factory = DecisionBoundaryFactory(logs)
        # This should correctly identify linear kernel
        assert factory.plot_type == "line"

        # Test with affine kernel
        from kernel_viz.kernels.base import affine_kernel

        logs["kernel"] = affine_kernel
        factory_affine = DecisionBoundaryFactory(logs)
        assert factory_affine.plot_type == "line"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
