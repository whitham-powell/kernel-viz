# kernel_visualizer.py
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.contour import QuadContourSet
from numpy.typing import ArrayLike, NDArray
from typing_extensions import TypeAlias

from .base import AnimationComponent
from .visualizer import PerceptronVisualizer

ContourOutput: TypeAlias = Union[QuadContourSet, List[PathCollection]]


def compute_decision_boundary(
    xs: NDArray[np.float64],
    alphas: NDArray[np.float64],
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
    kernel_params: Optional[Dict[str, Any]],
    fixed_dims: Optional[Dict[int, float]] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Computes a decision boundary with optional fixed dimensions."""

    n_dims = xs.shape[1]
    fixed_dims = fixed_dims or {}

    # Fix unspecified dimensions to zero
    fixed_x = np.zeros(n_dims)
    for dim, value in fixed_dims.items():
        fixed_x[dim] = value

    x_min, x_max = xs[:, 0].min() - 1, xs[:, 0].max() + 1
    y_min, y_max = xs[:, 1].min() - 1, xs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    zz = []
    kernel_kwargs = kernel_params or {}
    for point in grid:
        full_point = fixed_x.copy()
        full_point[0] = point[0]
        full_point[1] = point[1]
        zz.append(
            np.sum(
                [
                    alphas[j] * kernel(xs[j], full_point, **kernel_kwargs)
                    for j in range(len(xs))
                ],
            ),
        )
    zz = np.array(zz).reshape(xx.shape)  # type: ignore

    return xx, yy, zz


# Component creation methods for PerceptronVisualizer
# These methods extend the PerceptronVisualizer class defined in visualizer.py


def create_decision_boundary_component(
    self: PerceptronVisualizer,
    logs: Dict[str, Any],
    plot_type: Optional[str] = None,
    fixed_dims: Optional[Dict[int, float]] = None,
) -> AnimationComponent:
    """Creates decision boundary visualization component using factory pattern.

    This method now uses the DecisionBoundaryFactory for cleaner separation of concerns
    and better testability.
    """
    from .component_factory import DecisionBoundaryFactory

    factory = DecisionBoundaryFactory(logs, plot_type=plot_type, fixed_dims=fixed_dims)
    return factory.create(
        subplot_params={"gridspec": (0, slice(None))},
        name="decision_boundary",
    )


def create_alpha_evolution_component(
    self: PerceptronVisualizer,
    logs: Dict[str, Any],
) -> AnimationComponent:
    """Creates alpha evolution visualization component using factory pattern.

    This method now uses the AlphaEvolutionFactory for cleaner code organization
    and improved maintainability.
    """
    from .component_factory import AlphaEvolutionFactory

    factory = AlphaEvolutionFactory(logs, debug_mode=self.debug_mode)
    # Pass total_frames to the factory
    factory.total_frames = self.total_frames
    return factory.create(subplot_params={"gridspec": (0, 0)}, name="alpha_evolution")


def create_kernel_response_component(
    self: PerceptronVisualizer,
    logs: Dict[str, Any],
) -> AnimationComponent:
    """Shows the full kernel response surface evolution during training using factory pattern."""
    from .component_factory import KernelResponseFactory

    factory = KernelResponseFactory(logs, debug_mode=self.debug_mode)
    return factory.create(
        subplot_params={"gridspec": (0, 1), "colspan": 2},
        name="kernel_response",
    )


def create_kernel_matrix_component(
    self: PerceptronVisualizer,
    logs: Dict[str, Any],
) -> AnimationComponent:
    """Shows the kernel matrix heatmap with alpha values overlay using factory pattern."""
    from .component_factory import KernelMatrixFactory

    factory = KernelMatrixFactory(logs, debug_mode=self.debug_mode)
    return factory.create(subplot_params={"gridspec": (1, 1)}, name="kernel_matrix")


def create_misclassification_tracker_component(
    self: PerceptronVisualizer,
    logs: Dict[str, Any],
) -> AnimationComponent:
    """Visualizes misclassified training points dynamically using factory pattern."""
    from .component_factory import MisclassificationTrackerFactory

    factory = MisclassificationTrackerFactory(logs, debug_mode=self.debug_mode)
    return factory.create(
        subplot_params={"gridspec": (0, 0)},
        name="misclassification_tracker",
    )


def create_kernel_matrix_heatmap_component(
    self: PerceptronVisualizer,
    logs: Dict[str, Any],
) -> AnimationComponent:
    """Creates kernel matrix heatmap animation using the factory pattern.

    This is a new implementation using the factory pattern as a proof of concept
    for future refactoring of other visualization components.
    """
    from .component_factory import KernelMatrixHeatmapFactory

    factory = KernelMatrixHeatmapFactory(logs, debug_mode=self.debug_mode)
    return factory.create(
        subplot_params={"gridspec": (1, 1)},
        name="kernel_matrix_heatmap",
    )


# Attach component creation methods to PerceptronVisualizer
PerceptronVisualizer.create_decision_boundary_component = (  # type: ignore[attr-defined]
    create_decision_boundary_component
)
PerceptronVisualizer.create_alpha_evolution_component = create_alpha_evolution_component  # type: ignore[attr-defined]
PerceptronVisualizer.create_kernel_response_component = create_kernel_response_component  # type: ignore[attr-defined]
PerceptronVisualizer.create_kernel_matrix_component = create_kernel_matrix_component  # type: ignore[attr-defined]
PerceptronVisualizer.create_kernel_matrix_heatmap_component = (  # type: ignore[attr-defined]
    create_kernel_matrix_heatmap_component
)
PerceptronVisualizer.create_misclassification_tracker_component = (  # type: ignore[attr-defined]
    create_misclassification_tracker_component
)
