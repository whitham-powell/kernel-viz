# kernel_visualizer.py
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
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
    """Shows the full kernel response surface evolution during training.

    This component visualizes how the decision function f(x) = sum(alpha_i * K(x_i, x))
    evolves across the feature space, showing the complete response surface
    with a colorbar to indicate the magnitude of the response.
    """
    xs = logs["feature_space"]
    kernel = logs["kernel"]
    kernel_params = logs["kernel_params"] or {}
    true_labels = logs["true_labels"]

    # Pre-calculate grid points for response surface
    margin = 1.0  # Margin around data points
    x_min, x_max = xs[:, 0].min() - margin, xs[:, 0].max() + margin
    y_min, y_max = xs[:, 1].min() - margin, xs[:, 1].max() + margin
    grid_resolution = 50  # Balance between smoothness and performance

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten grid

    # Pre-compute global normalization bounds for consistent colorbar
    alphas_log = logs["alphas"]
    global_response_min = float("inf")
    global_response_max = float("-inf")

    for frame_data in alphas_log:
        alphas = frame_data["alphas"]
        response = np.zeros(len(grid_points), dtype=np.float64)
        for i, alpha in enumerate(alphas):
            if abs(alpha) > 1e-10:
                kernel_values = np.array(
                    [
                        kernel(xs[i], grid_point, **kernel_params)
                        for grid_point in grid_points
                    ],
                    dtype=np.float64,
                )
                response += alpha * kernel_values
        response_reshaped = response.reshape(xx.shape)
        global_response_min = min(global_response_min, response_reshaped.min())
        global_response_max = max(global_response_max, response_reshaped.max())

    # Ensure we have a valid range
    if global_response_max - global_response_min < 1e-10:
        global_response_min = -1
        global_response_max = 1

    if self.debug_mode:
        print("\nInitializing Kernel Response Component:")
        print(f"Feature space shape: {xs.shape}")
        print(f"Grid resolution: {grid_resolution}x{grid_resolution}")
        print(f"X range: [{x_min:.2f}, {x_max:.2f}]")
        print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
        print(
            f"Response range: [{global_response_min:.2f}, {global_response_max:.2f}]",
        )
        print(f"Kernel: {kernel.__name__}")
        print(f"Kernel params: {kernel_params}")

    def setup(ax: Axes) -> List[Artist]:
        if self.debug_mode:
            print("Setting up kernel response surface component")

        # Initial response surface
        surface = ax.contourf(
            xx,
            yy,
            np.zeros_like(xx),
            levels=20,
            cmap="RdBu_r",
            vmin=global_response_min,
            vmax=global_response_max,
        )

        # Add colorbar
        cbar = plt.colorbar(surface, ax=ax, pad=0.02)
        cbar.set_label("Kernel Response f(x)", rotation=270, labelpad=15)

        # Decision boundary line
        decision_boundary = ax.contour(
            xx,
            yy,
            np.zeros_like(xx),
            levels=[0],
            colors="black",
            linewidths=2,
            linestyles="--",
        )

        # Points with markers for positive/negative classes
        points_pos = ax.scatter(
            xs[true_labels == 1, 0],
            xs[true_labels == 1, 1],
            c="blue",
            s=80,
            marker="o",
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
            label="Class +1",
        )

        points_neg = ax.scatter(
            xs[true_labels == -1, 0],
            xs[true_labels == -1, 1],
            c="red",
            s=80,
            marker="s",
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
            label="Class -1",
        )

        # Configure axes
        ax.set_title("Kernel Response Surface")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc="upper right")
        ax.set_aspect("equal", adjustable="box")

        if self.debug_mode:
            print("Initial response surface plotted")
            print(f"Axes limits: x=[{ax.get_xlim()}], y=[{ax.get_ylim()}]")

        return [surface, decision_boundary, points_pos, points_neg]

    def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        if self.debug_mode and frame % 10 == 0:
            print(f"\nUpdating frame {frame}")

        surface, decision_boundary, points_pos, points_neg = artists
        alphas = logs["alphas"][frame]["alphas"]

        # Compute kernel response on grid
        response = np.zeros(len(grid_points), dtype=np.float64)

        # Only compute for non-zero alphas (support vectors)
        active_indices = np.where(np.abs(alphas) > 1e-10)[0]

        if len(active_indices) > 0:
            # Vectorized computation for efficiency
            for i in active_indices:
                kernel_values = np.array(
                    [
                        kernel(xs[i], grid_point, **kernel_params)
                        for grid_point in grid_points
                    ],
                    dtype=np.float64,
                )
                response = response + alphas[i] * kernel_values

        response_reshaped = response.reshape(xx.shape)

        # Clear existing contours (but not scatter plots)
        for coll in ax.collections[:]:
            if coll not in [points_pos, points_neg] and not isinstance(
                coll,
                type(ax.collections[0]),
            ):
                coll.remove()

        # Update response surface
        new_surface = ax.contourf(
            xx,
            yy,
            response_reshaped,
            levels=20,
            cmap="RdBu_r",
            vmin=global_response_min,
            vmax=global_response_max,
        )

        # Update decision boundary
        new_boundary = ax.contour(
            xx,
            yy,
            response_reshaped,
            levels=[0],
            colors="black",
            linewidths=2,
            linestyles="--",
        )

        # Highlight support vectors
        support_vector_mask_pos = np.abs(alphas[true_labels == 1]) > 1e-10
        support_vector_mask_neg = np.abs(alphas[true_labels == -1]) > 1e-10

        # Update point appearance for support vectors
        points_pos.set_sizes([120 if sv else 80 for sv in support_vector_mask_pos])
        points_neg.set_sizes([120 if sv else 80 for sv in support_vector_mask_neg])

        # Highlight active support vectors with different edge color
        points_pos.set_edgecolors(
            ["yellow" if sv else "white" for sv in support_vector_mask_pos],
        )
        points_neg.set_edgecolors(
            ["yellow" if sv else "white" for sv in support_vector_mask_neg],
        )

        ax.set_title(f"Kernel Response Surface - Iteration {frame + 1}")

        if self.debug_mode and frame % 10 == 0:
            print(f"Active support vectors: {len(active_indices)}")
            print(
                f"Response range: [{response_reshaped.min():.2f}, {response_reshaped.max():.2f}]",
            )

        return [new_surface, new_boundary, points_pos, points_neg]

    # Initial grid position will be updated by visualizer
    component = AnimationComponent(
        setup_func=setup,
        update_func=update,
        subplot_params={"gridspec": (0, 0)},
    )

    if self.debug_mode:
        print("Kernel response component initialized")

    return component


def create_kernel_matrix_component(
    self: PerceptronVisualizer,
    logs: Dict[str, Any],
) -> AnimationComponent:
    """Shows the kernel matrix heatmap with alpha values overlay.

    Since the kernel matrix is constant during training, this component
    focuses on showing how alpha values evolve relative to the kernel structure.
    """
    kernel_matrix = logs.get("kernel_matrix")

    if kernel_matrix is None:
        # Compute kernel matrix if not provided
        xs = logs["feature_space"]
        kernel = logs["kernel"]
        kernel_params = logs["kernel_params"] or {}
        n_samples = len(xs)
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = kernel(xs[i], xs[j], **kernel_params)

    n_samples = len(kernel_matrix)

    def setup(ax: Axes) -> List[Artist]:
        # Create a more informative visualization
        im = ax.imshow(
            kernel_matrix,
            cmap="RdBu_r",
            aspect="equal",
            vmin=-abs(kernel_matrix).max(),
            vmax=abs(kernel_matrix).max(),
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Kernel Value", rotation=270, labelpad=15)

        # Set ticks and labels
        if n_samples <= 20:  # Only show individual labels for small datasets
            ax.set_xticks(range(n_samples))
            ax.set_yticks(range(n_samples))
            ax.set_xticklabels([f"{i}" for i in range(n_samples)], fontsize=8)
            ax.set_yticklabels([f"{i}" for i in range(n_samples)], fontsize=8)

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        ax.set_title("Kernel Matrix K(x_i, x_j)")

        # Add grid for better readability
        ax.set_xticks(np.arange(n_samples) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_samples) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)

        # Initialize alpha indicators
        alpha_indicators = []
        for i in range(n_samples):
            # Add markers on diagonal to show support vectors
            marker = ax.plot(
                i,
                i,
                "o",
                color="yellow",
                markersize=0,
                markeredgecolor="black",
                markeredgewidth=1,
            )[0]
            alpha_indicators.append(marker)

        return [im] + alpha_indicators

    def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        alpha_indicators = artists[1:]
        alphas = logs["alphas"][frame]["alphas"]

        # Update support vector indicators
        for i, (alpha, marker) in enumerate(zip(alphas, alpha_indicators)):
            if abs(alpha) > 1e-10:
                # Support vector - show with size proportional to |alpha|
                marker.set_markersize(min(15, 5 + 10 * abs(alpha)))
                marker.set_color("yellow" if alpha > 0 else "cyan")
            else:
                marker.set_markersize(0)

        # Update title with iteration info
        n_support = np.sum(np.abs(alphas) > 1e-10)
        ax.set_title(
            f"Kernel Matrix - Iteration {frame + 1} ({n_support} support vectors)",
        )

        return artists

    return AnimationComponent(
        setup_func=setup,
        update_func=update,
        subplot_params={"gridspec": (1, 1)},
    )


def create_misclassification_tracker_component(
    self: PerceptronVisualizer,
    logs: Dict[str, Any],
) -> AnimationComponent:
    """
    Visualizes misclassified training points dynamically during perceptron updates.
    """
    xs = logs["feature_space"]  # Training points
    true_labels = logs["true_labels"]  # Ground-truth labels (+1 or -1)
    alphas_log = logs["alphas"]  # Alpha values for each iteration
    kernel = logs["kernel"]  # Kernel function
    kernel_params = logs["kernel_params"] or {}  # Kernel parameters

    def setup(ax: Axes) -> List[Artist]:
        """
        Setup the misclassification tracker visualization.
        """
        if self.debug_mode:
            print("Setting up misclassification tracker")

        # Initial scatter plot for all training points
        points = ax.scatter(
            xs[:, 0],
            xs[:, 1],
            c="gray",
            s=80,
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )

        # Configure plot
        ax.set_title("Misclassified Points Tracker")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # Add custom legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Correctly Classified",
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="black",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Misclassified",
                markerfacecolor="red",
                markersize=10,
                markeredgecolor="black",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", title="Point Status")

        return [points]

    def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """
        Update the visualization for the given iteration.
        """
        points = artists[0]
        alphas = alphas_log[frame]["alphas"]

        # Compute the decision function for all training points
        decision_function = np.zeros(len(xs))
        for i in range(len(xs)):
            decision_function[i] = np.sum(
                [
                    alphas[j] * kernel(xs[j], xs[i], **kernel_params)
                    for j in range(len(xs))
                ],
            )

        # Identify misclassified points
        misclassified = (true_labels * decision_function) < 0

        # Update colors: Red for misclassified, Gray for correctly classified
        colors = ["red" if m else "gray" for m in misclassified]
        points.set_color(colors)

        return [points]

    return AnimationComponent(
        setup_func=setup,
        update_func=update,
        subplot_params={"gridspec": (0, 0)},
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
