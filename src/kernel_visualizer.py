# kernel_visualizer.py
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import Animation, FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.contour import QuadContourSet
from matplotlib.gridspec import GridSpec
from numpy.typing import ArrayLike, NDArray
from typing_extensions import TypeAlias

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
    for point in grid:
        full_point = fixed_x.copy()
        full_point[0] = point[0]
        full_point[1] = point[1]
        zz.append(
            np.sum(
                [
                    alphas[j] * kernel(xs[j], full_point, **kernel_params)
                    for j in range(len(xs))
                ],
            ),
        )
    zz = np.array(zz).reshape(xx.shape)  # type: ignore

    return xx, yy, zz


# TODO: refactor AnimationComponent class to PerceptronVisualizer file
@dataclass
class AnimationComponent:
    """Represents a single visualization component."""

    setup_func: Callable[[Axes], List[Artist]]
    update_func: Callable[[int, Axes, List[Artist]], List[Artist]]
    subplot_params: Dict[str, Any]
    name: Optional[str] = None


# TODO: refactor PerceptronVisualizer class to PerceptronVisualizer file
class PerceptronVisualizer:
    def __init__(self) -> None:
        self.components: List[AnimationComponent] = []
        self.debug_mode = False
        self._animation: Optional[Animation] = None
        self.total_frames: Optional[int] = None

    def set_debug_mode(self, enabled: bool = True) -> None:
        """Enable or disable debug mode."""
        self.debug_mode = enabled

    def add_component(self, component: AnimationComponent) -> None:
        self.components.append(component)
        self._update_grid_layout()

    def _calculate_grid_dimensions(self) -> Tuple[int, int]:
        """Calculate optimal grid dimensions based on number of components"""
        n = len(self.components)
        if n <= 1:
            return (1, 1)
        elif n == 2:
            return (1, 2)
        elif n == 3:
            return (2, 2)
        elif n == 4:
            return (2, 2)
        else:
            # For more components create a roughly square grid
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            return (rows, cols)

    def _update_grid_layout(self) -> None:
        """Update grid positions for all components based on current configuration."""
        rows, cols = self._calculate_grid_dimensions()
        n_components = len(self.components)

        if self.debug_mode:
            print(
                f"Updating grid layout: {rows} x {cols} for {n_components} components",
            )

        # Special layouts for common cases
        if n_components == 1:
            self.components[0].subplot_params["gridspec"] = (0, slice(None))
        elif n_components == 2:
            self.components[0].subplot_params["gridspec"] = (0, 0)
            self.components[1].subplot_params["gridspec"] = (0, 1)

        else:
            # General case: fill grid left to right, top to bottom
            for idx, component in enumerate(self.components):
                row = idx // cols
                col = idx % cols

                # Special handling for components that should span multiple columns
                if idx == n_components - 1 and cols > 1 and idx % cols == 0:
                    component.subplot_params["gridspec"] = (row, slice(col, cols))
                else:
                    component.subplot_params["gridspec"] = (row, col)

                if self.debug_mode:
                    print(
                        f"Component {idx} ({component.name}) position: {component.subplot_params['gridspec']}",
                    )

    def _save_animation(self, save_path: str, fps: int) -> None:
        """Save animation with error handling."""
        if self._animation is None:
            raise ValueError("No animation to save")

        try:
            file_extension = save_path.split(".")[-1].lower()
            if file_extension in ["mp4", "mov"]:
                writer = "ffmpeg"
            elif file_extension == "gif":
                writer = "pillow"
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")

            self._animation.save(save_path, writer=writer, fps=fps)

        except Exception as e:
            print(f"Error saving animation: {e}")
            raise

    def animate(  # noqa : C901
        self,
        logs: Dict[str, Any],
        figsize: Tuple[float, float] = (15, 10),
        save_path: Optional[str] = None,
        fps: int = 10,
        debug: bool = False,
    ) -> Animation:
        """Create and display/save the combined animation."""
        self.set_debug_mode(debug)
        self.total_frames = len(logs["misclassification_count"])
        begin_animate_time = time.time()

        if len(self.components) == 0:
            raise ValueError("No components added to visualizer")

        # TODO add error handling for missing logs / use if raise
        assert (
            self.total_frames is not None and self.total_frames > 0
        ), f"Animation requires valid number of frames. self.total_frames={self.total_frames}"

        print(f"Starting animation with {self.total_frames} frames")

        plt.close("all")  # Close any existing figures
        self.figure = plt.figure(figsize=figsize)
        rows, cols = self._calculate_grid_dimensions()
        self.grid_spec = GridSpec(
            rows,
            cols,
            figure=self.figure,
            width_ratios=[1, 1.2],
        )

        if self.debug_mode:
            print("Debug mode enabled")
            print(f"Created {rows}x{cols} grid for {len(self.components)} components")

        # Initialize components
        component_artists = []
        for idx, component in enumerate(self.components):
            if self.debug_mode:
                print(f"Setting up component {idx}: {component.name}")

            grid_pos = component.subplot_params.get("gridspec")
            ax = self.figure.add_subplot(self.grid_spec[grid_pos])

            try:
                artists = component.setup_func(ax)
                component_artists.append((component, ax, artists))
            except Exception as e:
                print(f"Error setting up component {idx} : {component.name}: {e}")
                raise

        def update(frame: int) -> List[Artist]:
            update_frame_time = time.time()

            if self.debug_mode:
                assert (
                    self.total_frames is not None
                ), "self.total_frames is None in animate() -> update()"
                print(f"\nProcessing frame {frame} / {self.total_frames - 1}")

            all_artists = []

            for component, ax, artists in component_artists:
                try:
                    updated_artists = component.update_func(frame, ax, artists)
                    if not isinstance(updated_artists, list):
                        print(
                            f"Warning: Component {component.name} returned non-list: type={type(updated_artists)}",
                        )
                        updated_artists = list(updated_artists)

                    all_artists.extend(updated_artists)
                except Exception as e:
                    print(
                        f"Error updating component {component.name} at frame {frame}: {e}",
                    )
                    raise

            print(
                f"Frame {frame} completed in {time.time() - update_frame_time:.3f}s",
            )

            return all_artists

        # Main figure layout configuration
        self.figure.tight_layout(pad=1.75)
        # TODO should this be a class attribute?
        interval = 1000 / fps

        self._animation = FuncAnimation(
            self.figure,
            update,
            frames=self.total_frames,
            interval=interval,
            repeat=False,
            blit=True,
        )

        if save_path:
            if self.debug_mode:
                print(f"Saving animation to {save_path}")
            self._save_animation(save_path, fps)

        plt.tight_layout()
        print(
            f"Animation configured with {self.total_frames} and ready for display or saving in {time.time() - begin_animate_time:.3f}s",
        )

        return self._animation

    # Component creation methods
    def create_decision_boundary_component(
        self,
        logs: Dict[str, Any],
        plot_type: Optional[str] = None,
        fixed_dims: Optional[Dict[int, float]] = None,
    ) -> AnimationComponent:
        """Creates decision boundary visualization component."""
        xs = logs["feature_space"]
        ys = logs["true_labels"]
        kernel = logs["kernel"]
        kernel_params = logs["kernel_params"] or {}

        # Determine plot type based on kernel if not specified
        if plot_type is None:
            plot_type = (
                "line"
                if kernel.__name__ == ("linear_kernel" or "affine_kernel")
                else "contour"
            )

        def clear_old_contours(ax: Axes) -> None:
            """Remove all contour collections from previous frame."""
            for artist in ax.collections[1:]:  # Keep scatter plot
                artist.remove()

        def update_line_plot(
            ax: Axes,
            artists: List[Artist],
            xx: ArrayLike,
            yy: ArrayLike,
            zz: ArrayLike,
        ) -> List[Artist]:
            """Update the line plot showing decision boundary."""
            scatter, line = artists
            temp_contour = ax.contour(xx, yy, zz, levels=[0], colors="black")

            if temp_contour.collections[0].get_paths():
                vertices = temp_contour.collections[0].get_paths()[0].vertices
                line.set_data(vertices[:, 0], vertices[:, 1])

            for coll in temp_contour.collections:
                coll.remove()

            return [scatter, line]

        def update_contour_plot(
            ax: Axes,
            artists: List[Artist],
            xx: ArrayLike,
            yy: ArrayLike,
            zz: ArrayLike,
            frame: int,
        ) -> List[Artist]:
            """Update the filled contour plot showing decision regions."""
            scatter = artists[0]
            contour = ax.contourf(
                xx,
                yy,
                zz,
                levels=[-1, 0, 1],
                alpha=0.3,
                cmap="coolwarm",
            )
            # Update title for each frame
            ax.set_title(f"Decision Boundary - Iteration {frame + 1}")
            return [scatter] + list(contour.collections)

        def setup(ax: Axes) -> List[Artist]:
            # Setup scatter plot
            scatter = ax.scatter(
                xs[:, 0],
                xs[:, 1],
                c=ys,
                cmap="bwr",
                edgecolor="k",
                zorder=2,
            )

            # Set margins
            margin = 0.1
            x_min, x_max = xs[:, 0].min(), xs[:, 0].max()
            y_min, y_max = xs[:, 1].min(), xs[:, 1].max()
            ax.set_xlim(
                [x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)],
            )
            ax.set_ylim(
                [y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)],
            )
            # Set initial title
            ax.set_title("Decision Boundary - Iteration 1")

            # Set axes
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

            # Initialize line if using line plot
            if plot_type == "line":
                line = ax.plot([], [], "k-", lw=2)[0]
                return [scatter, line]
            return [scatter]

        def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
            """Update the visualization for the current frame."""
            # Clear any existing contours
            clear_old_contours(ax)

            # Compute new boundary
            alphas = logs["alphas"][frame]["alphas"]
            xx, yy, zz = compute_decision_boundary(
                xs,
                alphas,
                kernel,
                kernel_params,
                fixed_dims,
            )

            if plot_type == "line":
                return update_line_plot(ax, artists, xx, yy, zz)
            else:
                return update_contour_plot(ax, artists, xx, yy, zz, frame)

        return AnimationComponent(
            setup_func=setup,
            update_func=update,
            subplot_params={"gridspec": (0, slice(None))},
        )

    def create_alpha_evolution_component(
        self,
        logs: Dict[str, Any],
    ) -> AnimationComponent:
        """Visualizes how alpha values change over training iterations."""
        alphas_history = logs["alphas"]
        n_samples = len(alphas_history[0]["alphas"])

        all_alphas = np.array([entry["alphas"] for entry in alphas_history])

        if self.debug_mode:
            print("\nInitializing Alpha Evolution Component:")
            print(f"Number of samples: {n_samples}")
            print(f"Number of iterations: {len(alphas_history)}")
            print(f"Shape of all_alphas: {all_alphas.shape}")
            print(f"First iteration alphas:\n{all_alphas[0]}")

        # Calculate global min/max and appropriate margins
        min_alpha = np.min(all_alphas)
        max_alpha = np.max(all_alphas)
        alpha_range = max_alpha - min_alpha
        min_margin = 0.1  # minimum margin to ensure visibility
        margin = max(min_margin, alpha_range * 0.1)

        # Y-axis limits
        y_min = min_alpha - margin
        y_max = max_alpha + margin

        if self.debug_mode:
            print("\nValue Ranges:")
            print(f"Alpha range: [{min_alpha:.4f}, {max_alpha:.4f}]")
            print(f"Margin: {margin:.4f}")
            print(f"Y-axis limits: [{y_min:.4f}, {y_max:.4f}]")

        def setup(ax: Axes) -> List[Artist]:
            if self.debug_mode:
                print("Setting up alpha evolution component")

            lines = []
            for i in range(n_samples):
                (line,) = ax.plot(
                    [],
                    [],
                    label=f"$\\alpha_{{{i}}}$",
                    alpha=0.3,
                    linewidth=0.5,
                    color="gray",
                )
                lines.append(line)

            # Configure axes
            ax.set_autoscale_on(False)
            ax.set_xlim(
                0,
                (
                    self.total_frames
                    if self.total_frames is not None
                    else len(alphas_history)
                ),
            )
            ax.set_ylim(y_min, y_max)

            # Aspect ratio control
            ax.set_aspect(1)

            # Set labels and grid
            ax.set_title("Alpha Values Evolution")
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Alpha Value")
            ax.grid(True, linestyle="--", alpha=0.7)

            # Position legend relative to alpha evolution plot itself
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=3,
                borderaxespad=0,
                fontsize="small",
            )

            if self.debug_mode:
                print(f"Created {len(lines)} lines")
                print(f"X-axis range: [0, {ax.get_xlim()[1]}]")
                print(f"Y-axis range: [{ax.get_ylim()[0]:.4f}, {ax.get_ylim()[1]:.4f}]")

            return lines

        def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:

            if self.debug_mode:
                print(f"\n Updating frame {frame}")

            current_alphas = alphas_history[frame]["alphas"]
            legend = ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=3,
                borderaxespad=0,
                fontsize="small",
            )

            # Update each line
            for idx, line in enumerate(artists):
                x_data = range(frame + 1)
                y_data = [alphas_history[j]["alphas"][idx] for j in range(frame + 1)]

                # Update line data
                line.set_data(x_data, y_data)

                # Update line appearance based on current alpha value
                is_active = abs(current_alphas[idx]) > 1e-10
                line.set_color("red" if is_active else "gray")
                line.set_alpha(0.7 if is_active else 0.1)
                line.set_zorder(2 if is_active else 1)  # Active lines/points on top

                # Update the legend line color and alpha
                if legend:
                    legend_line = legend.get_lines()[idx]
                    legend_line.set_color("red" if is_active else "gray")
                    legend_line.set_alpha(0.7 if is_active else 0.1)

            # Update statistics and title
            n_active = np.sum(np.abs(current_alphas) > 1e-10)
            active_percentage = (n_active / n_samples) * 100

            ax.set_title(
                f"Alpha Values Evolution - Iteration {frame + 1}\n"
                f"Active points: {n_active}/{n_samples} ({active_percentage:.1f}%)",
            )

            if self.debug_mode and frame % 10 == 0:
                print(f"Active points: {n_active}")
                print(f"Max current alpha: {np.max(np.abs(current_alphas)):.4f}")

            return artists

        component = AnimationComponent(
            setup_func=setup,
            update_func=update,
            subplot_params={"gridspec": (0, 0)},
        )

        if self.debug_mode:
            print(f"\nGrid position: {component.subplot_params['gridspec']}")
            print("Component initialization complete")

        return component

    # FIXME: This is probably not useful for kernelized perceptron or its implemented incorrectly. Results in overly complicated and fancy decision boundary animation
    def create_kernel_response_component(
        self,
        logs: Dict[str, Any],
    ) -> AnimationComponent:
        """Shows how the kernel response changes with alpha values."""
        xs = logs["feature_space"]
        kernel = logs["kernel"]
        kernel_params = logs["kernel_params"] or {}
        true_labels = logs["true_labels"]
        alphas_log = logs["alphas"]

        # Pre-calculate grid points
        margin = 1.0  # Consistent margin around data points
        x_min, x_max = xs[:, 0].min() - margin, xs[:, 0].max() + margin
        y_min, y_max = xs[:, 1].min() - margin, xs[:, 1].max() + margin
        grid_resolution = 50

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution),
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten grid

        # Pre-compute global normalization bounds
        global_response_min = float("inf")
        global_response_max = float("-inf")

        for frame_data in alphas_log:
            alphas = frame_data["alphas"]
            response = np.zeros_like(xx, dtype=np.float64)
            for i, alpha in enumerate(alphas):
                response += alpha * np.array(
                    [
                        kernel(xs[i], grid_point, **kernel_params)
                        for grid_point in grid_points
                    ],
                ).reshape(xx.shape)
                global_response_min = min(global_response_min, response.min())
                global_response_max = max(global_response_max, response.max())

        def normalize_response(response: NDArray[np.float64]) -> NDArray[np.float64]:
            """Normalize the kernel response values."""
            if global_response_max - global_response_min > 1e-10:
                return (response - global_response_min) / (
                    global_response_max - global_response_min
                )
            return response

        if self.debug_mode:
            print("\nInitializing Kernel Response Component:")
            print(f"Feature space shape: {xs.shape}")
            print(f"Grid resolution: {grid_resolution}x{grid_resolution}")
            print(f"X range: [{x_min:.2f}, {x_max:.2f}]")
            print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
            print(f"Kernel: {kernel.__name__}")
            print(f"Kernel params: {kernel_params}")
            print("\nKernel Response Normalization:")
            print(
                f"Global response range: [{global_response_min:.4f}, {global_response_max:.4f}]",
            )

        def setup(ax: Axes) -> List[Artist]:
            if self.debug_mode:
                print("Setting up kernel response component")

            # Initial response surface

            surface = ax.contourf(xx, yy, np.zeros_like(xx), levels=20, cmap="PuOr")

            # Add contour lines for level sets
            contours = ax.contour(
                xx,
                yy,
                np.zeros_like(xx),
                levels=10,
                colors="k",
                alpha=0.2,
                linewidths=0.5,
            )

            # Points with different markers for positive/negative classes

            points_pos = ax.scatter(
                xs[true_labels == 1, 0],  # x coordinates for positive class
                xs[true_labels == 1, 1],  # y coordinates for positive class
                c="gray",
                s=80,
                marker="o",  # circle for positive class
                edgecolor="black",
                linewidth=1,
                zorder=2,
                label="Positive class",
            )

            points_neg = ax.scatter(
                xs[true_labels == -1, 0],  # x coordinates for negative class
                xs[true_labels == -1, 1],  # y coordinates for negative class
                c="gray",
                s=80,
                marker="x",  # x for negative class
                edgecolor="black",
                linewidth=1,
                zorder=2,
                label="Negative class",
            )

            # Configure colorbars
            surface_colorbar = plt.colorbar(surface, ax=ax, location="right", pad=0.1)
            surface_colorbar.set_label("Kernel Response")

            # Add contour labels
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

            # Configure axes
            ax.set_title("Kernel Response Surface")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

            # Custom legend
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    label="Active Point",
                    markersize=10,
                    markeredgecolor="black",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    label="Training Point",
                    markersize=10,
                    markeredgecolor="black",
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

            if self.debug_mode:
                print("Initial surface and points plotted")
                print(f"Axes limits: x=[{ax.get_xlim()}], y=[{ax.get_ylim()}]")

            return [surface, points_pos, points_neg, contours]

        def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
            if self.debug_mode and frame % 10 == 0:
                print(f"\nUpdating frame {frame}")

            surface, points_pos, points_neg, contours = artists
            alphas = logs["alphas"][frame]["alphas"]

            # Initialize response grid
            response = np.zeros_like(xx, dtype=np.float64)
            grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten grid

            # Compute kernel response for each active alpha in current frame
            active_indices = np.where(np.abs(alphas) > 1e-10)[0]
            for i in active_indices:
                # Compute kernel response point-wise
                response += alphas[i] * np.array(
                    [
                        kernel(xs[i], grid_point, **kernel_params)
                        for grid_point in grid_points
                    ],
                ).reshape(xx.shape)

            # Normalize the response
            response = normalize_response(response)

            # Clear existing contour collections but preserve scatter plot
            for coll in ax.collections[:]:
                if not isinstance(
                    coll,
                    plt.matplotlib.collections.PathCollection,
                ):  # Scatter plots are PathCollections
                    coll.remove()

            if self.debug_mode and frame % 10 == 0:
                print(f"Active points: {len(active_indices)}")

            # Update visualization
            new_surface = ax.contourf(xx, yy, response, levels=20, cmap="PuOr")
            new_contours = ax.contour(
                xx,
                yy,
                response,
                levels=10,
                colors="k",
                alpha=0.2,
                linewidths=0.5,
            )
            ax.clabel(new_contours, inline=True, fontsize=8, fmt="%.1f")

            # Update scatter points dynamically based on alpha values
            points_pos.set_color(
                [
                    "red" if abs(alpha) > 1e-10 else "gray"
                    for alpha in alphas[true_labels == 1]
                ],
            )
            points_neg.set_color(
                [
                    "red" if abs(alpha) > 1e-10 else "gray"
                    for alpha in alphas[true_labels == -1]
                ],
            )

            ax.set_title(f"Kernel Response Surface - Iteration {frame + 1}")

            if self.debug_mode:
                print(f"Frame {frame}: Active support vectors = {len(active_indices)}")
                print("Normalized response range: [0, 1]")

            return [new_surface, points_pos, points_neg, new_contours]

        # Initial grid position will be updated by visualizer
        component = AnimationComponent(
            setup_func=setup,
            update_func=update,
            subplot_params={"gridspec": (0, 0)},
        )

        if self.debug_mode:
            print("Kernel response component initialized")

        return component

    # FIXME: Not sure if this is useful for kernelized perceptron animation since the kernel matrix does not change
    # FIXME: create_kernel_matrix_component is untested
    def create_kernel_matrix_component(
        self,
        logs: Dict[str, Any],
    ) -> AnimationComponent:
        """Shows the constant kernel matrix and current alpha-weighted values."""
        kernel_matrix = logs["kernel_matrix"]

        def setup(ax: Axes) -> List[Artist]:
            im = ax.imshow(kernel_matrix, cmap="viridis", aspect="equal")
            plt.colorbar(im, ax=ax)
            ax.set_title("Kernel Matrix & Alpha Contributions")
            ax.set_xticks(range(len(kernel_matrix)))
            ax.set_yticks(range(len(kernel_matrix)))

            # Add secondary axis for alpha weights
            ax2 = ax.twinx()
            (alpha_line,) = ax2.plot([], [], "r-", label="Current Alphas")
            ax2.set_ylabel("Alpha Values")
            ax2.legend()

            return [im, alpha_line]

        def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
            im, alpha_line = artists
            alphas = logs["alphas"][frame]["alphas"]

            # Show current alpha values alongside kernel matrix
            alpha_line.set_data(range(len(alphas)), alphas)
            ax.set_title(f"Kernel Matrix & Alphas - Iteration {frame}")

            return artists

        return AnimationComponent(
            setup_func=setup,
            update_func=update,
            subplot_params={"gridspec": (1, 1)},
        )

    def create_misclassification_tracker_component(
        self,
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
