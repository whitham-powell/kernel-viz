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

    def remove_component(self, component: AnimationComponent) -> None:
        raise NotImplementedError("remove_component() is not implemented yet")

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
        # Determine the number of frames from the logged data
        if "misclassification_count" in logs and logs["misclassification_count"]:
            self.total_frames = len(logs["misclassification_count"])
        elif "alphas" in logs and logs["alphas"]:
            self.total_frames = len(logs["alphas"])
        else:
            raise ValueError(
                "Cannot determine number of frames from logs. No misclassification_count or alphas data found.",
            )
        begin_animate_time = time.time()

        if len(self.components) == 0:
            raise ValueError("No components added to visualizer")

        # Validate frame count
        if self.total_frames is None or self.total_frames <= 0:
            raise ValueError(
                f"Animation requires valid number of frames. Got {self.total_frames} frames. "
                "Ensure the logger has recorded data during training.",
            )

        print(f"Starting animation with {self.total_frames} frames")

        plt.close("all")  # Close any existing figures
        self.figure = plt.figure(figsize=figsize)
        rows, cols = self._calculate_grid_dimensions()

        # Only specify width_ratios for multi-column layouts
        grid_params = {}
        if cols == 2:
            grid_params["width_ratios"] = [1, 1.2]

        self.grid_spec = GridSpec(
            rows,
            cols,
            figure=self.figure,
            **grid_params,
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

    def create_kernel_response_component(
        self,
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
        self,
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
