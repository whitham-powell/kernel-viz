# kernel_visualizer.py
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import Animation, FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray
from typing_extensions import TypeAlias

from src.kernelized_perceptron import PerceptronLogger

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


def setup_animation(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    ax: plt.Axes,
) -> PathCollection:
    """
    Setup initial plot elements.

    Parameters:
        xs: Input features array of shape (n_samples, n_features)
        ys: Target labels array of shape (n_samples,)
        ax: Matplotlib axes object for plotting

    Returns:
        PathCollection: The scatter plot object
    """
    scatter = ax.scatter(xs[:, 0], xs[:, 1], c=ys, cmap="bwr", edgecolor="k", zorder=2)

    margin = 0.1
    x_min, x_max = xs[:, 0].min(), xs[:, 0].max()
    y_min, y_max = xs[:, 1].min(), xs[:, 1].max()

    ax.set_xlim([x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)])
    ax.set_ylim([y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)])

    return scatter


def create_update_func(
    xs: NDArray[np.float64],
    ax: Axes,
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
    kernel_params: Dict[str, Any],
    alphas_logs: List[Dict[str, NDArray[np.float64]]],
    plot_type: str,
    fixed_dims: Optional[Dict[int, float]],
) -> Callable[[int], Union[List[Line2D], List[PathCollection]]]:
    """
    Creates the update function for the animation.

    Parameters:
        xs: Input features array
        ax: Matplotlib axes object
        kernel: Kernel function for decision boundary computation
        kernel_params: Parameters for the kernel function
        alphas_logs: List of dictionaries containing alpha values for each frame
        plot_type: Type of visualization ('line' or 'contour')
        fixed_dims: Dictionary of fixed dimensions

    Returns:
        Callable: Update function for the animation
    """
    contour: Optional[QuadContourSet] = None
    line: Optional[Line2D] = None

    if plot_type == "line":
        (line,) = ax.plot([], [], "k-", lw=2)

    def update(frame: int) -> Union[List[Line2D], List[PathCollection]]:
        nonlocal contour, line
        alphas = alphas_logs[frame]["alphas"]

        # Compute decision boundary
        xx, yy, zz = compute_decision_boundary(
            xs,
            alphas,
            kernel,
            kernel_params,
            fixed_dims,
        )

        # Update visualization based on plot type
        if plot_type == "line":
            if contour:
                for c in contour.collections:
                    c.remove()
            contour = ax.contour(xx, yy, zz, levels=[0], colors="black")
            if line is not None and contour.collections[0].get_paths():
                line_coords = contour.collections[0].get_paths()[0].vertices
                line.set_data(line_coords[:, 0], line_coords[:, 1])
                return [line]
            return []
        else:
            if contour:
                for c in contour.collections:
                    c.remove()
            contour = ax.contourf(
                xx,
                yy,
                zz,
                levels=[-1, 0, 1],
                alpha=0.3,
                cmap="coolwarm",
            )
            ax.set_title(f"Iteration {frame + 1}")
            return contour.collections  # type: ignore

    return update


def animate_decision_boundary(
    logger: PerceptronLogger,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    plot_type: Optional[str] = None,
    save_path: Optional[str] = None,
    fps: int = 10,
    fixed_dims: Optional[Dict[int, float]] = None,
) -> None:
    """
    Animate the decision boundary evolution.

    Parameters:
        logger: PerceptronLogger object containing training history
        xs: Input features
        ys: Target labels
        plot_type: Type of visualization ('line' or 'contour')
        save_path: Path to save the animation
        fps: Frames per second for the animation
        fixed_dims: Dictionary of fixed dimensions
    """
    # Get training history
    logs = logger.get_logs()
    alphas_logs = logs["alphas"]
    kernel = logs["kernel"]
    kernel_params = logs["kernel_params"]

    # Determine plot type based on kernel
    if plot_type is None:
        plot_type = "line" if kernel.__name__ == "linear_kernel" else "contour"

    # Setup figure and initial plot
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_animation(xs, ys, ax)

    # Create animation
    update_func = create_update_func(
        xs,
        ax,
        kernel,
        kernel_params,
        alphas_logs,
        plot_type,
        fixed_dims,
    )

    ani = FuncAnimation(
        fig,
        update_func,
        frames=len(alphas_logs),
        repeat=False,
        blit=True,  # Enable blitting for better performance
    )

    # Save animation if requested
    if save_path:
        save_animation(ani, save_path, fps)

    plt.show()
    plt.close()


def save_animation(ani: FuncAnimation, save_path: str, fps: int) -> None:
    """Save animation to file."""
    file_extension = save_path.split(".")[-1].lower()

    if file_extension in ["mp4", "mov"]:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(save_path, writer=writer)
    elif file_extension == "gif":
        ani.save(save_path, writer="pillow", fps=fps)
    else:
        raise ValueError(
            f"Unsupported file extension: {file_extension}. Use 'mp4', 'mov', or 'gif'.",
        )


# TODO: refactor AnimationComponent class to PerceptronVisualizer file
@dataclass
class AnimationComponent:
    """Represents a single visualization component."""

    setup_func: Callable[[plt.Axes], List[Artist]]
    update_func: Callable[[int, plt.Axes, List[Artist]], List[Artist]]
    subplot_params: Dict[str, Any]


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

    def _setup_grid(self, fig: plt.Figure, n_components: int) -> plt.GridSpec:
        """Setup the grid layout based on number of components."""
        if self.debug_mode:
            print(f"Setting up grid with {n_components} components")

        if n_components == 1:
            return plt.GridSpec(1, 1, figure=fig)
        if n_components <= 2:
            return plt.GridSpec(1, 2, figure=fig)
        else:
            return plt.GridSpec(2, 2, figure=fig)

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

        # TODO add error handling for missing logs / use if raise
        assert (
            self.total_frames is not None and self.total_frames > 0
        ), f"Animation requires valid number of frames. self.total_frames={self.total_frames}"

        if self.debug_mode:
            print("Debug mode enabled")
            print(f"Starting animation with {self.total_frames} frames")

        # FIXME: Close any existing figures - added for debugging
        plt.close("all")  # Close any existing figures - added for debugging
        fig = plt.figure(figsize=figsize)

        # Determine grid layout based on number of components
        n_components = len(self.components)
        if n_components == 0:
            raise ValueError("No components added to visualizer")

        gs = self._setup_grid(fig, n_components)

        # FIXME remove before submission
        # elif n_components == 1:
        #     gs = plt.GridSpec(1, 1, figure=fig)
        #     self.components[0].subplot_params["gridspec"] = (0, 0)
        # elif n_components == 2:
        #     gs = plt.GridSpec(1, 2, figure=fig)
        #     self.components[0].subplot_params["gridspec"] = (0, 0)
        #     self.components[1].subplot_params["gridspec"] = (0, 1)
        # else:
        #     gs = plt.GridSpec(2, 2, figure=fig)

        # Initialize components
        component_artists = []
        for idx, component in enumerate(self.components):
            if self.debug_mode:
                print(f"Setting up component {idx}")

            ax = fig.add_subplot(gs[component.subplot_params["gridspec"]])
            try:
                artists = component.setup_func(ax)
                if self.debug_mode:
                    print(
                        f"Component {idx} setup completed with {len(artists)} artists",
                    )
                component_artists.append((component, ax, artists))

            except Exception as e:
                print(f"Error setting up component {idx}: {e}")
                raise

        def update(frame: int) -> List[Artist]:
            if self.debug_mode:
                start_time = time.time()
                assert (
                    self.total_frames is not None
                ), "self.total_frames is None in animate() -> update()"
                print(f"\nProcessing frame {frame} / {self.total_frames - 1}")

            all_artists = []

            for idx, (component, ax, artists) in enumerate(component_artists):
                try:
                    if self.debug_mode:
                        print(f"Updating component {idx}")
                    updated_artists = component.update_func(frame, ax, artists)
                    if not isinstance(updated_artists, list):
                        print(
                            f"Warning: Component {idx} returned non-list: type={type(updated_artists)}",
                        )
                        updated_artists = list(updated_artists)

                    all_artists.extend(updated_artists)
                except Exception as e:
                    print(f"Error updating component {idx} at frame {frame}: {e}")
                    raise

            # FIXME We'll need to access the animation object, so we might need to make it nonlocal
            # FIXME remove before submission
            # for component, ax, artists in component_artists:
            #     updated_artists = component.update_func(frame, ax, artists)
            #     all_artists.extend(updated_artists)

            if self.debug_mode:
                print(f"Frame {frame} completed in {time.time() - start_time:.3f}s")
                print(f"Returned {len(all_artists)} artists")

            # FIXME remove before submission
            # if debug:
            #     print(f"Next frame requested: {frame}")
            #     print(
            #         f"Alpha values at frame {frame}: {logs['alphas'][frame]['alphas']}",
            #     )

            #     print(f"Processing frame {frame}")
            #     print(f"\nUpdate called for frame {frame}")
            #     print(f"Time: {time.time()}")

            return all_artists

        # TODO should this be a class attribute?
        interval = 1000 / fps

        self._animation = FuncAnimation(
            fig,
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
        # ani = FuncAnimation(
        #     fig,
        #     update,
        #     frames=len(logs["alphas"]),
        #     repeat=False,
        #     blit=True,
        #     interval=200,
        # )
        # if debug:
        #     print(f"Animation configured with {len(logs['alphas'])} frames")
        #     print(f"Animation object: {ani}")  # See if animation is properly configured
        # if save_path:
        # save_animation(ani, save_path, fps)

        plt.tight_layout()
        if self.debug_mode:
            print(
                f"Animation configured with {self.total_frames} and ready for display",
            )

        return self._animation

        # # display(fig)
        # plt.show()
        # # plt.close()
        # return ani

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

        def clear_old_contours(ax: plt.Axes) -> None:
            """Remove all contour collections from previous frame."""
            for artist in ax.collections[1:]:  # Keep scatter plot
                artist.remove()

        def update_line_plot(
            ax: plt.Axes,
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
            ax: plt.Axes,
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
            ax.set_title(f"Iteration {frame + 1}")
            return [scatter] + list(contour.collections)

        def setup(ax: plt.Axes) -> List[Artist]:
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

            ax.set_title("Decision Boundary")

            # Initialize line if using line plot
            if plot_type == "line":
                line = ax.plot([], [], "k-", lw=2)[0]
                return [scatter, line]
            return [scatter]

        def update(frame: int, ax: plt.Axes, artists: List[Artist]) -> List[Artist]:
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
        debug: bool = False,
    ) -> AnimationComponent:
        """Visualizes how alpha values change over training iterations."""
        alphas_history = logs["alphas"]
        n_samples = len(alphas_history[0]["alphas"])

        print(f"Number of samples: {n_samples}")
        print(f"Number of iterations: {len(alphas_history)}")
        print(
            f"Sample alpha values from first iteration: {alphas_history[0]['alphas']}",
        )

        def setup(ax: plt.Axes) -> List[Artist]:
            lines = []
            for i in range(n_samples):
                (line,) = ax.plot(
                    [],
                    [],
                    label=f"Point {i}",
                    alpha=0.3,
                    linewidth=0.5,
                    color="gray",
                )
                lines.append(line)

            ax.set_autoscale_on(False)

            # Debug prints
            print("In setup:")
            print(f"Y-axis limits: {ax.get_ylim()}")
            print(f"Created {len(lines)} lines")

            ax.set_xlim(0, len(alphas_history))

            # Find min/max alpha values
            all_alphas = np.array([entry["alphas"] for entry in alphas_history])
            min_alpha = min(np.min(alphas) for alphas in all_alphas)
            max_alpha = max(np.max(alphas) for alphas in all_alphas)
            margin = (max_alpha - min_alpha) * 0.1
            print(f"all_alphas first few entries: {all_alphas[:2]}")
            print(f"min_alpha: {min_alpha}")
            print(f"max_alpha: {max_alpha}")
            print(f"margin: {margin}")
            print(f"Setting y limits to: ({min_alpha - margin}, {max_alpha + margin})")
            ax.set_ylim(min_alpha - margin, max_alpha + margin)

            ax.set_title("Alpha Values Evolution")
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Alpha Value")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            return lines

        def update(frame: int, ax: plt.Axes, artists: List[Artist]) -> List[Artist]:
            print(f"In update frame {frame}:")
            alphas = alphas_history[frame]["alphas"]

            # Update lines
            for i, line in enumerate(artists):
                x_data = range(frame + 1)
                y_data = [alphas_history[j]["alphas"][i] for j in range(frame + 1)]
                print(f"Point {i}: x_data={x_data}, y_data={y_data}")
                line.set_data(x_data, y_data)

                # Color based on current alpha value
                if abs(alphas[i]) > 1e-10:
                    line.set_color("red")
                    line.set_alpha(0.7)  # Make active points more visible
                else:
                    line.set_color("gray")
                    line.set_alpha(0.1)  # Make inactive points very faint

            # Calculate stats about non-zero alphas
            n_points = len(alphas)
            active_points = np.abs(alphas) > 1e-10
            n_active = np.sum(active_points)
            active_percentage = (n_active / n_points) * 100

            ax.set_title(
                f"Alpha Values Evolution - Iteration {frame}\n"
                f"Active points: {n_active}/{n_points} ({active_percentage:.1f}%)",
            )

            return artists

        return AnimationComponent(
            setup_func=setup,
            update_func=update,
            subplot_params={"gridspec": (1, 0)},
        )

    def create_kernel_response_component(
        self,
        logs: Dict[str, Any],
    ) -> AnimationComponent:
        """Shows how the kernel response changes with alpha values."""
        xs = logs["feature_space"]
        kernel = logs["kernel"]
        kernel_params = logs["kernel_params"] or {}

        def setup(ax: plt.Axes) -> List[Artist]:
            # Create meshgrid for visualization
            x_min, x_max = xs[:, 0].min() - 1, xs[:, 0].max() + 1
            y_min, y_max = xs[:, 1].min() - 1, xs[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 50),
                np.linspace(y_min, y_max, 50),
            )

            # Initial response surface
            surface = ax.contourf(xx, yy, np.zeros_like(xx), levels=20)
            points = ax.scatter(xs[:, 0], xs[:, 1], c="red", s=50, zorder=2)
            plt.colorbar(surface, ax=ax)
            ax.set_title("Kernel Response Surface")

            return [surface, points]

        def update(frame: int, ax: plt.Axes, artists: List[Artist]) -> List[Artist]:
            surface, points = artists
            alphas = logs["alphas"][frame]["alphas"]

            # Clear previous surface
            for collection in surface.collections:
                collection.remove()

            # Get grid points
            x_min, x_max = xs[:, 0].min() - 1, xs[:, 0].max() + 1
            y_min, y_max = xs[:, 1].min() - 1, xs[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 50),
                np.linspace(y_min, y_max, 50),
            )

            # Calculate response using constant kernel matrix
            response = np.zeros_like(xx)
            for i in range(len(xs)):
                if abs(alphas[i]) > 1e-10:
                    for i in range(xx.shape[0]):
                        for j in range(xx.shape[1]):

                            grid_point = np.array([xx[i, j], yy[i, j]])
                            response[i, j] += alphas[i] * kernel(
                                xs[i],
                                grid_point,
                                **kernel_params,
                            )

            new_surface = ax.contourf(xx, yy, response, levels=20)
            points.set_array(alphas)
            return [new_surface, points]

        return AnimationComponent(
            setup_func=setup,
            update_func=update,
            subplot_params={"gridspec": (1, 1)},
        )

    def create_kernel_matrix_component(
        self,
        logs: Dict[str, Any],
    ) -> AnimationComponent:
        """Shows the constant kernel matrix and current alpha-weighted values."""
        kernel_matrix = logs["kernel_matrix"]

        def setup(ax: plt.Axes) -> List[Artist]:
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

        def update(frame: int, ax: plt.Axes, artists: List[Artist]) -> List[Artist]:
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

        # FIXME: old code remove before submission

    # def animate(
    #     self,
    #     logs: Dict[str, Any],
    #     figsize: Tuple[float, float] = (15, 10),
    #     save_path: Optional[str] = None,
    #     fps: int = 10,
    # ) -> None:
    #     """Create and display/save the combined animation."""
    #     fig = plt.figure(figsize=figsize)

    #     # Adjust GridSpec based on number of components
    #     n_components = len(self.components)
    #     if n_components <= 2:
    #         gs = plt.GridSpec(1, 2, figure=fig)
    #     else:
    #         gs = plt.GridSpec(2, 2, figure=fig)

    #     component_artists = {}
    #     for component in self.components:
    #         ax = fig.add_subplot(gs[component.subplot_params["gridspec"]])
    #         artists = component.setup_func(ax)
    #         component_artists[component] = (ax, artists)

    #     def update(frame: int) -> List[Artist]:
    #         all_artists = []
    #         for component, (ax, artists) in component_artists.items():
    #             updated_artists = component.update_func(frame, ax, artists)
    #             all_artists.extend(updated_artists)
    #         return all_artists

    #     ani = FuncAnimation(
    #         fig,
    #         update,
    #         frames=len(logs["alphas"]),
    #         repeat=False,
    #         blit=True,
    #     )

    #     if save_path:
    #         save_animation(ani, save_path, fps)

    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()
