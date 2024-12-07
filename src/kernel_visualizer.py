# kernel_visualizer.py
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
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


# def animate_decision_boundary(
#     logger: PerceptronLogger,
#     xs: NDArray[np.float64],
#     ys: NDArray[np.float64],
#     plot_type: str,
#     save_path: Optional[str] = None,
#     fps: int = 10,
#     fixed_dims: Optional[Dict[int, float]] = None,
# ) -> None:
#     """Animate the decision boundary evolution."""
#     logs = logger.get_logs()
#     alphas_logs = logs["alphas"]
#     kernel = logs["kernel"]
#     kernel_params = logs["kernel_params"]

#     # Determine plot type if not specified
#     if plot_type is None:
#         kernel_name = kernel.__name__
#         plot_type = "line" if kernel_name == "linear_kernel" else "contour"

#     fig, ax = plt.subplots()
#     scatter = ax.scatter(  # noqa: F841
#         xs[:, 0],
#         xs[:, 1],
#         c=ys,
#         cmap="bwr",
#         edgecolor="k",
#     )

#     (line,) = ax.plot([], [], "k-", lw=2) if plot_type == "line" else None  # type: ignore
#     contour = None

#     def update(frame: int) -> Any:
#         nonlocal contour
#         alphas = alphas_logs[frame]["values"]

#         # For both linear and non-linear cases
#         xx, yy, zz = compute_decision_boundary(
#             xs,
#             alphas,
#             kernel,
#             kernel_params,
#             fixed_dims=fixed_dims,
#         )

#         if plot_type == "line":
#             contour = ax.contour(xx, yy, zz, levels=[0], colors="black")
#             path = contour.collections[0].get_paths()
#             if path:
#                 line_coords = path[0].vertices
#                 line.set_data(line_coords[:, 0], line_coords[:, 1])

#         elif plot_type == "contour":
#             if contour:
#                 for c in contour.collections:
#                     c.remove()
#             contour = ax.contourf(
#                 xx,
#                 yy,
#                 zz,
#                 levels=[-1, 0, 1],
#                 alpha=0.3,
#                 cmap="coolwarm",
#             )

#         ax.set_title(f"Iteration {frame + 1}")

#     ani = FuncAnimation(fig, update, frames=len(alphas_logs), repeat=False)

#     # Save animation if requested
#     if save_path:
#         file_extension = save_path.split(".")[-1]
#         if file_extension in ["mp4", "mov"]:
#             ani.save(save_path, fps=fps, extra_args=["-vcodec", "libx264"])
#         elif file_extension == "gif":
#             ani.save(save_path, fps=fps, writer="imagemagick")
#         else:
#             raise ValueError(
#                 f"Unsupported file extension: {file_extension}. Use 'mp4', 'mov', or 'gif'.",
#             )

#     plt.show()
