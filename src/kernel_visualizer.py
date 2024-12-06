# kernel_visualizer.py
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import ArrayLike, NDArray

from src.kernelized_perceptron import PerceptronLogger


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


def animate_decision_boundary(
    logger: PerceptronLogger,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    plot_type: str,
    save_path: Optional[str] = None,
    fps: int = 10,
    fixed_dims: Optional[Dict[int, float]] = None,
) -> None:
    """Animate the decision boundary evolution."""
    logs = logger.get_logs()
    alphas_logs = logs["alphas"]
    kernel = logs["kernel"]
    kernel_params = logs["kernel_params"]

    # Determine plot type if not specified
    if plot_type is None:
        kernel_name = kernel.__name__
        plot_type = "line" if kernel_name == "linear_kernel" else "contour"

    fig, ax = plt.subplots()
    scatter = ax.scatter(  # noqa: F841
        xs[:, 0],
        xs[:, 1],
        c=ys,
        cmap="bwr",
        edgecolor="k",
    )
    (line,) = ax.plot([], [], "k-", lw=2) if plot_type == "line" else None  # type: ignore
    contour = None

    def update(frame: int) -> Any:
        nonlocal contour
        alphas = alphas_logs[frame]["values"]

        if plot_type == "line":
            # Compute weights and bias for the linear decision boundary
            w = np.sum(alphas[:, None] * ys[:, None] * xs, axis=0)
            b = np.mean(ys - np.dot(xs, w))
            x_min, x_max = xs[:, 0].min() - 1, xs[:, 0].max() + 1
            y_min = -(w[0] * x_min + b) / w[1]
            y_max = -(w[0] * x_max + b) / w[1]
            line.set_data([x_min, x_max], [y_min, y_max])
        elif plot_type == "contour":
            # Compute and plot the non-linear decision boundary
            xx, yy, zz = compute_decision_boundary(
                xs,
                alphas,
                kernel,
                kernel_params,
                fixed_dims=fixed_dims,
            )
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

    ani = FuncAnimation(fig, update, frames=len(alphas_logs), repeat=False)

    # Save animation if requested
    if save_path:
        file_extension = save_path.split(".")[-1]
        if file_extension in ["mp4", "mov"]:
            ani.save(save_path, fps=fps, extra_args=["-vcodec", "libx264"])
        elif file_extension == "gif":
            ani.save(save_path, fps=fps, writer="imagemagick")
        else:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Use 'mp4', 'mov', or 'gif'.",
            )

    plt.show()
