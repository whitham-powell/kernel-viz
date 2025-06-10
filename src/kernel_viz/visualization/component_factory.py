"""Factory pattern implementation for visualization components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .base import AnimationComponent
from .core import compute_decision_boundary


class ComponentFactory(ABC):
    """Abstract base class for component factories."""

    def __init__(self, logs: Dict[str, Any]):
        """Initialize factory with log data."""
        self.logs = logs
        self._extract_data()

    @abstractmethod
    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        pass

    @abstractmethod
    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        pass

    @abstractmethod
    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        pass

    def create(self, **kwargs: Any) -> AnimationComponent:
        """Create the animation component."""
        return AnimationComponent(
            setup_func=self.setup,
            update_func=self.update,
            subplot_params=kwargs.get("subplot_params", {"gridspec": (0, 0)}),
            name=kwargs.get("name", self.__class__.__name__),
        )


class DecisionBoundaryFactory(ComponentFactory):
    """Factory for creating decision boundary visualization components."""

    def __init__(
        self,
        logs: Dict[str, Any],
        plot_type: Optional[str] = None,
        fixed_dims: Optional[Dict[int, float]] = None,
    ):
        self.plot_type = plot_type
        self.fixed_dims = fixed_dims
        super().__init__(logs)

    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.xs = self.logs["feature_space"]
        self.ys = self.logs["true_labels"]
        self.kernel = self.logs["kernel"]
        self.kernel_params = self.logs["kernel_params"] or {}

        # Determine plot type if not specified
        if self.plot_type is None:
            self.plot_type = (
                "line"
                if self.kernel.__name__ in ("linear_kernel", "affine_kernel")
                else "contour"
            )

    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        # Setup scatter plot
        scatter = ax.scatter(
            self.xs[:, 0],
            self.xs[:, 1],
            c=self.ys,
            cmap="bwr",
            edgecolor="k",
            zorder=2,
        )

        # Set margins
        margin = 0.1
        x_min, x_max = self.xs[:, 0].min(), self.xs[:, 0].max()
        y_min, y_max = self.xs[:, 1].min(), self.xs[:, 1].max()
        ax.set_xlim(
            [x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)],
        )
        ax.set_ylim(
            [y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)],
        )

        # Set initial title and labels
        ax.set_title("Decision Boundary - Iteration 1")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # Initialize line if using line plot
        if self.plot_type == "line":
            line = ax.plot([], [], "k-", lw=2)[0]
            return [scatter, line]
        return [scatter]

    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        # Clear old contours
        self._clear_old_contours(ax)

        # Compute new boundary
        alphas = self.logs["alphas"][frame]["alphas"]
        xx, yy, zz = compute_decision_boundary(
            self.xs,
            alphas,
            self.kernel,
            self.kernel_params,
            self.fixed_dims,
        )

        if self.plot_type == "line":
            return self._update_line_plot(ax, artists, xx, yy, zz)
        else:
            return self._update_contour_plot(ax, artists, xx, yy, zz, frame)

    def _clear_old_contours(self, ax: Axes) -> None:
        """Remove all contour collections from previous frame."""
        for artist in ax.collections[1:]:  # Keep scatter plot
            artist.remove()

    def _update_line_plot(
        self,
        ax: Axes,
        artists: List[Artist],
        xx: ArrayLike,
        yy: ArrayLike,
        zz: ArrayLike,
    ) -> List[Artist]:
        """Update the line plot showing decision boundary."""
        scatter, line = artists
        temp_contour = ax.contour(xx, yy, zz, levels=[0], colors="black")

        # Get paths from contour (handle deprecation)
        if hasattr(temp_contour, "collections"):
            collections = temp_contour.collections
            if collections[0].get_paths():
                vertices = collections[0].get_paths()[0].vertices
                line.set_data(vertices[:, 0], vertices[:, 1])
        else:
            # For newer matplotlib versions
            paths = temp_contour.get_paths()
            if paths and paths[0]:
                vertices = paths[0].vertices
                line.set_data(vertices[:, 0], vertices[:, 1])

        # Remove temporary contour
        if hasattr(temp_contour, "collections"):
            for coll in temp_contour.collections:
                coll.remove()
        else:
            temp_contour.remove()

        return [scatter, line]

    def _update_contour_plot(
        self,
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
        # Get contour collections (handle deprecation)
        if hasattr(contour, "collections"):
            contour_artists = list(contour.collections)
        else:
            # For newer matplotlib versions
            contour_artists = contour.get_paths()
        return [scatter] + contour_artists


class AlphaEvolutionFactory(ComponentFactory):
    """Factory for creating alpha evolution visualization components."""

    def __init__(self, logs: Dict[str, Any], debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.total_frames: Optional[int] = None  # Can be set externally if needed
        super().__init__(logs)

    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.alphas_history = self.logs["alphas"]
        self.n_samples = len(self.alphas_history[0]["alphas"])
        self.all_alphas = np.array([entry["alphas"] for entry in self.alphas_history])

        # Calculate global min/max for consistent y-axis
        min_alpha = np.min(self.all_alphas)
        max_alpha = np.max(self.all_alphas)
        alpha_range = max_alpha - min_alpha
        min_margin = 0.1
        margin = max(min_margin, alpha_range * 0.1)

        self.y_min = min_alpha - margin
        self.y_max = max_alpha + margin

    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        lines = []
        for i in range(self.n_samples):
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
                else len(self.alphas_history)
            ),
        )
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect(1)

        # Set labels and grid
        ax.set_title("Alpha Values Evolution")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Alpha Value")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Position legend
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=3,
            borderaxespad=0,
            fontsize="small",
        )

        return lines

    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        current_alphas = self.alphas_history[frame]["alphas"]
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
            y_data = [self.alphas_history[j]["alphas"][idx] for j in range(frame + 1)]

            # Update line data
            line.set_data(x_data, y_data)

            # Update line appearance based on current alpha value
            is_active = abs(current_alphas[idx]) > 1e-10
            line.set_color("red" if is_active else "gray")
            line.set_alpha(0.7 if is_active else 0.1)
            line.set_zorder(2 if is_active else 1)

            # Update legend line color
            if legend:
                legend_line = legend.get_lines()[idx]
                legend_line.set_color("red" if is_active else "gray")
                legend_line.set_alpha(0.7 if is_active else 0.1)

        # Update title with statistics
        n_active = np.sum(np.abs(current_alphas) > 1e-10)
        active_percentage = (n_active / self.n_samples) * 100

        ax.set_title(
            f"Alpha Values Evolution - Iteration {frame + 1}\n"
            f"Active points: {n_active}/{self.n_samples} ({active_percentage:.1f}%)",
        )

        return artists


class KernelMatrixHeatmapFactory(ComponentFactory):
    """Factory for creating kernel matrix heatmap animation components.

    This component visualizes the kernel matrix as a heatmap and animates
    the evolution of support vectors (non-zero alpha values) during training.
    """

    def __init__(self, logs: Dict[str, Any], debug_mode: bool = False):
        self.debug_mode = debug_mode
        super().__init__(logs)

    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.kernel_matrix = self.logs.get("kernel_matrix")
        self.xs = self.logs["feature_space"]
        self.kernel = self.logs["kernel"]
        self.kernel_params = self.logs["kernel_params"] or {}
        self.alphas_history = self.logs["alphas"]
        self.n_samples = len(self.xs)

        # Compute kernel matrix if not provided
        if self.kernel_matrix is None:
            self.kernel_matrix = np.zeros((self.n_samples, self.n_samples))
            for i in range(self.n_samples):
                for j in range(self.n_samples):
                    self.kernel_matrix[i, j] = self.kernel(
                        self.xs[i],
                        self.xs[j],
                        **self.kernel_params,
                    )

        # Pre-compute color normalization bounds
        self.vmin = -abs(self.kernel_matrix).max()
        self.vmax = abs(self.kernel_matrix).max()

        if self.debug_mode:
            print(f"Kernel matrix shape: {self.kernel_matrix.shape}")
            print(f"Kernel matrix range: [{self.vmin:.3f}, {self.vmax:.3f}]")

    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        # Create heatmap
        im = ax.imshow(
            self.kernel_matrix,
            cmap="RdBu_r",
            aspect="equal",
            vmin=self.vmin,
            vmax=self.vmax,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Kernel Value", rotation=270, labelpad=15)

        # Set ticks and labels for small datasets
        if self.n_samples <= 20:
            ax.set_xticks(range(self.n_samples))
            ax.set_yticks(range(self.n_samples))
            ax.set_xticklabels([f"{i}" for i in range(self.n_samples)], fontsize=8)
            ax.set_yticklabels([f"{i}" for i in range(self.n_samples)], fontsize=8)

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        ax.set_title("Kernel Matrix K(x_i, x_j) - Iteration 1")

        # Add grid for better readability
        ax.set_xticks(np.arange(self.n_samples) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n_samples) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)

        # Initialize support vector indicators
        # Create markers for diagonal (self-similarity) and off-diagonal separately
        self.sv_markers_diag = []
        self.sv_markers_row = []
        self.sv_markers_col = []

        for i in range(self.n_samples):
            # Diagonal marker (where i == j)
            marker_diag = ax.plot(
                i,
                i,
                "o",
                color="yellow",
                markersize=0,
                markeredgecolor="black",
                markeredgewidth=1,
                zorder=3,
            )[0]
            self.sv_markers_diag.append(marker_diag)

            # Row/column highlight markers
            row_markers = []
            col_markers = []
            for j in range(self.n_samples):
                if i != j:
                    # Row marker
                    marker_row = ax.plot(
                        j,
                        i,
                        "s",
                        color="yellow",
                        markersize=0,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        alpha=0.6,
                        zorder=2,
                    )[0]
                    row_markers.append(marker_row)

                    # Column marker
                    marker_col = ax.plot(
                        i,
                        j,
                        "s",
                        color="yellow",
                        markersize=0,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        alpha=0.6,
                        zorder=2,
                    )[0]
                    col_markers.append(marker_col)

            self.sv_markers_row.append(row_markers)
            self.sv_markers_col.append(col_markers)

        # Store all artists
        all_markers = [im] + self.sv_markers_diag
        for row in self.sv_markers_row:
            all_markers.extend(row)
        for col in self.sv_markers_col:
            all_markers.extend(col)

        return all_markers

    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        alphas = self.alphas_history[frame]["alphas"]

        # Update support vector indicators
        for i, alpha in enumerate(alphas):
            is_support_vector = abs(alpha) > 1e-10

            if is_support_vector:
                # Diagonal marker - larger size for support vectors
                marker_size = min(15, 5 + 10 * abs(alpha))
                color = "yellow" if alpha > 0 else "cyan"
                self.sv_markers_diag[i].set_markersize(marker_size)
                self.sv_markers_diag[i].set_color(color)

                # Row/column markers - smaller size
                small_marker_size = min(8, 3 + 5 * abs(alpha))
                for marker in self.sv_markers_row[i]:
                    marker.set_markersize(small_marker_size)
                    marker.set_color(color)
                for marker in self.sv_markers_col[i]:
                    marker.set_markersize(small_marker_size)
                    marker.set_color(color)
            else:
                # Hide markers for non-support vectors
                self.sv_markers_diag[i].set_markersize(0)
                for marker in self.sv_markers_row[i]:
                    marker.set_markersize(0)
                for marker in self.sv_markers_col[i]:
                    marker.set_markersize(0)

        # Update title with iteration info
        n_support = np.sum(np.abs(alphas) > 1e-10)
        ax.set_title(
            f"Kernel Matrix K(x_i, x_j) - Iteration {frame + 1}\n"
            f"Support Vectors: {n_support}/{self.n_samples}",
        )

        if self.debug_mode and frame % 10 == 0:
            print(f"Frame {frame}: {n_support} support vectors")

        return artists


class KernelResponseFactory(ComponentFactory):
    """Factory for creating kernel response surface visualization components.

    This component visualizes how the decision function f(x) = sum(alpha_i * K(x_i, x))
    evolves across the feature space, showing the complete response surface
    with a colorbar to indicate the magnitude of the response.
    """

    def __init__(self, logs: Dict[str, Any], debug_mode: bool = False):
        self.debug_mode = debug_mode
        super().__init__(logs)

    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.xs = self.logs["feature_space"]
        self.kernel = self.logs["kernel"]
        self.kernel_params = self.logs["kernel_params"] or {}
        self.true_labels = self.logs["true_labels"]
        self.alphas_log = self.logs["alphas"]

        # Pre-calculate grid points for response surface
        margin = 1.0  # Margin around data points
        self.x_min = self.xs[:, 0].min() - margin
        self.x_max = self.xs[:, 0].max() + margin
        self.y_min = self.xs[:, 1].min() - margin
        self.y_max = self.xs[:, 1].max() + margin
        self.grid_resolution = 50  # Balance between smoothness and performance

        self.xx, self.yy = np.meshgrid(
            np.linspace(self.x_min, self.x_max, self.grid_resolution),
            np.linspace(self.y_min, self.y_max, self.grid_resolution),
        )

        self.grid_points = np.c_[self.xx.ravel(), self.yy.ravel()]  # Flatten grid

        # Pre-compute global normalization bounds for consistent colorbar
        self.global_response_min = float("inf")
        self.global_response_max = float("-inf")

        for frame_data in self.alphas_log:
            alphas = frame_data["alphas"]
            response = np.zeros(len(self.grid_points), dtype=np.float64)
            for i, alpha in enumerate(alphas):
                if abs(alpha) > 1e-10:
                    kernel_values = np.array(
                        [
                            self.kernel(self.xs[i], grid_point, **self.kernel_params)
                            for grid_point in self.grid_points
                        ],
                        dtype=np.float64,
                    )
                    response += alpha * kernel_values
            response_reshaped = response.reshape(self.xx.shape)
            self.global_response_min = min(
                self.global_response_min,
                response_reshaped.min(),
            )
            self.global_response_max = max(
                self.global_response_max,
                response_reshaped.max(),
            )

        # Ensure we have a valid range
        if self.global_response_max - self.global_response_min < 1e-10:
            self.global_response_min = -1
            self.global_response_max = 1

        if self.debug_mode:
            print("\nInitializing Kernel Response Component:")
            print(f"Feature space shape: {self.xs.shape}")
            print(f"Grid resolution: {self.grid_resolution}x{self.grid_resolution}")
            print(f"X range: [{self.x_min:.2f}, {self.x_max:.2f}]")
            print(f"Y range: [{self.y_min:.2f}, {self.y_max:.2f}]")
            print(
                f"Response range: [{self.global_response_min:.2f}, {self.global_response_max:.2f}]",
            )
            print(f"Kernel: {self.kernel.__name__}")
            print(f"Kernel params: {self.kernel_params}")

    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        if self.debug_mode:
            print("Setting up kernel response surface component")

        # Initial response surface
        surface = ax.contourf(
            self.xx,
            self.yy,
            np.zeros_like(self.xx),
            levels=20,
            cmap="RdBu_r",
            vmin=self.global_response_min,
            vmax=self.global_response_max,
        )

        # Add colorbar
        cbar = plt.colorbar(surface, ax=ax, pad=0.02)
        cbar.set_label("Kernel Response f(x)", rotation=270, labelpad=15)

        # Decision boundary line
        decision_boundary = ax.contour(
            self.xx,
            self.yy,
            np.zeros_like(self.xx),
            levels=[0],
            colors="black",
            linewidths=2,
            linestyles="--",
        )

        # Points with markers for positive/negative classes
        points_pos = ax.scatter(
            self.xs[self.true_labels == 1, 0],
            self.xs[self.true_labels == 1, 1],
            c="blue",
            s=80,
            marker="o",
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
            label="Class +1",
        )
        points_neg = ax.scatter(
            self.xs[self.true_labels == -1, 0],
            self.xs[self.true_labels == -1, 1],
            c="red",
            s=80,
            marker="s",
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
            label="Class -1",
        )

        # Configuration
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("Kernel Response Surface - Iteration 1")
        ax.legend(loc="upper right")

        # Store the colorbar mappable for updates
        self._colorbar_mappable = surface

        # Return the contour objects, not their collections
        return [surface, decision_boundary, points_pos, points_neg]

    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        # Clear old contours
        for artist in ax.collections[:-2]:  # Keep scatter plots
            artist.remove()

        # Compute response surface for current alphas
        alphas = self.alphas_log[frame]["alphas"]
        response = np.zeros(len(self.grid_points), dtype=np.float64)

        for i, alpha in enumerate(alphas):
            if abs(alpha) > 1e-10:
                kernel_values = np.array(
                    [
                        self.kernel(self.xs[i], grid_point, **self.kernel_params)
                        for grid_point in self.grid_points
                    ],
                    dtype=np.float64,
                )
                response += alpha * kernel_values

        response_surface = response.reshape(self.xx.shape)

        # Update surface
        surface = ax.contourf(
            self.xx,
            self.yy,
            response_surface,
            levels=20,
            cmap="RdBu_r",
            vmin=self.global_response_min,
            vmax=self.global_response_max,
        )

        # Update decision boundary
        decision_boundary = ax.contour(
            self.xx,
            self.yy,
            response_surface,
            levels=[0],
            colors="black",
            linewidths=2,
            linestyles="--",
        )

        # Highlight support vectors
        points_pos = artists[-2]
        points_neg = artists[-1]

        # Update support vector highlighting
        support_vector_mask_pos = np.abs(alphas[self.true_labels == 1]) > 1e-10
        support_vector_mask_neg = np.abs(alphas[self.true_labels == -1]) > 1e-10

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

        # Update title (keep simple to match original)
        ax.set_title(f"Kernel Response Surface - Iteration {frame + 1}")

        # Return the contour objects and scatter plots
        return [surface, decision_boundary, points_pos, points_neg]


class KernelMatrixFactory(ComponentFactory):
    """Factory for creating kernel matrix visualization with alpha overlay.

    This component shows the kernel matrix heatmap with alpha values overlay.
    Since the kernel matrix is constant during training, this component
    focuses on showing how alpha values evolve relative to the kernel structure.
    """

    def __init__(self, logs: Dict[str, Any], debug_mode: bool = False):
        self.debug_mode = debug_mode
        super().__init__(logs)

    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.kernel_matrix = self.logs.get("kernel_matrix")
        self.xs = self.logs["feature_space"]
        self.kernel = self.logs["kernel"]
        self.kernel_params = self.logs["kernel_params"] or {}
        self.alphas_history = self.logs["alphas"]

        if self.kernel_matrix is None:
            # Compute kernel matrix if not provided
            n_samples = len(self.xs)
            self.kernel_matrix = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    self.kernel_matrix[i, j] = self.kernel(
                        self.xs[i],
                        self.xs[j],
                        **self.kernel_params,
                    )

        self.n_samples = len(self.kernel_matrix)

    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        # Create a more informative visualization
        assert self.kernel_matrix is not None  # Ensured in _extract_data
        im = ax.imshow(
            self.kernel_matrix,
            cmap="RdBu_r",
            aspect="equal",
            vmin=-float(abs(self.kernel_matrix).max()),
            vmax=float(abs(self.kernel_matrix).max()),
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Kernel Value", rotation=270, labelpad=15)

        # Set ticks and labels
        if self.n_samples <= 20:  # Only show individual labels for small datasets
            ax.set_xticks(range(self.n_samples))
            ax.set_yticks(range(self.n_samples))
            ax.set_xticklabels([f"{i}" for i in range(self.n_samples)], fontsize=8)
            ax.set_yticklabels([f"{i}" for i in range(self.n_samples)], fontsize=8)

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        ax.set_title("Kernel Matrix K(x_i, x_j)")

        # Add grid for better readability
        ax.set_xticks(np.arange(self.n_samples) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n_samples) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)

        # Initialize alpha indicators
        alpha_indicators = []
        for i in range(self.n_samples):
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

    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        alpha_indicators = artists[1:]
        alphas = self.alphas_history[frame]["alphas"]

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


class MisclassificationTrackerFactory(ComponentFactory):
    """Factory for creating misclassification tracker visualization.

    This component visualizes misclassified training points dynamically
    during perceptron updates.
    """

    def __init__(self, logs: Dict[str, Any], debug_mode: bool = False):
        self.debug_mode = debug_mode
        super().__init__(logs)

    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.xs = self.logs["feature_space"]  # Training points
        self.true_labels = self.logs["true_labels"]  # Ground-truth labels (+1 or -1)
        self.alphas_log = self.logs["alphas"]  # Alpha values for each iteration
        self.kernel = self.logs["kernel"]  # Kernel function
        self.kernel_params = self.logs["kernel_params"] or {}  # Kernel parameters

    def setup(self, ax: Axes) -> List[Artist]:
        """Setup the misclassification tracker visualization."""
        if self.debug_mode:
            print("Setting up misclassification tracker")

        # Initial scatter plot for all training points
        points = ax.scatter(
            self.xs[:, 0],
            self.xs[:, 1],
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

    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update the visualization for the given iteration."""
        points = artists[0]
        alphas = self.alphas_log[frame]["alphas"]

        # Compute the decision function for all training points
        decision_function = np.zeros(len(self.xs))
        for i in range(len(self.xs)):
            decision_function[i] = np.sum(
                [
                    alphas[j]
                    * self.kernel(self.xs[j], self.xs[i], **self.kernel_params)
                    for j in range(len(self.xs))
                ],
            )

        # Identify misclassified points
        misclassified = (self.true_labels * decision_function) < 0

        # Update colors: Red for misclassified, Gray for correctly classified
        colors = ["red" if m else "gray" for m in misclassified]
        points.set_color(colors)

        # Update title (keep it simple to match original)
        # The original implementation doesn't update the title per frame

        return [points]


# Factory method to create components
def create_component(
    component_type: str,
    logs: Dict[str, Any],
    **kwargs: Any,
) -> AnimationComponent:
    """Factory method to create visualization components.

    Args:
        component_type: Type of component to create
        logs: Training logs data
        **kwargs: Additional arguments for specific component types

    Returns:
        AnimationComponent instance
    """
    factories = {
        "decision_boundary": DecisionBoundaryFactory,
        "alpha_evolution": AlphaEvolutionFactory,
        "kernel_matrix_heatmap": KernelMatrixHeatmapFactory,
        "kernel_response": KernelResponseFactory,
        "kernel_matrix": KernelMatrixFactory,
        "misclassification_tracker": MisclassificationTrackerFactory,
    }

    if component_type not in factories:
        raise ValueError(f"Unknown component type: {component_type}")

    factory_class = factories[component_type]
    factory = factory_class(logs, **kwargs)
    component: AnimationComponent = factory.create()
    return component
