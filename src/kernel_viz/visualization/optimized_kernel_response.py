"""Optimized kernel response surface visualization.

This module provides performance-optimized implementations for kernel response
surface visualization, including vectorized computations and caching strategies.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from .component_factory import ComponentFactory


class OptimizedKernelResponseFactory(ComponentFactory):
    """Optimized factory for kernel response surface visualization.

    Key optimizations:
    1. Vectorized kernel computations using broadcasting
    2. Pre-computed kernel matrices for common kernels
    3. Lazy evaluation of grid points
    4. Cached kernel evaluations for static support vectors
    5. Reduced grid resolution with interpolation for smoother visuals
    """

    def __init__(
        self,
        logs: Dict[str, Any],
        debug_mode: bool = False,
        grid_resolution: int = 30,  # Lower default for better performance
        use_interpolation: bool = True,
        cache_kernels: bool = True,
    ):
        self.debug_mode = debug_mode
        self.grid_resolution = grid_resolution
        self.use_interpolation = use_interpolation
        self.cache_kernels = cache_kernels
        self._kernel_cache: Dict[Tuple[int, int], np.ndarray] = {}
        super().__init__(logs)

    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.xs = self.logs["feature_space"]
        self.kernel = self.logs["kernel"]
        self.kernel_params = self.logs["kernel_params"] or {}
        self.true_labels = self.logs["true_labels"]
        self.alphas_log = self.logs["alphas"]
        self.n_samples = len(self.xs)

        # Pre-calculate grid bounds
        margin = 1.0
        self.x_min = self.xs[:, 0].min() - margin
        self.x_max = self.xs[:, 0].max() + margin
        self.y_min = self.xs[:, 1].min() - margin
        self.y_max = self.xs[:, 1].max() + margin

        # Lazy grid initialization
        self._grid_initialized = False
        self.xx: Optional[np.ndarray] = None
        self.yy: Optional[np.ndarray] = None
        self.grid_points: Optional[np.ndarray] = None

        # Pre-compute normalization bounds
        self._compute_response_bounds()

    def _initialize_grid(self) -> None:
        """Lazily initialize computation grid."""
        if self._grid_initialized:
            return

        self.xx, self.yy = np.meshgrid(
            np.linspace(self.x_min, self.x_max, self.grid_resolution),
            np.linspace(self.y_min, self.y_max, self.grid_resolution),
        )
        self.grid_points = np.c_[self.xx.ravel(), self.yy.ravel()]
        self._grid_initialized = True

        # Pre-compute kernel matrix if caching is enabled
        if self.cache_kernels:
            self._precompute_kernel_matrix()

    def _precompute_kernel_matrix(self) -> None:
        """Pre-compute kernel evaluations between training points and grid."""
        assert self.grid_points is not None  # Ensured in _initialize_grid
        if self.debug_mode:
            print(
                f"Pre-computing kernel matrix ({self.n_samples} x {len(self.grid_points)})",
            )

        # Check if kernel supports vectorized operations
        kernel_name = self.kernel.__name__

        if kernel_name == "linear_kernel":
            # Vectorized linear kernel: K(x, y) = x @ y.T
            assert self.grid_points is not None  # Ensured in _initialize_grid
            self._kernel_matrix = self.xs @ self.grid_points.T

        elif kernel_name == "rbf_gaussian_kernel":
            # Vectorized RBF kernel: K(x, y) = exp(-||x-y||^2 / (2*sigma^2))
            sigma = self.kernel_params.get("sigma", 1.0)
            # Compute pairwise squared distances efficiently
            assert self.grid_points is not None  # Ensured in _initialize_grid
            X_sqnorms = np.sum(self.xs**2, axis=1, keepdims=True)
            Y_sqnorms = np.sum(self.grid_points**2, axis=1)
            distances_sq = X_sqnorms + Y_sqnorms - 2 * (self.xs @ self.grid_points.T)
            self._kernel_matrix = np.exp(-distances_sq / (2 * sigma**2))

        elif kernel_name == "polynomial_kernel":
            # Vectorized polynomial kernel: K(x, y) = (x @ y + c)^d
            assert self.grid_points is not None  # Ensured in _initialize_grid
            degree = self.kernel_params.get("degree", 3)
            c = self.kernel_params.get("c", 1.0)
            dot_products = self.xs @ self.grid_points.T
            self._kernel_matrix = (dot_products + c) ** degree

        else:
            # Fall back to loop-based computation for unknown kernels
            assert self.grid_points is not None  # Ensured in _initialize_grid
            self._kernel_matrix = np.zeros((self.n_samples, len(self.grid_points)))
            for i in range(self.n_samples):
                for j in range(len(self.grid_points)):
                    self._kernel_matrix[i, j] = self.kernel(
                        self.xs[i],
                        self.grid_points[j],
                        **self.kernel_params,
                    )

        if self.debug_mode:
            print("Kernel matrix pre-computation complete")

    def _compute_response_bounds(self) -> None:
        """Compute global response bounds for consistent visualization."""
        # Initialize grid for bounds computation
        self._initialize_grid()

        self.global_response_min = float("inf")
        self.global_response_max = float("-inf")

        for frame_data in self.alphas_log:
            alphas = frame_data["alphas"]

            # Skip frames with no support vectors
            active_mask = np.abs(alphas) > 1e-10
            if not np.any(active_mask):
                continue

            # Compute response using vectorized operations
            if hasattr(self, "_kernel_matrix"):
                # Use pre-computed kernel matrix
                response = np.dot(alphas, self._kernel_matrix)
            else:
                # Compute on-the-fly
                response = self._compute_response_vectorized(alphas)

            assert self.xx is not None  # Ensured in _initialize_grid
            response_surface = response.reshape(self.xx.shape)
            self.global_response_min = min(
                self.global_response_min,
                response_surface.min(),
            )
            self.global_response_max = max(
                self.global_response_max,
                response_surface.max(),
            )

        # Ensure valid range
        if self.global_response_max - self.global_response_min < 1e-10:
            self.global_response_min = -1
            self.global_response_max = 1

    def _compute_response_vectorized(self, alphas: np.ndarray) -> np.ndarray:
        """Compute response surface using vectorized operations."""
        assert self.grid_points is not None  # Ensured in _initialize_grid
        response = np.zeros(len(self.grid_points))

        # Only compute for non-zero alphas
        active_indices = np.where(np.abs(alphas) > 1e-10)[0]

        if len(active_indices) == 0:
            return response

        # Batch computation for active support vectors
        for idx in active_indices:
            cache_key = (int(idx), 0)
            if self.cache_kernels and cache_key in self._kernel_cache:
                # Use cached kernel values
                kernel_values = self._kernel_cache[cache_key]
            else:
                # Compute kernel values
                assert self.grid_points is not None  # Ensured in _initialize_grid
                kernel_values = np.array(
                    [
                        self.kernel(self.xs[idx], point, **self.kernel_params)
                        for point in self.grid_points
                    ],
                )

                if self.cache_kernels:
                    self._kernel_cache[cache_key] = kernel_values

            response += alphas[idx] * kernel_values

        return response

    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        # Ensure grid is initialized
        self._initialize_grid()

        if self.debug_mode:
            print("Setting up optimized kernel response surface")
            print(f"Grid resolution: {self.grid_resolution}x{self.grid_resolution}")
            print(f"Interpolation: {self.use_interpolation}")
            print(f"Kernel caching: {self.cache_kernels}")

        # Initial response surface (zeros)
        initial_surface = np.zeros_like(self.xx)

        if self.use_interpolation and self.grid_resolution < 50:
            # Use higher resolution for display via interpolation
            from scipy.interpolate import RectBivariateSpline

            x_display = np.linspace(self.x_min, self.x_max, 100)
            y_display = np.linspace(self.y_min, self.y_max, 100)
            xx_display, yy_display = np.meshgrid(x_display, y_display)

            # Interpolate to higher resolution
            interp = RectBivariateSpline(
                np.linspace(self.x_min, self.x_max, self.grid_resolution),
                np.linspace(self.y_min, self.y_max, self.grid_resolution),
                initial_surface,
            )
            surface_display = interp(x_display, y_display)

            surface = ax.contourf(
                xx_display,
                yy_display,
                surface_display,
                levels=20,
                cmap="RdBu_r",
                vmin=self.global_response_min,
                vmax=self.global_response_max,
            )

            # Store display grid for updates
            self._xx_display = xx_display
            self._yy_display = yy_display
        else:
            surface = ax.contourf(
                self.xx,
                self.yy,
                initial_surface,
                levels=20,
                cmap="RdBu_r",
                vmin=self.global_response_min,
                vmax=self.global_response_max,
            )

        # Add colorbar
        cbar = plt.colorbar(surface, ax=ax, pad=0.02)
        cbar.set_label("Kernel Response f(x)", rotation=270, labelpad=15)

        # Decision boundary
        decision_boundary = ax.contour(
            self.xx,
            self.yy,
            initial_surface,
            levels=[0],
            colors="black",
            linewidths=2,
            linestyles="--",
        )

        # Scatter plots for data points
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

        return [surface, decision_boundary, points_pos, points_neg]

    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        # Clear old contours efficiently
        while len(ax.collections) > 2:  # Keep scatter plots
            ax.collections[0].remove()

        # Get current alphas
        alphas = self.alphas_log[frame]["alphas"]

        # Compute response surface
        if hasattr(self, "_kernel_matrix"):
            # Use pre-computed kernel matrix (fastest)
            response = np.dot(alphas, self._kernel_matrix)
        else:
            # Use vectorized computation
            response = self._compute_response_vectorized(alphas)

        assert self.xx is not None  # Ensured in _initialize_grid
        response_surface = response.reshape(self.xx.shape)

        # Apply interpolation if enabled
        if self.use_interpolation and hasattr(self, "_xx_display"):
            from scipy.interpolate import RectBivariateSpline

            interp = RectBivariateSpline(
                np.linspace(self.x_min, self.x_max, self.grid_resolution),
                np.linspace(self.y_min, self.y_max, self.grid_resolution),
                response_surface,
            )
            x_display = np.linspace(self.x_min, self.x_max, 100)
            y_display = np.linspace(self.y_min, self.y_max, 100)
            response_display = interp(x_display, y_display)

            surface = ax.contourf(
                self._xx_display,
                self._yy_display,
                response_display,
                levels=20,
                cmap="RdBu_r",
                vmin=self.global_response_min,
                vmax=self.global_response_max,
            )

            decision_boundary = ax.contour(
                self._xx_display,
                self._yy_display,
                response_display,
                levels=[0],
                colors="black",
                linewidths=2,
                linestyles="--",
            )
        else:
            surface = ax.contourf(
                self.xx,
                self.yy,
                response_surface,
                levels=20,
                cmap="RdBu_r",
                vmin=self.global_response_min,
                vmax=self.global_response_max,
            )

            decision_boundary = ax.contour(
                self.xx,
                self.yy,
                response_surface,
                levels=[0],
                colors="black",
                linewidths=2,
                linestyles="--",
            )

        # Update support vector highlighting
        points_pos = artists[-2]
        points_neg = artists[-1]

        support_vector_mask_pos = np.abs(alphas[self.true_labels == 1]) > 1e-10
        support_vector_mask_neg = np.abs(alphas[self.true_labels == -1]) > 1e-10

        points_pos.set_sizes([120 if sv else 80 for sv in support_vector_mask_pos])
        points_neg.set_sizes([120 if sv else 80 for sv in support_vector_mask_neg])

        points_pos.set_edgecolors(
            ["yellow" if sv else "white" for sv in support_vector_mask_pos],
        )
        points_neg.set_edgecolors(
            ["yellow" if sv else "white" for sv in support_vector_mask_neg],
        )

        # Update title
        ax.set_title(f"Kernel Response Surface - Iteration {frame + 1}")

        return [surface, decision_boundary, points_pos, points_neg]


def benchmark_kernel_response(
    logs: Dict[str, Any],
    n_frames: int = 10,
) -> Dict[str, float]:
    """Benchmark original vs optimized kernel response implementations."""
    import time

    from .component_factory import KernelResponseFactory

    results = {}

    # Test original implementation
    start_time = time.time()
    factory_orig = KernelResponseFactory(logs)
    component_orig = factory_orig.create()

    fig, ax = plt.subplots()
    artists = component_orig.setup_func(ax)

    for frame in range(n_frames):
        component_orig.update_func(frame % len(logs["alphas"]), ax, artists)

    results["original"] = time.time() - start_time
    plt.close(fig)

    # Test optimized implementation with different settings
    configs: List[Tuple[str, Dict[str, Any]]] = [
        (
            "optimized_default",
            {"grid_resolution": 30, "use_interpolation": True, "cache_kernels": True},
        ),
        (
            "optimized_no_cache",
            {"grid_resolution": 30, "use_interpolation": True, "cache_kernels": False},
        ),
        (
            "optimized_high_res",
            {"grid_resolution": 50, "use_interpolation": False, "cache_kernels": True},
        ),
        (
            "optimized_minimal",
            {"grid_resolution": 20, "use_interpolation": True, "cache_kernels": True},
        ),
    ]

    for name, config in configs:
        start_time = time.time()
        factory_opt = OptimizedKernelResponseFactory(logs, **config)
        component_opt = factory_opt.create()

        fig, ax = plt.subplots()
        artists = component_opt.setup_func(ax)

        for frame in range(n_frames):
            component_opt.update_func(frame % len(logs["alphas"]), ax, artists)

        results[name] = time.time() - start_time
        plt.close(fig)

    return results
