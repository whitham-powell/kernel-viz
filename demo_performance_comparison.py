"""Demo script comparing original vs optimized kernel response visualization.

This script demonstrates the performance improvements achieved by the optimized
kernel response implementation, including:
- Vectorized kernel computations
- Pre-computed kernel matrices
- Lower resolution with interpolation
- Kernel caching
"""

import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from kernel_viz.algorithms.perceptron import PerceptronLogger, kernelized_perceptron
from kernel_viz.kernels.base import (
    linear_kernel,
    polynomial_kernel,
    rbf_gaussian_kernel,
)
from kernel_viz.visualization.component_factory import KernelResponseFactory
from kernel_viz.visualization.optimized_kernel_response import (
    OptimizedKernelResponseFactory,
)


def generate_spiral_data(
    n_samples: int = 100,
    noise: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate spiral dataset for testing."""
    np.random.seed(42)
    n = n_samples // 2

    # Generate spiral coordinates
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
    r_pos = theta + np.pi
    r_neg = theta

    # Positive class spiral
    x_pos = r_pos * np.cos(theta) + noise * np.random.randn(n)
    y_pos = r_pos * np.sin(theta) + noise * np.random.randn(n)

    # Negative class spiral
    x_neg = r_neg * np.cos(theta) + noise * np.random.randn(n)
    y_neg = r_neg * np.sin(theta) + noise * np.random.randn(n)

    X = np.vstack([np.column_stack([x_pos, y_pos]), np.column_stack([x_neg, y_neg])])
    y = np.hstack([np.ones(n), -np.ones(n)])

    return X, y


def time_visualization(
    factory: Any,
    logs: Dict[str, Any],
    n_frames: int = 10,
    title: str = "Component",
) -> float:
    """Time the visualization performance."""
    component = factory.create()

    fig, ax = plt.subplots(figsize=(6, 6))
    artists = component.setup_func(ax)

    start_time = time.time()
    for frame in range(n_frames):
        component.update_func(frame % len(logs["alphas"]), ax, artists)
        plt.pause(0.001)  # Force rendering

    elapsed = time.time() - start_time

    ax.set_title(f"{title}\nTime: {elapsed:.2f}s for {n_frames} frames")

    return elapsed


def compare_implementations(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Any,
    kernel_params: Dict[str, Any],
) -> List[Tuple[str, float, int]]:
    """Compare original and optimized implementations."""
    # Train model
    logger = PerceptronLogger()
    kernelized_perceptron(
        X,
        y,
        kernel,
        kernel_params=kernel_params,
        max_iter=20,
        logger=logger,
    )
    logs = logger.get_logs()

    print(f"\nTesting with {kernel.__name__} kernel on {len(X)} samples...")
    print(f"Number of iterations: {len(logs['alphas'])}")

    # Create figure for comparison
    plt.figure(figsize=(15, 10))

    # Test different configurations
    configs = [
        ("Original (50x50 grid)", KernelResponseFactory(logs), 50),
        (
            "Optimized (30x30 + interpolation)",
            OptimizedKernelResponseFactory(
                logs,
                grid_resolution=30,
                use_interpolation=True,
                cache_kernels=True,
            ),
            30,
        ),
        (
            "Optimized (20x20 + interpolation)",
            OptimizedKernelResponseFactory(
                logs,
                grid_resolution=20,
                use_interpolation=True,
                cache_kernels=True,
            ),
            20,
        ),
        (
            "Optimized (50x50 no cache)",
            OptimizedKernelResponseFactory(
                logs,
                grid_resolution=50,
                use_interpolation=False,
                cache_kernels=False,
            ),
            50,
        ),
    ]

    times = []

    for i, (name, factory, grid_res) in enumerate(configs):
        plt.subplot(2, 2, i + 1)

        # Time the visualization
        elapsed = time_visualization(factory, logs, n_frames=10, title=name)
        times.append((name, elapsed, grid_res))

        print(f"{name}: {elapsed:.3f}s")

    plt.tight_layout()
    return times


def create_performance_summary(
    all_times: Dict[str, List[Tuple[str, float, int]]],
) -> None:
    """Create a summary plot of performance results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Extract data for plotting
    kernels = list(all_times.keys())
    configs = [t[0] for t in all_times[kernels[0]]]

    # Create bar chart
    x = np.arange(len(configs))
    width = 0.25

    for i, kernel in enumerate(kernels):
        times = [t[1] for t in all_times[kernel]]
        ax1.bar(x + i * width, times, width, label=kernel)

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Performance Comparison by Configuration")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([c.split(" ")[0] for c in configs], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Create speedup chart
    for i, kernel in enumerate(kernels):
        original_time = all_times[kernel][0][1]  # First config is original
        speedups = [original_time / t[1] for t in all_times[kernel]]
        ax2.plot(configs, speedups, "o-", label=kernel, markersize=8)

    ax2.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Speedup Factor")
    ax2.set_title("Speedup Relative to Original Implementation")
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([c.split(" ")[0] for c in configs], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


def main() -> None:
    """Run the performance comparison demo."""
    print("Kernel Response Surface Performance Comparison")
    print("=" * 50)

    # Generate datasets of different sizes
    datasets = [
        ("Small (20 points)", *generate_spiral_data(20)),
        ("Medium (50 points)", *generate_spiral_data(50)),
        ("Large (100 points)", *generate_spiral_data(100)),
    ]

    kernels = [
        ("Linear", linear_kernel, {}),
        ("RBF", rbf_gaussian_kernel, {"sigma": 1.0}),
        ("Polynomial", polynomial_kernel, {"degree": 3, "c": 1.0}),
    ]

    all_times = {}

    for kernel_name, kernel, params in kernels:
        print(f"\n\n{kernel_name} Kernel Performance")
        print("-" * 40)

        # Test with medium dataset
        X, y = datasets[1][1], datasets[1][2]
        times = compare_implementations(X, y, kernel, params)
        all_times[kernel_name] = times

        plt.savefig(f"performance_comparison_{kernel_name.lower()}.png", dpi=150)
        plt.show()

    # Create summary
    create_performance_summary(all_times)
    plt.savefig("performance_summary.png", dpi=150)
    plt.show()

    # Print recommendations
    print("\n\nPerformance Optimization Recommendations:")
    print("-" * 50)
    print("1. For real-time visualization: Use 20x20 or 30x30 grid with interpolation")
    print(
        "2. For linear kernels: Pre-computed kernel matrix provides significant speedup",
    )
    print("3. For RBF kernels: Vectorized distance computation improves performance")
    print(
        "4. For large datasets: Lower grid resolution with interpolation maintains quality",
    )
    print("5. Caching provides benefits when animating through many frames")


if __name__ == "__main__":
    main()
