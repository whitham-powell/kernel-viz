# %% [markdown]
# # Kernelized Perceptron Usage Examples
#
# This notebook demonstrates how to use the kernelized perceptron implementation
# from the `kernel_viz` package. We'll explore different kernel functions and
# visualize the decision boundaries for various datasets.

# %% [markdown]
# ## Setup and Imports

from typing import Any, Union

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.preprocessing import StandardScaler

# Import our kernel visualizer components
from kernel_viz.algorithms.perceptron import PerceptronLogger, kernelized_perceptron
from kernel_viz.kernels.base import (
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    rbf_gaussian_kernel,
)
from kernel_viz.kernels.composite import additive_kernel
from kernel_viz.visualization.core import PerceptronVisualizer

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## Example 1: Linearly Separable Data
#
# First, let's start with a simple linearly separable dataset to verify our
# kernelized perceptron works correctly with a linear kernel.

# %%
# Generate linearly separable data
X_linear, y_linear = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42,
)

# Convert labels to {-1, 1}
y_linear = 2 * y_linear - 1

# Standardize the features
scaler = StandardScaler()
X_linear = scaler.fit_transform(X_linear)

# Train with linear kernel
logger_linear = PerceptronLogger()
alphas_linear = kernelized_perceptron(
    X_linear,
    y_linear,
    kernel=linear_kernel,
    kernel_params={},
    max_epochs=10,
    logger=logger_linear,
)

print(
    f"Training completed in {len(logger_linear.get_logs()['misclassification_counts'])} epochs",
)
print(
    f"Final misclassifications: {logger_linear.get_logs()['misclassification_counts'][-1]}",
)

# %% [markdown]
# ### Visualize Linear Kernel Results

# %%
# Create visualizer
visualizer = PerceptronVisualizer(logger_linear, X_linear, y_linear, "Linear Kernel")

# Plot the decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
visualizer.plot_2d_decision_boundary(ax=ax)
plt.title("Linear Kernel - Decision Boundary")
plt.show()

# %% [markdown]
# ## Example 2: XOR Problem (Non-linearly Separable)
#
# The XOR problem is a classic example where linear separation is impossible.
# We'll use different kernels to solve this problem.

# %%
# Generate XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([-1, 1, 1, -1])

# Add more samples around the XOR points for better visualization
n_samples_per_point = 20
X_xor_extended = []
y_xor_extended = []

for i in range(len(X_xor)):
    for _ in range(n_samples_per_point):
        noise = np.random.normal(0, 0.1, 2)
        X_xor_extended.append(X_xor[i] + noise)
        y_xor_extended.append(y_xor[i])

X_xor = np.array(X_xor_extended)
y_xor = np.array(y_xor_extended)

# %% [markdown]
# ### Try Different Kernels on XOR Problem

# %%
# Dictionary to store results for different kernels
xor_results = {}

# Test different kernels
kernels_to_test = [
    ("Linear", linear_kernel, {}),
    ("Polynomial (degree=2)", polynomial_kernel, {"degree": 2, "c": 1.0}),
    ("RBF (σ=0.5)", rbf_gaussian_kernel, {"sigma": 0.5}),
    ("Laplacian (γ=2.0)", laplacian_kernel, {"gamma": 2.0}),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, kernel, params) in enumerate(kernels_to_test):
    logger = PerceptronLogger()
    alphas = kernelized_perceptron(
        X_xor,
        y_xor,
        kernel=kernel,
        kernel_params=params,
        max_epochs=100,
        logger=logger,
    )

    xor_results[name] = {
        "logger": logger,
        "alphas": alphas,
        "final_misclass": logger.get_logs()["misclassification_counts"][-1],
    }

    # Visualize
    visualizer = PerceptronVisualizer(logger, X_xor, y_xor, name)
    visualizer.plot_2d_decision_boundary(ax=axes[idx])
    axes[idx].set_title(
        f"{name}\nFinal misclass: {xor_results[name]['final_misclass']}",
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Example 3: Moon Dataset
#
# The two moons dataset is another classic non-linearly separable problem.

# %%
# Generate moon dataset
X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)
y_moons = 2 * y_moons - 1  # Convert to {-1, 1}

# Standardize
X_moons = StandardScaler().fit_transform(X_moons)

# %% [markdown]
# ### Compare RBF Kernel with Different Sigma Values

# %%
# Test different sigma values for RBF kernel
sigma_values = [0.1, 0.5, 1.0, 2.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, sigma in enumerate(sigma_values):
    logger = PerceptronLogger()
    alphas = kernelized_perceptron(
        X_moons,
        y_moons,
        kernel=rbf_gaussian_kernel,
        kernel_params={"sigma": sigma},
        max_epochs=50,
        logger=logger,
    )

    visualizer = PerceptronVisualizer(logger, X_moons, y_moons, f"RBF (σ={sigma})")
    visualizer.plot_2d_decision_boundary(ax=axes[idx])

    final_misclass = logger.get_logs()["misclassification_counts"][-1]
    axes[idx].set_title(f"RBF Kernel (σ={sigma})\nFinal misclass: {final_misclass}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Example 4: Composite Kernels
#
# We can combine multiple kernels using additive or multiplicative composition.

# %%
# Generate circular dataset
X_circles, y_circles = make_circles(
    n_samples=200,
    noise=0.1,
    factor=0.5,
    random_state=42,
)
y_circles = 2 * y_circles - 1
X_circles = StandardScaler().fit_transform(X_circles)

# %% [markdown]
# ### Additive Kernel Combination


# %%
# Create an additive kernel combining polynomial and RBF
def poly_rbf_additive(
    x: np.ndarray,
    y: np.ndarray,
    **params: Any,
) -> Union[float, np.ndarray]:
    kernel_funcs = [
        lambda a, b: polynomial_kernel(a, b, degree=2, c=1.0),
        lambda a, b: rbf_gaussian_kernel(a, b, sigma=1.0),
    ]
    result = additive_kernel(x, y, kernel_funcs)
    return result  # type: ignore[no-any-return]


# Train with composite kernel
logger_additive = PerceptronLogger()
alphas_additive = kernelized_perceptron(
    X_circles,
    y_circles,
    kernel=poly_rbf_additive,
    kernel_params={},
    max_epochs=50,
    logger=logger_additive,
)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Additive kernel result
visualizer_add = PerceptronVisualizer(
    logger_additive,
    X_circles,
    y_circles,
    "Polynomial + RBF (Additive)",
)
visualizer_add.plot_2d_decision_boundary(ax=ax1)

# Compare with single RBF kernel
logger_rbf = PerceptronLogger()
alphas_rbf = kernelized_perceptron(
    X_circles,
    y_circles,
    kernel=rbf_gaussian_kernel,
    kernel_params={"sigma": 1.0},
    max_epochs=50,
    logger=logger_rbf,
)

visualizer_rbf = PerceptronVisualizer(logger_rbf, X_circles, y_circles, "RBF Only")
visualizer_rbf.plot_2d_decision_boundary(ax=ax2)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Example 5: Tracking Training Progress
#
# Let's visualize how the alpha values and misclassification counts evolve during training.

# %%
# Use a moderately complex dataset
X_complex, y_complex = make_moons(n_samples=300, noise=0.25, random_state=42)
y_complex = 2 * y_complex - 1
X_complex = StandardScaler().fit_transform(X_complex)

# Train with RBF kernel and track progress
logger_progress = PerceptronLogger()
alphas_progress = kernelized_perceptron(
    X_complex,
    y_complex,
    kernel=rbf_gaussian_kernel,
    kernel_params={"sigma": 0.8},
    max_epochs=30,
    logger=logger_progress,
)

# %% [markdown]
# ### Visualize Training Progress

# %%
logs = logger_progress.get_logs()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot misclassification counts over epochs
ax1.plot(logs["misclassification_counts"], marker="o")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Misclassification Count")
ax1.set_title("Training Progress: Misclassifications per Epoch")
ax1.grid(True, alpha=0.3)

# Plot alpha evolution for a subset of samples
alpha_history = np.array(logs["alphas"])
n_samples_to_plot = min(10, alpha_history.shape[1])
for i in range(n_samples_to_plot):
    ax2.plot(alpha_history[:, i], alpha=0.7, label=f"Sample {i}")

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Alpha Value")
ax2.set_title("Evolution of Alpha Values for First 10 Samples")
ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Example 6: Using the Visualizer's Animation Components
#
# The PerceptronVisualizer provides various animation components to create
# comprehensive visualizations of the training process.

# %%
# Generate a dataset for animation
X_anim, y_anim = make_moons(n_samples=100, noise=0.2, random_state=42)
y_anim = 2 * y_anim - 1
X_anim = StandardScaler().fit_transform(X_anim)

# Train with detailed logging
logger_anim = PerceptronLogger()
alphas_anim = kernelized_perceptron(
    X_anim,
    y_anim,
    kernel=rbf_gaussian_kernel,
    kernel_params={"sigma": 1.0},
    max_epochs=20,
    logger=logger_anim,
)

# Create visualizer
visualizer_anim = PerceptronVisualizer(
    logger_anim,
    X_anim,
    y_anim,
    "RBF Kernel Animation",
)

# %% [markdown]
# ### Single Component Animations

# %%
# Show decision boundary at specific epochs
epochs_to_show = [0, 5, 10, 19]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, epoch in enumerate(epochs_to_show):
    visualizer_anim.create_single_component_animation(
        component_type="decision_boundary",
        epoch=epoch,
        ax=axes[idx],
    )
    axes[idx].set_title(f"Decision Boundary at Epoch {epoch}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Alpha Evolution Visualization

# %%
# Visualize alpha evolution
fig, ax = plt.subplots(figsize=(10, 6))
visualizer_anim.create_single_component_animation(
    component_type="alpha_evolution",
    ax=ax,
)
plt.show()

# %% [markdown]
# ## Example 7: Kernel Matrix Visualization
#
# Let's compute and visualize the kernel matrix for different kernels.

# %%
# Use a small dataset for clear visualization
X_small = X_xor[:20]  # Use first 20 points from XOR dataset
n_points = len(X_small)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

kernels_for_matrix = [
    ("Linear", linear_kernel, {}),
    ("Polynomial (d=2)", polynomial_kernel, {"degree": 2, "c": 1.0}),
    ("RBF (σ=0.5)", rbf_gaussian_kernel, {"sigma": 0.5}),
    ("Laplacian (γ=1.0)", laplacian_kernel, {"gamma": 1.0}),
]

for idx, (name, kernel, params) in enumerate(kernels_for_matrix):
    # Compute kernel matrix
    K = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            K[i, j] = kernel(X_small[i], X_small[j], **params)

    # Visualize
    im = axes[idx].imshow(K, cmap="viridis", aspect="auto")
    axes[idx].set_title(f"{name} Kernel Matrix")
    axes[idx].set_xlabel("Sample Index")
    axes[idx].set_ylabel("Sample Index")
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **Basic usage** of the kernelized perceptron with different kernel functions
# 2. **Solving non-linearly separable problems** like XOR and moon datasets
# 3. **Hyperparameter effects** (e.g., sigma in RBF kernel)
# 4. **Composite kernels** using additive and multiplicative combinations
# 5. **Training progress tracking** with misclassification counts and alpha evolution
# 6. **Visualization capabilities** including decision boundaries and kernel matrices
#
# The kernelized perceptron is a powerful algorithm that can solve non-linearly
# separable problems by implicitly mapping data to higher-dimensional feature spaces
# through the kernel trick.
