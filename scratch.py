# %% 
import os
import pathlib
# os.chdir(pathlib.Path(__file__).parent.absolute())

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Optional, Union
from numpy.typing import NDArray, ArrayLike

from src.kernels import (
    affine_kernel,
    exponential_kernel,
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    quadratic_kernel,
    rbf_gaussian_kernel,
)

from src.kernelized_perceptron import kernelized_perceptron, predict, PerceptronLogger

def plot_decision_boundary(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    alphas: NDArray[np.float64],
    kernel: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]],
    kernel_params: Optional[Dict[str, Any]] = None,
):
    # Create a grid of points
    x_min, x_max = xs[:, 0].min() - 1, xs[:, 0].max() + 1
    y_min, y_max = xs[:, 1].min() - 1, xs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Evaluate the decision function on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([predict(xs, alphas, point, kernel, kernel_params) for point in grid_points])
    zz = zz.reshape(xx.shape)
    
    # Plot the data points
    plt.figure(figsize=(10, 8))
    plt.scatter(xs[:, 0], xs[:, 1], c=ys, cmap=plt.cm.Paired, edgecolor='k', s=100)
    
    # Plot the decision boundary
    plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='red')
    plt.title("Kernelized Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Example usage
# Generate example data
np.random.seed(42)
xs = np.random.randn(20, 2)
ys = np.sign(xs[:, 0] * xs[:, 1])

# Generate a linearly separable dataset
# np.random.seed(42)
# n_samples = 20
# # Class +1
# x1 = np.random.randn(int(n_samples/2), 2) + np.array([2, 2])
# # Class -1
# x2 = np.random.randn(int(n_samples/2), 2) + np.array([-2, -2])

# # Combine into one dataset
# xs = np.vstack((x1, x2))
# ys = np.hstack((np.ones(int(n_samples/2)), -np.ones(int(n_samples/2))))

# Define a simple linear kernel
def linear_kernel(x, y, **kwargs):
    return np.dot(x, y)

# Fit the perceptron
logger = PerceptronLogger()
alphas = kernelized_perceptron(xs, ys, linear_kernel, {}, max_iter=10, logger=logger)

# Plot the result
plot_decision_boundary(xs, ys, alphas, linear_kernel, {})

# %%

from src.kernel_visualizer import animate_decision_boundary
from src.kernelized_perceptron import PerceptronLogger

animate_decision_boundary(logger, xs, ys, save_path="kernelized_perceptron.mp4", fps=5)

# %%
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

import numpy as np
from src.kernelized_perceptron import kernelized_perceptron, PerceptronLogger
from src.kernels import linear_kernel
from src.kernel_visualizer import PerceptronVisualizer

# Generate some random 2D data
# np.random.seed(42)  # for reproducibility
# n_samples = 20  # try different numbers to see how visualization scales

# # Create two clusters
# X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
# X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
# xs = np.vstack([X1, X2])

# # Create labels
# ys = np.array([1] * (n_samples // 2) + [-1] * (n_samples // 2))

np.random.seed(42)
xs = np.random.randn(20, 2)
ys = np.sign(xs[:, 0] * xs[:, 1])

# Train perceptron and log
logger = PerceptronLogger()
alphas = kernelized_perceptron(
    xs, 
    ys, 
    kernel=linear_kernel,
    max_iter=10,
    logger=logger
)

# Create visualization
visualizer = PerceptronVisualizer()
visualizer.add_component(
    visualizer.create_alpha_evolution_component(logger.get_logs(), debug=True)
)

# Let's also add some debug prints in the animate call
print("About to animate...")
anim = visualizer.animate(logger.get_logs(), save_path="perceptron_alpha_evolution.mp4", fps=5, debug=True)
plt.show(block=True)  # Make sure the plot stays visible

# %%

import numpy as np
import matplotlib.pyplot as plt
from src.kernelized_perceptron import PerceptronLogger, kernelized_perceptron
from src.kernel_visualizer import PerceptronVisualizer
from src.kernels import linear_kernel, quadratic_kernel, rbf_gaussian_kernel, polynomial_kernel, laplacian_kernel, exponential_kernel, affine_kernel

# %%
# Generate a linearly separable dataset
# np.random.seed(42)
# n_samples = 20
# # Class +1
# x1 = np.random.randn(int(n_samples/2), 2) + np.array([2, 2])
# # Class -1
# x2 = np.random.randn(int(n_samples/2), 2) + np.array([-2, -2])

# # Combine into one dataset
# xs = np.vstack((x1, x2))
# ys = np.hstack((np.ones(int(n_samples/2)), -np.ones(int(n_samples/2))))
#%%
import numpy as np
from sklearn.datasets import make_classification

# Generate a linearly separable dataset
xs, ys = make_classification(
    n_samples=50, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1, class_sep=1.0, random_state=42
)

# Add Gaussian noise to the data
noise = np.random.normal(scale=0.5, size=xs.shape)
xs += noise

# Convert labels from {0, 1} to {-1, +1}
ys = 2 * ys - 1

# Visualize the dataset
plt.scatter(xs[ys == 1][:, 0], xs[ys == 1][:, 1], label="Class +1", color="blue")
plt.scatter(xs[ys == -1][:, 0], xs[ys == -1][:, 1], label="Class -1", color="red")
plt.title("Semi-Separable Dataset with Noise")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# # Generate sample data
# np.random.seed(42)
# n_samples = 20
# xs = np.random.randn(n_samples, 2)
# ys = np.sign(xs[:, 0] * xs[:, 1])

# Train perceptron and log
logger = PerceptronLogger()
alphas = kernelized_perceptron(
    xs,
    ys,
    kernel=rbf_gaussian_kernel,
    max_iter=10,
    logger=logger
)
# %%
# Create visualization
visualizer = PerceptronVisualizer()
decision_boundary_component = visualizer.create_decision_boundary_component(logger.get_logs())
visualizer.add_component(decision_boundary_component)

# Test the setup and update functions
fig, ax = plt.subplots(figsize=(8, 6))
artists = decision_boundary_component.setup_func(ax)
updated_artists = decision_boundary_component.update_func(0, ax, artists)
# assert len(updated_artists) == len(artists), f"Expected {len(artists)} artists, got {len(updated_artists)}"

# Animate and save
print("About to animate decision boundary...")
anim = visualizer.animate(logger.get_logs(), save_path="perceptron_decision_boundary.mp4", fps=5, debug=True)
assert anim is not None
assert os.path.exists("perceptron_decision_boundary.mp4")

print("Test passed!")
# %%

# Create visualization
visualizer = PerceptronVisualizer()
alpha_evolution_component = visualizer.create_alpha_evolution_component(logger.get_logs())
visualizer.add_component(alpha_evolution_component)

# Test the setup and update functions
fig, ax = plt.subplots(figsize=(8, 6))
artists = alpha_evolution_component.setup_func(ax)
updated_artists = alpha_evolution_component.update_func(0, ax, artists)
assert len(updated_artists) == len(artists)

# Animate and save
print("About to animate alpha evolution...")
anim = visualizer.animate(logger.get_logs(), save_path="perceptron_alpha_evolution.mp4", fps=5, debug=True)
assert anim is not None
assert os.path.exists("perceptron_alpha_evolution.mp4")

print("Test passed!")
# %%

# Create visualization
visualizer = PerceptronVisualizer()
kernel_response_component = visualizer.create_kernel_response_component(logger.get_logs())
visualizer.add_component(kernel_response_component)

# Test the setup and update functions
fig, ax = plt.subplots(figsize=(8, 6))
artists = kernel_response_component.setup_func(ax)
updated_artists = kernel_response_component.update_func(0, ax, artists)
assert len(updated_artists) == len(artists)

# Animate and save
print("About to animate kernel response...")
anim = visualizer.animate(logger.get_logs(), save_path="perceptron_kernel_response.mp4", fps=5, debug=True)
assert anim is not None
assert os.path.exists("perceptron_kernel_response.mp4")

print("Test passed!")

# %%
# Create visualization
visualizer = PerceptronVisualizer()
misclassification_tracker_component = visualizer.create_misclassification_tracker_component(logger.get_logs())
visualizer.add_component(misclassification_tracker_component)

# Test the setup and update functions
fig, ax = plt.subplots(figsize=(8, 6))
artists = misclassification_tracker_component.setup_func(ax)
updated_artists = misclassification_tracker_component.update_func(0, ax, artists)
assert len(updated_artists) == len(artists)

# Animate and save
print("About to animate misclassification tracker...")
anim = visualizer.animate(logger.get_logs(), save_path="perceptron_misclassification_tracker.mp4", fps=5, debug=True)
assert anim is not None
assert os.path.exists("perceptron_misclassification_tracker.mp4")

print("Test passed!")

# %%
visualizer = PerceptronVisualizer()

# Create component with name
decision_boundary = visualizer.create_decision_boundary_component(logger.get_logs())
decision_boundary.name = "Decision Boundary"  # Add name

# Create component with name
alpha_evolution = visualizer.create_alpha_evolution_component(logger.get_logs())
alpha_evolution.name = "Alpha Evolution"  # Add name

# Create component with name
kernel_response = visualizer.create_kernel_response_component(logger.get_logs())
kernel_response.name = "Kernel Response"  # Add name

# Create component with name
misclassification_tracker = visualizer.create_misclassification_tracker_component(logger.get_logs())
misclassification_tracker.name = "Misclassification Tracker"  # Add name

# Add components in desired order
visualizer.add_component(alpha_evolution)
visualizer.add_component(decision_boundary)
visualizer.add_component(kernel_response)
# visualizer.add_component(misclassification_tracker)

anim = visualizer.animate(logger.get_logs(), save_path="perceptron_combined.mp4", fps=5, debug=True)
# %%
