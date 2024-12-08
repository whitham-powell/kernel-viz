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
