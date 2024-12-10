# presentation_demos.py

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

from src.kernel_visualizer import PerceptronVisualizer
from src.kernelized_perceptron import PerceptronLogger, kernelized_perceptron
from src.kernels import laplacian_kernel

# Generate some random 2D data
np.random.seed(42)  # for reproducibility
n_samples = 50  # try different numbers to see how visualization scales


# Generate a semi linearly separable dataset
xs, ys = make_classification(
    n_samples=n_samples,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=42,
)

# Add Gaussian noise to the data
noise = np.random.normal(scale=0.5, size=xs.shape)
xs += noise

# Convert labels from {0, 1} to {-1, +1}
ys = 2 * ys - 1

# Visualize the dataset
plt.scatter(xs[ys == 1][:, 0], xs[ys == 1][:, 1], label="Class +1", color="red")
plt.scatter(xs[ys == -1][:, 0], xs[ys == -1][:, 1], label="Class -1", color="blue")
plt.title("Semi-Separable Dataset with Noise")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

demo_kernel = laplacian_kernel
demo_kernel_params = {"gamma": 0.5}
demo_kernel_name = demo_kernel.__name__

# Train perceptron and log
logger = PerceptronLogger()
alphas = kernelized_perceptron(
    xs,
    ys,
    kernel=demo_kernel,  # type: ignore
    kernel_params=demo_kernel_params,
    max_iter=5,
    logger=logger,
)
demo_logs = logger.get_logs()

# Create visualization
visualizer = PerceptronVisualizer()

# Create component with name
decision_boundary = visualizer.create_decision_boundary_component(demo_logs)
decision_boundary.name = "Decision Boundary"  # Add name

# Create component with name
alpha_evolution = visualizer.create_alpha_evolution_component(demo_logs)
alpha_evolution.name = "Alpha Evolution"  # Add name

# Create component with name
kernel_response = visualizer.create_kernel_response_component(demo_logs)
kernel_response.name = "Kernel Response"  # Add name


# Add components in desired order
visualizer.add_component(alpha_evolution)
visualizer.add_component(decision_boundary)
visualizer.add_component(kernel_response)

anim = visualizer.animate(
    demo_logs,
    save_path=f"perceptron_demo_combined_{demo_kernel_name}.mp4",
    fps=2,
)
# %%
