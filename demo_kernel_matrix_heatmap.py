"""Demo script to showcase kernel matrix heatmap animation."""

import numpy as np
from kernel_viz.algorithms.perceptron import PerceptronLogger, kernelized_perceptron
from kernel_viz.kernels import rbf_gaussian_kernel
from kernel_viz.visualization import PerceptronVisualizer


def create_demo_data():
    """Create XOR dataset for demonstration."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([1, -1, -1, 1], dtype=np.float64)
    return X, y


def main():
    # Create data
    X, y = create_demo_data()
    
    # Train with logger
    logger = PerceptronLogger()
    kernel_params = {"sigma": 0.5}
    
    # Run training
    alphas = kernelized_perceptron(
        X, y, 
        kernel=rbf_gaussian_kernel,
        kernel_params=kernel_params,
        max_epochs=20,
        logger=logger
    )
    
    # Create visualizer and add kernel matrix heatmap component
    visualizer = PerceptronVisualizer()
    logs = logger.get_logs()
    
    # Add the new kernel matrix heatmap animation using factory pattern
    kernel_matrix_component = visualizer.create_kernel_matrix_heatmap_component(logs)
    visualizer.add_component(kernel_matrix_component)
    
    # Also add decision boundary for comparison
    decision_boundary_component = visualizer.create_decision_boundary_component(logs)
    visualizer.add_component(decision_boundary_component)
    
    # Generate animation
    animation = visualizer.animate(interval=500)
    animation.save("kernel_matrix_heatmap_demo.mp4", writer="ffmpeg", dpi=150)
    print("Animation saved as kernel_matrix_heatmap_demo.mp4")
    
    # Show static view
    visualizer.show()


if __name__ == "__main__":
    main()