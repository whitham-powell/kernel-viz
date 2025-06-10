# Usage Examples

This guide provides comprehensive examples of using the kernel visualization framework for statistical learning.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Kernel Functions](#kernel-functions)
3. [Kernelized Perceptron](#kernelized-perceptron)
4. [Visualization Components](#visualization-components)
5. [Advanced Examples](#advanced-examples)

## Quick Start

### Basic Example: XOR Classification

```python
import numpy as np
from kernel_viz.algorithms.perceptron import kernelized_perceptron, PerceptronLogger
from kernel_viz.kernels import rbf_gaussian_kernel
from kernel_viz.visualization import PerceptronVisualizer

# Create XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([1, -1, -1, 1], dtype=np.float64)  # XOR labels

# Create logger to track training
logger = PerceptronLogger()

# Train kernelized perceptron
alphas = kernelized_perceptron(
    X, y,
    kernel=rbf_gaussian_kernel,
    kernel_params={'sigma': 0.5},
    max_epochs=20,
    logger=logger
)

# Visualize the results
visualizer = PerceptronVisualizer()
logs = logger.get_logs()

# Add decision boundary visualization
decision_boundary = visualizer.create_decision_boundary_component(logs)
visualizer.add_component(decision_boundary)

# Animate and save
animation = visualizer.animate(interval=500)
animation.save("xor_classification.gif", writer="pillow", fps=2)
```

## Kernel Functions

### Available Kernels

The framework provides several kernel functions:

```python
from kernel_viz.kernels import (
    linear_kernel,
    affine_kernel,
    quadratic_kernel,
    polynomial_kernel,
    rbf_gaussian_kernel,
    exponential_kernel,
    laplacian_kernel
)
```

### Kernel Usage Examples

#### Linear Kernel
```python
# k(x, y) = x · y
result = linear_kernel([1, 2], [3, 4])  # = 1*3 + 2*4 = 11
```

#### Polynomial Kernel
```python
# k(x, y) = (x · y + c)^d
result = polynomial_kernel([1, 2], [3, 4], degree=2, c=1.0)  # = (11 + 1)^2 = 144
```

#### RBF (Gaussian) Kernel
```python
# k(x, y) = exp(-||x - y||² / (2σ²))
result = rbf_gaussian_kernel([1, 2], [3, 4], sigma=1.0)
```

### Custom Kernel Example
```python
def custom_kernel(x, y, alpha=1.0):
    """Custom sigmoid-like kernel."""
    x_vec = np.asarray(x)
    y_vec = np.asarray(y)
    return np.tanh(alpha * np.dot(x_vec, y_vec))

# Use in training
alphas = kernelized_perceptron(X, y, kernel=custom_kernel, kernel_params={'alpha': 0.5})
```

## Kernelized Perceptron

### Basic Training

```python
# Linear classification
X_linear = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.float64)
y_linear = np.array([1, 1, -1, -1], dtype=np.float64)

# Train with linear kernel
alphas = kernelized_perceptron(X_linear, y_linear, kernel=linear_kernel)
```

### Training with Logger

```python
logger = PerceptronLogger()

alphas = kernelized_perceptron(
    X, y,
    kernel=polynomial_kernel,
    kernel_params={'degree': 3, 'c': 1.0},
    max_epochs=50,
    logger=logger
)

# Access training history
logs = logger.get_logs()
print(f"Final misclassification count: {logs['misclassification_count'][-1]}")
print(f"Number of support vectors: {np.sum(np.abs(alphas) > 1e-10)}")
```

### Making Predictions

```python
def predict(x_new, X_train, alphas, kernel, kernel_params=None):
    """Predict label for new point."""
    kernel_params = kernel_params or {}
    prediction = sum(
        alpha * kernel(X_train[i], x_new, **kernel_params)
        for i, alpha in enumerate(alphas) if abs(alpha) > 1e-10
    )
    return np.sign(prediction)

# Predict new point
x_new = np.array([0.5, 0.5])
label = predict(x_new, X, alphas, rbf_gaussian_kernel, {'sigma': 0.5})
```

## Visualization Components

### Multiple Visualization Components

```python
visualizer = PerceptronVisualizer()
logs = logger.get_logs()

# 1. Decision Boundary
decision_boundary = visualizer.create_decision_boundary_component(logs)
visualizer.add_component(decision_boundary)

# 2. Alpha Evolution
alpha_evolution = visualizer.create_alpha_evolution_component(logs)
visualizer.add_component(alpha_evolution)

# 3. Kernel Response Surface
kernel_response = visualizer.create_kernel_response_component(logs)
visualizer.add_component(kernel_response)

# 4. Kernel Matrix Heatmap (using factory pattern)
kernel_matrix = visualizer.create_kernel_matrix_heatmap_component(logs)
visualizer.add_component(kernel_matrix)

# Create animation with all components
animation = visualizer.animate(interval=300)
visualizer.show()  # Display final state
```

### Customizing Visualizations

```python
# Decision boundary for high-dimensional data
# Fix dimensions 2, 3, 4 while visualizing dimensions 0 and 1
component = visualizer.create_decision_boundary_component(
    logs,
    fixed_dims={2: 0.0, 3: 1.0, 4: -0.5}
)

# Force specific plot type
component = visualizer.create_decision_boundary_component(
    logs,
    plot_type='contour'  # or 'line' for linear kernels
)
```

## Advanced Examples

### Multi-class Classification (One-vs-All)

```python
from sklearn.datasets import make_classification

# Generate multi-class data
X, y = make_classification(
    n_samples=150, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=3, random_state=42
)

# Convert to -1, 0, 1 labels
y = y - 1

# Train one-vs-all classifiers
classifiers = []
for class_label in [-1, 0, 1]:
    # Create binary labels
    y_binary = np.where(y == class_label, 1, -1)

    logger = PerceptronLogger()
    alphas = kernelized_perceptron(
        X, y_binary,
        kernel=rbf_gaussian_kernel,
        kernel_params={'sigma': 1.0},
        max_epochs=50,
        logger=logger
    )

    classifiers.append({
        'class': class_label,
        'alphas': alphas,
        'logger': logger
    })

# Visualize each classifier
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, clf in enumerate(classifiers):
    visualizer = PerceptronVisualizer()
    logs = clf['logger'].get_logs()

    # Create decision boundary
    component = visualizer.create_decision_boundary_component(logs)

    # Custom setup for subplot
    plt.sca(axes[idx])
    axes[idx].set_title(f"Class {clf['class']} vs Rest")

    # Manually run setup and final frame
    artists = component.setup_func(axes[idx])
    component.update_func(len(logs['alphas']) - 1, axes[idx], artists)

plt.tight_layout()
plt.savefig("multiclass_classification.png")
```

### Composite Kernels

```python
from kernel_viz.kernels.composite import additive_kernel, multiplicative_kernel

# Combine multiple kernels
def create_composite_kernel():
    kernels = [
        (linear_kernel, {}),
        (rbf_gaussian_kernel, {'sigma': 1.0}),
    ]

    # Additive combination: k(x,y) = k1(x,y) + k2(x,y)
    return lambda x, y: additive_kernel(x, y, kernels)

# Train with composite kernel
composite_k = create_composite_kernel()
alphas = kernelized_perceptron(X, y, kernel=composite_k)
```

### Real-world Dataset Example

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare data
iris = load_iris()
X = iris.data[:100, :2]  # First two features, two classes
y = iris.target[:100]
y = np.where(y == 0, -1, 1)  # Convert to -1, 1

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train with different kernels
kernels_to_test = [
    ('Linear', linear_kernel, {}),
    ('Polynomial', polynomial_kernel, {'degree': 3, 'c': 1.0}),
    ('RBF', rbf_gaussian_kernel, {'sigma': 0.5}),
]

results = []
for name, kernel, params in kernels_to_test:
    logger = PerceptronLogger()

    alphas = kernelized_perceptron(
        X_train, y_train,
        kernel=kernel,
        kernel_params=params,
        max_epochs=100,
        logger=logger
    )

    # Test accuracy
    correct = 0
    for i in range(len(X_test)):
        pred = predict(X_test[i], X_train, alphas, kernel, params)
        if pred == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)
    results.append({
        'kernel': name,
        'accuracy': accuracy,
        'support_vectors': np.sum(np.abs(alphas) > 1e-10),
        'logger': logger
    })

    print(f"{name} Kernel: {accuracy:.2%} accuracy, "
          f"{results[-1]['support_vectors']} support vectors")
```

### Performance Optimization

```python
# Pre-compute kernel matrix for faster training
def precompute_kernel_matrix(X, kernel, kernel_params=None):
    """Pre-compute the kernel matrix for efficiency."""
    n = len(X)
    K = np.zeros((n, n))
    kernel_params = kernel_params or {}

    for i in range(n):
        for j in range(i, n):  # Exploit symmetry
            K[i, j] = kernel(X[i], X[j], **kernel_params)
            K[j, i] = K[i, j]

    return K

# Use in logger
K = precompute_kernel_matrix(X, rbf_gaussian_kernel, {'sigma': 0.5})
logger = PerceptronLogger()
logger.logs['kernel_matrix'] = K

# Train (will use pre-computed matrix if available)
alphas = kernelized_perceptron(
    X, y,
    kernel=rbf_gaussian_kernel,
    kernel_params={'sigma': 0.5},
    logger=logger
)
```

### Debugging and Analysis

```python
# Enable debug mode for detailed output
visualizer = PerceptronVisualizer()
visualizer.set_debug_mode(True)

# Analyze convergence
logs = logger.get_logs()
misclassifications = logs['misclassification_count']

plt.figure(figsize=(10, 5))
plt.plot(misclassifications)
plt.xlabel('Iteration')
plt.ylabel('Misclassification Count')
plt.title('Training Convergence')
plt.grid(True)
plt.show()

# Analyze alpha sparsity over time
alpha_history = [entry['alphas'] for entry in logs['alphas']]
sparsity = [np.sum(np.abs(alphas) > 1e-10) for alphas in alpha_history]

plt.figure(figsize=(10, 5))
plt.plot(sparsity)
plt.xlabel('Iteration')
plt.ylabel('Number of Support Vectors')
plt.title('Support Vector Evolution')
plt.grid(True)
plt.show()
```

## Tips and Best Practices

1. **Kernel Selection**:
   - Use linear kernel for linearly separable data
   - Use RBF kernel for non-linear boundaries
   - Start with default parameters, then tune

2. **Parameter Tuning**:
   - RBF: smaller σ → more complex boundaries
   - Polynomial: higher degree → more flexibility but risk of overfitting

3. **Visualization**:
   - Use multiple components for comprehensive analysis
   - Save animations for presentations
   - Use debug mode for troubleshooting

4. **Performance**:
   - Pre-compute kernel matrices for large datasets
   - Limit max_epochs for faster experimentation
   - Use sparse representations (only store non-zero alphas)

For more examples, see the `notebooks/` directory and `presentation_demos.py`.
