# Mathematical Background

This document provides the mathematical foundations and derivations for the kernelized methods implemented in this framework.

## Table of Contents
1. [Kernel Functions](#kernel-functions)
2. [Kernelized Perceptron](#kernelized-perceptron)
3. [Reproducing Kernel Hilbert Spaces](#reproducing-kernel-hilbert-spaces)
4. [Kernel Properties](#kernel-properties)
5. [References](#references)

## Kernel Functions

### Definition

A kernel function is a similarity measure between data points that implicitly maps data to a higher-dimensional feature space without explicitly computing the transformation.

**Definition**: A function $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ is a kernel if there exists a feature map $\phi: \mathcal{X} \rightarrow \mathcal{H}$ such that:

$$k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}$$

where $\mathcal{H}$ is a Hilbert space and $\langle \cdot, \cdot \rangle_{\mathcal{H}}$ is the inner product in $\mathcal{H}$.

### Implemented Kernels

#### 1. Linear Kernel
$$k(x, y) = x^T y$$

- Feature map: $\phi(x) = x$ (identity)
- Suitable for linearly separable data

#### 2. Polynomial Kernel
$$k(x, y) = (x^T y + c)^d$$

- Parameters: degree $d \in \mathbb{N}$, constant $c \geq 0$
- Feature map dimension: $\binom{n + d}{d}$ for $n$-dimensional input
- Special cases:
  - $d = 1$: Affine kernel
  - $d = 2$: Quadratic kernel

#### 3. Gaussian RBF Kernel
$$k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

- Parameter: bandwidth $\sigma > 0$
- Feature map: infinite-dimensional
- Properties:
  - $k(x, x) = 1$ for all $x$
  - $0 < k(x, y) \leq 1$ for all $x, y$
  - Smaller $\sigma$ → more local influence

#### 4. Laplacian Kernel
$$k(x, y) = \exp\left(-\gamma \|x - y\|_1\right)$$

- Parameter: $\gamma > 0$
- Uses $L_1$ norm instead of $L_2$
- More robust to outliers than RBF

#### 5. Exponential Kernel
$$k(x, y) = \exp\left(-\frac{\|x - y\|}{\sigma}\right)$$

- Parameter: $\sigma > 0$
- Special case of Laplacian with $\gamma = 1/\sigma$ and $L_2$ norm

## Kernelized Perceptron

### Classical Perceptron

The classical perceptron algorithm finds a linear separator:

1. Initialize weights $w = 0$
2. For each training example $(x_i, y_i)$:
   - If $y_i \cdot w^T x_i \leq 0$ (misclassified):
     - Update: $w \leftarrow w + y_i x_i$

### Kernelized Version

The kernelized perceptron works in the dual space using coefficients $\alpha_i$:

**Algorithm**:
1. Initialize $\alpha_i = 0$ for all $i$
2. For each training example $(x_i, y_i)$:
   - Compute decision function: $f(x_i) = \sum_{j=1}^n \alpha_j k(x_j, x_i)$
   - If $y_i \cdot f(x_i) \leq 0$ (misclassified):
     - Update: $\alpha_i \leftarrow \alpha_i + y_i$

**Key insight**: The weight vector can be expressed as:
$$w = \sum_{i=1}^n \alpha_i \phi(x_i)$$

### Decision Function

For a new point $x$:
$$f(x) = \sum_{i=1}^n \alpha_i k(x_i, x)$$

Only support vectors (points with $\alpha_i \neq 0$) contribute to the decision.

### Convergence

**Theorem** (Novikoff, 1962): If the training data is linearly separable in the feature space with margin $\gamma > 0$, the kernelized perceptron converges in at most $(R/\gamma)^2$ updates, where $R$ is the radius of the data in feature space.

## Reproducing Kernel Hilbert Spaces

### RKHS Definition

A Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}$ is a Hilbert space of functions where:

1. **Point evaluation is continuous**: For each $x \in \mathcal{X}$, the evaluation functional $f \mapsto f(x)$ is continuous
2. **Reproducing property**: $f(x) = \langle f, k(x, \cdot) \rangle_{\mathcal{H}}$

### Representer Theorem

**Theorem**: For any regularized empirical risk minimization problem:
$$\min_{f \in \mathcal{H}} \sum_{i=1}^n L(y_i, f(x_i)) + \lambda \|f\|_{\mathcal{H}}^2$$

The optimal solution has the form:
$$f^*(x) = \sum_{i=1}^n \alpha_i k(x_i, x)$$

This justifies the dual representation used in kernelized algorithms.

## Kernel Properties

### Positive Semi-Definiteness

**Definition**: A kernel $k$ is positive semi-definite if for any $n \in \mathbb{N}$, any $x_1, ..., x_n \in \mathcal{X}$, and any $c_1, ..., c_n \in \mathbb{R}$:

$$\sum_{i=1}^n \sum_{j=1}^n c_i c_j k(x_i, x_j) \geq 0$$

**Theorem** (Mercer): A continuous kernel $k$ is positive semi-definite if and only if it can be expressed as an inner product in some Hilbert space.

### Kernel Operations

Valid kernels can be constructed from existing kernels:

1. **Linear combination**: If $k_1, k_2$ are kernels and $a_1, a_2 \geq 0$:
   $$k(x, y) = a_1 k_1(x, y) + a_2 k_2(x, y)$$

2. **Product**: If $k_1, k_2$ are kernels:
   $$k(x, y) = k_1(x, y) \cdot k_2(x, y)$$

3. **Polynomial**: If $k_1$ is a kernel and $p$ is a polynomial with non-negative coefficients:
   $$k(x, y) = p(k_1(x, y))$$

### Feature Space Geometry

For RBF kernel with feature map $\phi$:
- $\|\phi(x)\|^2 = k(x, x) = 1$ (all points on unit sphere)
- $\|\phi(x) - \phi(y)\|^2 = 2(1 - k(x, y))$
- Distance in feature space related to similarity in input space

## Practical Considerations

### Kernel Selection

1. **Linear kernel**: Use when data is likely linearly separable
2. **Polynomial kernel**: 
   - Low degree (2-3) for moderate non-linearity
   - Higher degree risks overfitting
3. **RBF kernel**: 
   - Universal approximator
   - Good default choice
   - Tune $\sigma$ carefully

### Parameter Selection

#### RBF Bandwidth ($\sigma$)
- Small $\sigma$: Complex decision boundary, risk of overfitting
- Large $\sigma$: Smooth decision boundary, risk of underfitting
- Heuristic: Set $\sigma$ to median pairwise distance

#### Polynomial Degree
- Degree 2: Captures quadratic relationships
- Degree 3: Cubic relationships
- Higher degrees: Computational cost increases rapidly

### Computational Complexity

- Kernel evaluation: $O(d)$ for $d$-dimensional data
- Training: $O(n^2 T)$ for $n$ samples, $T$ iterations
- Prediction: $O(ns)$ where $s$ is number of support vectors

## Implementation Notes

### Numerical Stability

1. **RBF kernel**: Use log-sum-exp trick for very small/large $\sigma$
2. **Polynomial kernel**: Beware of numerical overflow for large degrees
3. **Kernel matrix**: Check condition number for ill-conditioning

### Efficiency

1. **Kernel caching**: Store computed kernel values
2. **Sparse representation**: Only store non-zero $\alpha_i$
3. **Approximate methods**: For large datasets, consider:
   - Random Fourier features
   - Nyström approximation

## References

### Foundational Papers

1. **Rosenblatt, F. (1958)**. "The perceptron: A probabilistic model for information storage and organization in the brain." *Psychological Review*, 65(6), 386-408.

2. **Aizerman, M., Braverman, E., & Rozonoer, L. (1964)**. "Theoretical foundations of the potential function method in pattern recognition learning." *Automation and Remote Control*, 25, 821-837.

3. **Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992)**. "A training algorithm for optimal margin classifiers." *Proceedings of the 5th Annual ACM Workshop on Computational Learning Theory*, 144-152.

### Kernel Methods

4. **Schölkopf, B., & Smola, A. J. (2002)**. *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.

5. **Shawe-Taylor, J., & Cristianini, N. (2004)**. *Kernel Methods for Pattern Analysis*. Cambridge University Press.

6. **Hofmann, T., Schölkopf, B., & Smola, A. J. (2008)**. "Kernel methods in machine learning." *Annals of Statistics*, 36(3), 1171-1220.

### RKHS Theory

7. **Aronszajn, N. (1950)**. "Theory of reproducing kernels." *Transactions of the American Mathematical Society*, 68(3), 337-404.

8. **Wahba, G. (1990)**. *Spline Models for Observational Data*. SIAM.

### Practical Guides

9. **Karatzoglou, A., Meyer, D., & Hornik, K. (2006)**. "Support vector machines in R." *Journal of Statistical Software*, 15(9), 1-28.

10. **Gretton, A. (2019)**. "Introduction to RKHS, and some simple kernel algorithms." Lecture notes, Gatsby Unit, UCL.

### Online Resources

- [Kernel Methods Tutorial](http://www.kernel-methods.net/)
- [CS229 Lecture Notes on Kernels](http://cs229.stanford.edu/notes/cs229-notes3.pdf)
- [A Primer on Kernel Methods](https://www.cs.toronto.edu/~duvenaud/cookbook/)