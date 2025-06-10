# TODOS.md

This file tracks all TODO/FIXME items found in the codebase that need to be addressed before implementing the broader features listed in README.md.

## Critical Fixes (High Priority)

These issues may affect correctness or usability and should be addressed first:

### 1. Visualization Component Issues
- [x] **FIXME** (src/kernel_viz/visualization/core.py:172): Frame count determination is problematic - should be based on iteration count
- [x] **FIXME** (src/kernel_viz/visualization/core.py:557): Kernel response animation may be incorrectly implemented for kernelized perceptron
- [x] **FIXME** (src/kernel_viz/visualization/core.py:791-792): Kernel matrix animation may not be useful since kernel matrix doesn't change during training

### 2. Algorithm Correctness
- [x] **TODO** (src/kernel_viz/algorithms/perceptron.py:86): Replace print statement with exception/warning for kernel positive definiteness check

### 3. Missing Error Handling
- [x] **TODO** (src/kernel_viz/visualization/core.py:179): Add error handling for missing logs in visualization

## Code Organization (Medium Priority)

### 1. Refactoring Tasks
- [x] **TODO** (src/kernel_viz/visualization/core.py:61): Refactor AnimationComponent class to separate file (moved to base.py)
- [x] **TODO** (src/kernel_viz/visualization/core.py:72): Refactor PerceptronVisualizer class to separate file (moved to visualizer.py)
- [ ] **TODO** (src/kernel_viz/visualization/core.py:257): Determine if certain attributes should be class-level

### 2. Test Improvements
- [x] **TODO** (tests/test_kernelized_perceptron.py:96): Split large test into multiple smaller tests
- [x] **TODO** (tests/test_kernels.py:15-16): Parameterize tests for different data types (ints, floats, numpy arrays)

## External Dependencies (Low Priority)

- [ ] **TODO** (src/kernel_viz/utils/transforms.py:46): Implement polynomial features without sklearn dependency

## Visualization Features from README (To Complete)

These are mentioned in README.md but need implementation:

### Core Visualizations
- [x] Complete kernel response component visualization (now shows full response surface with colorbar)
- [x] Implement kernel matrix heatmap visualization (using factory pattern as proof of concept)
- [ ] Add tests for all visualization components

### Documentation
- [ ] Add comprehensive unit tests for all kernels (beyond existing tests)
- [ ] Add usage examples to documentation
- [ ] Provide step-by-step instructions for running the code
- [ ] Include mathematical derivations or references for kernelized methods

## Notes

- The "Future Work" items in README.md (kernelized PCA, SVM, K-means, etc.) are intentionally not included here as they represent new features rather than fixes to existing code
- Focus should be on fixing the critical issues first, especially those affecting correctness
- After addressing these TODOs, we can move on to the broader feature implementations listed in README.md
