# kernel-viz
This project implements a framework for visualizing kernels via a kernelized perceptron for STAT 671: Statistical Learning I at Portland State University. Future extensions may include visualizing other RKHS-based methods like kernel PCA and SVMs to explore kernelized learning techniques.


# Dependencies
 - ffmpeg if you want to save animations as videos
    ```bash
    sudo apt-get install ffmpeg
    ```

# To-Do
- [ ] **Core Implementation**
  - [ ] Juptyer notebook for kernelized perceptron usage examples

  - **Visualizations**
    - [x] Multiple component animations
    - [x] Single component animations
    - [ ] Kernel response component - WIP
    - [ ] Kernel matrix heatmap
    - [x] Alpha evolution component
    - [x] 2D decision boundary plotting

- **Documentation**
  - [ ] Add usage examples
  - [ ] Provide step-by-step instructions for running the code
  - [ ] Include mathematical derivations or references for kernelized methods

- **Testing**
  - [ ] Comprehensive unit tests for all kernels
  - [ ] Add tests for visualizations

# Future Work
- **Core Extensions**
  - [ ] Implement kernelized PCA
  - [ ] Implement kernelized SVM
  - [ ] Implement kernelized K-means
  - [ ] Explore other kernelized methods (e.g., kernelized regression)

- **Advanced Visualizations**
  - [ ] Evolution of decision boundaries during training
  - [ ] Visualization of support vectors
  - [ ] Interactive visualizations for parameter adjustments
  - [ ] 3D visualization of decision surfaces
  - [ ] Loss/accuracy tracking during training
  - [ ] Performance optimizations for large datasets
