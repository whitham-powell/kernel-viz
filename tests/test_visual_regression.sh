#!/bin/bash
# Visual regression test script for factory implementations
# This script runs visual comparisons between original and factory implementations

set -e  # Exit on error

echo "=== Visual Regression Testing for Factory Implementations ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create temporary directory for test outputs
TEST_DIR="visual_regression_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "Working directory: $(pwd)"
echo ""

# Function to run a comparison test
run_comparison() {
    local test_name=$1
    local kernel=$2
    local max_iter=$3

    echo -e "${YELLOW}Running test: $test_name${NC}"

    # Create Python script for this test
    cat > "${test_name}.py" << EOF
import numpy as np
import matplotlib.pyplot as plt
from kernel_viz.algorithms.perceptron import kernelized_perceptron, PerceptronLogger
from kernel_viz.kernels.base import $kernel
from kernel_viz.visualization.visualizer import PerceptronVisualizer
from kernel_viz.visualization.core import (
    create_decision_boundary_component,
    create_alpha_evolution_component
)
from kernel_viz.visualization.component_factory import (
    DecisionBoundaryFactory,
    AlphaEvolutionFactory
)

# Generate data
np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([1, -1, -1, 1], dtype=np.float64)

# Train model
logger = PerceptronLogger()
kernelized_perceptron(X, y, $kernel,
                     kernel_params=$4,
                     max_iter=$max_iter, logger=logger)
logs = logger.get_logs()

# Create visualizer
viz = PerceptronVisualizer()
viz.total_frames = $max_iter

# Test decision boundary
print("  Testing decision boundary component...")
comp_orig = create_decision_boundary_component(viz, logs)
factory = DecisionBoundaryFactory(logs)
comp_factory = factory.create()

# Compare a specific frame
frame = min(4, $max_iter - 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

artists1 = comp_orig.setup_func(ax1)
comp_orig.update_func(frame, ax1, artists1)
ax1.set_title("Original Implementation")

artists2 = comp_factory.setup_func(ax2)
comp_factory.update_func(frame, ax2, artists2)
ax2.set_title("Factory Implementation")

plt.suptitle("$test_name - Decision Boundary Comparison (Frame {})".format(frame))
plt.tight_layout()
plt.savefig("${test_name}_decision_boundary.png", dpi=150)
plt.close()

# Test alpha evolution
print("  Testing alpha evolution component...")
comp_orig_alpha = create_alpha_evolution_component(viz, logs)
factory_alpha = AlphaEvolutionFactory(logs)
factory_alpha.total_frames = viz.total_frames
comp_factory_alpha = factory_alpha.create()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

artists1 = comp_orig_alpha.setup_func(ax1)
comp_orig_alpha.update_func(frame, ax1, artists1)
ax1.set_title("Original Implementation")

artists2 = comp_factory_alpha.setup_func(ax2)
comp_factory_alpha.update_func(frame, ax2, artists2)
ax2.set_title("Factory Implementation")

plt.suptitle("$test_name - Alpha Evolution Comparison (Frame {})".format(frame))
plt.tight_layout()
plt.savefig("${test_name}_alpha_evolution.png", dpi=150)
plt.close()

print("  ✓ Test completed")
EOF

    # Run the test
    uv run python "${test_name}.py"

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Success${NC}"
    else
        echo -e "  ${RED}✗ Failed${NC}"
        exit 1
    fi
    echo ""
}

# Run tests with different configurations
run_comparison "test_rbf_kernel" "rbf_gaussian_kernel" "10" "{'sigma': 0.5}"
run_comparison "test_linear_kernel" "linear_kernel" "5" "{}"
run_comparison "test_polynomial_kernel" "polynomial_kernel" "8" "{'degree': 3}"

# Check if ImageMagick is available for pixel comparison
if command -v compare &> /dev/null; then
    echo -e "${YELLOW}Running pixel-level comparisons with ImageMagick...${NC}"

    # Function to compare images
    compare_images() {
        local img1=$1
        local img2=$2
        local diff_img="${img1%.png}_diff.png"

        # Extract left and right halves
        convert "$img1" -crop 50%x100%+0+0 "${img1%.png}_left.png"
        convert "$img1" -crop 50%x100%+50%+0 "${img1%.png}_right.png"

        # Compare halves
        compare -metric AE "${img1%.png}_left.png" "${img1%.png}_right.png" "$diff_img" 2>&1 || true

        # Get difference count
        local diff_pixels=$(compare -metric AE "${img1%.png}_left.png" "${img1%.png}_right.png" null: 2>&1 || echo "0")

        if [ "$diff_pixels" -lt "1000" ]; then
            echo -e "  ${GREEN}✓ $img1: Pixel difference = $diff_pixels (acceptable)${NC}"
        else
            echo -e "  ${RED}✗ $img1: Pixel difference = $diff_pixels (too high!)${NC}"
        fi

        # Clean up temp files
        rm -f "${img1%.png}_left.png" "${img1%.png}_right.png"
    }

    # Compare all generated images
    for img in *_decision_boundary.png *_alpha_evolution.png; do
        if [ -f "$img" ]; then
            compare_images "$img"
        fi
    done
else
    echo -e "${YELLOW}ImageMagick not found. Skipping pixel-level comparison.${NC}"
fi

echo ""
echo -e "${GREEN}=== Visual Regression Testing Complete ===${NC}"
echo "Generated images are in: $(pwd)"
echo ""

# Create summary HTML
cat > summary.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Visual Regression Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .test { margin-bottom: 30px; border: 1px solid #ccc; padding: 10px; }
        img { max-width: 100%; height: auto; }
        .comparison { display: flex; gap: 10px; margin-top: 10px; }
        .comparison > div { flex: 1; }
    </style>
</head>
<body>
    <h1>Visual Regression Test Results</h1>
    <p>Generated on: $(date)</p>
EOF

for img in *_decision_boundary.png *_alpha_evolution.png; do
    if [ -f "$img" ]; then
        echo "<div class='test'>" >> summary.html
        echo "<h2>$img</h2>" >> summary.html
        echo "<img src='$img' />" >> summary.html
        echo "</div>" >> summary.html
    fi
done

echo "</body></html>" >> summary.html

echo "Summary HTML created: $(pwd)/summary.html"
