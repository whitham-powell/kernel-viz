"""
Visualization components for kernel methods.

This module provides visualization tools for kernelized algorithms including
decision boundary plotting, animations, and interactive visualizations.
"""

from .core import compute_decision_boundary
from .visualizer import PerceptronVisualizer

__all__ = [
    "PerceptronVisualizer",
    "compute_decision_boundary",
]
