"""Adapter to integrate factory pattern with existing PerceptronVisualizer.

This module provides functions that can replace the existing monkey-patched
methods while maintaining the same interface.
"""

from typing import Any, Dict, Optional

from .base import AnimationComponent
from .component_factory import create_component


def create_decision_boundary_component_v2(
    self: Any,
    logs: Dict[str, Any],
    plot_type: Optional[str] = None,
    fixed_dims: Optional[Dict[int, float]] = None,
) -> AnimationComponent:
    """Creates decision boundary visualization component using factory pattern.

    This is a drop-in replacement for the existing method that uses
    the factory pattern internally.
    """
    return create_component(
        "decision_boundary",
        logs,
        plot_type=plot_type,
        fixed_dims=fixed_dims,
    )


def create_alpha_evolution_component_v2(
    self: Any,
    logs: Dict[str, Any],
) -> AnimationComponent:
    """Creates alpha evolution visualization component using factory pattern.

    This is a drop-in replacement for the existing method that uses
    the factory pattern internally.
    """
    # Use debug_mode from the visualizer instance if available
    debug_mode = getattr(self, "debug_mode", False)

    return create_component("alpha_evolution", logs, debug_mode=debug_mode)


# Example of how to extend with new factories
def register_component_factory(component_type: str, factory_class: Any) -> None:
    """Register a new component factory.

    This allows extending the system with new component types without
    modifying existing code.
    """
    # This would modify the factory registry in component_factory.py
    # For now, this is just a demonstration
    pass


# Example usage showing migration path
if __name__ == "__main__":
    # Show how to migrate existing code
    print("Migration example:")
    print("Before (current implementation):")
    print("  component = visualizer.create_decision_boundary_component(logs)")
    print()
    print("After (using factory adapter):")
    print("  component = create_decision_boundary_component_v2(visualizer, logs)")
    print()
    print("Or directly using factory:")
    print("  component = create_component('decision_boundary', logs)")

    # Show extensibility
    print("\nExtensibility example:")
    print("# Register new component type")
    print("register_component_factory('kernel_matrix', KernelMatrixFactory)")
    print("# Use it")
    print("component = create_component('kernel_matrix', logs)")
