"""Factory pattern implementation for visualization components."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .base import AnimationComponent
from .core import compute_decision_boundary


class ComponentFactory(ABC):
    """Abstract base class for component factories."""
    
    def __init__(self, logs: Dict[str, Any]):
        """Initialize factory with log data."""
        self.logs = logs
        self._extract_data()
    
    @abstractmethod
    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        pass
    
    @abstractmethod
    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        pass
    
    @abstractmethod
    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        pass
    
    def create(self, **kwargs) -> AnimationComponent:
        """Create the animation component."""
        return AnimationComponent(
            setup_func=self.setup,
            update_func=self.update,
            subplot_params=kwargs.get("subplot_params", {"gridspec": (0, 0)}),
            name=kwargs.get("name", self.__class__.__name__)
        )


class DecisionBoundaryFactory(ComponentFactory):
    """Factory for creating decision boundary visualization components."""
    
    def __init__(
        self, 
        logs: Dict[str, Any], 
        plot_type: Optional[str] = None,
        fixed_dims: Optional[Dict[int, float]] = None
    ):
        self.plot_type = plot_type
        self.fixed_dims = fixed_dims
        super().__init__(logs)
    
    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.xs = self.logs["feature_space"]
        self.ys = self.logs["true_labels"]
        self.kernel = self.logs["kernel"]
        self.kernel_params = self.logs["kernel_params"] or {}
        
        # Determine plot type if not specified
        if self.plot_type is None:
            self.plot_type = (
                "line"
                if self.kernel.__name__ in ("linear_kernel", "affine_kernel")
                else "contour"
            )
    
    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        # Setup scatter plot
        scatter = ax.scatter(
            self.xs[:, 0],
            self.xs[:, 1],
            c=self.ys,
            cmap="bwr",
            edgecolor="k",
            zorder=2,
        )
        
        # Set margins
        margin = 0.1
        x_min, x_max = self.xs[:, 0].min(), self.xs[:, 0].max()
        y_min, y_max = self.xs[:, 1].min(), self.xs[:, 1].max()
        ax.set_xlim(
            [x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)]
        )
        ax.set_ylim(
            [y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)]
        )
        
        # Set initial title and labels
        ax.set_title("Decision Boundary - Iteration 1")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        
        # Initialize line if using line plot
        if self.plot_type == "line":
            line = ax.plot([], [], "k-", lw=2)[0]
            return [scatter, line]
        return [scatter]
    
    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        # Clear old contours
        self._clear_old_contours(ax)
        
        # Compute new boundary
        alphas = self.logs["alphas"][frame]["alphas"]
        xx, yy, zz = compute_decision_boundary(
            self.xs,
            alphas,
            self.kernel,
            self.kernel_params,
            self.fixed_dims,
        )
        
        if self.plot_type == "line":
            return self._update_line_plot(ax, artists, xx, yy, zz)
        else:
            return self._update_contour_plot(ax, artists, xx, yy, zz, frame)
    
    def _clear_old_contours(self, ax: Axes) -> None:
        """Remove all contour collections from previous frame."""
        for artist in ax.collections[1:]:  # Keep scatter plot
            artist.remove()
    
    def _update_line_plot(
        self, 
        ax: Axes, 
        artists: List[Artist], 
        xx: ArrayLike, 
        yy: ArrayLike, 
        zz: ArrayLike
    ) -> List[Artist]:
        """Update the line plot showing decision boundary."""
        scatter, line = artists
        temp_contour = ax.contour(xx, yy, zz, levels=[0], colors="black")
        
        if temp_contour.collections[0].get_paths():
            vertices = temp_contour.collections[0].get_paths()[0].vertices
            line.set_data(vertices[:, 0], vertices[:, 1])
        
        for coll in temp_contour.collections:
            coll.remove()
        
        return [scatter, line]
    
    def _update_contour_plot(
        self,
        ax: Axes,
        artists: List[Artist],
        xx: ArrayLike,
        yy: ArrayLike,
        zz: ArrayLike,
        frame: int,
    ) -> List[Artist]:
        """Update the filled contour plot showing decision regions."""
        scatter = artists[0]
        contour = ax.contourf(
            xx,
            yy,
            zz,
            levels=[-1, 0, 1],
            alpha=0.3,
            cmap="coolwarm",
        )
        # Update title for each frame
        ax.set_title(f"Decision Boundary - Iteration {frame + 1}")
        return [scatter] + list(contour.collections)


class AlphaEvolutionFactory(ComponentFactory):
    """Factory for creating alpha evolution visualization components."""
    
    def __init__(self, logs: Dict[str, Any], debug_mode: bool = False):
        self.debug_mode = debug_mode
        super().__init__(logs)
    
    def _extract_data(self) -> None:
        """Extract necessary data from logs."""
        self.alphas_history = self.logs["alphas"]
        self.n_samples = len(self.alphas_history[0]["alphas"])
        self.all_alphas = np.array([entry["alphas"] for entry in self.alphas_history])
        
        # Calculate global min/max for consistent y-axis
        min_alpha = np.min(self.all_alphas)
        max_alpha = np.max(self.all_alphas)
        alpha_range = max_alpha - min_alpha
        min_margin = 0.1
        margin = max(min_margin, alpha_range * 0.1)
        
        self.y_min = min_alpha - margin
        self.y_max = max_alpha + margin
    
    def setup(self, ax: Axes) -> List[Artist]:
        """Setup initial visualization."""
        lines = []
        for i in range(self.n_samples):
            line, = ax.plot(
                [],
                [],
                label=f"$\\alpha_{{{i}}}$",
                alpha=0.3,
                linewidth=0.5,
                color="gray",
            )
            lines.append(line)
        
        # Configure axes
        ax.set_autoscale_on(False)
        ax.set_xlim(0, len(self.alphas_history))
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect(1)
        
        # Set labels and grid
        ax.set_title("Alpha Values Evolution")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Alpha Value")
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Position legend
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=3,
            borderaxespad=0,
            fontsize="small",
        )
        
        return lines
    
    def update(self, frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
        """Update visualization for given frame."""
        current_alphas = self.alphas_history[frame]["alphas"]
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=3,
            borderaxespad=0,
            fontsize="small",
        )
        
        # Update each line
        for idx, line in enumerate(artists):
            x_data = range(frame + 1)
            y_data = [self.alphas_history[j]["alphas"][idx] for j in range(frame + 1)]
            
            # Update line data
            line.set_data(x_data, y_data)
            
            # Update line appearance based on current alpha value
            is_active = abs(current_alphas[idx]) > 1e-10
            line.set_color("red" if is_active else "gray")
            line.set_alpha(0.7 if is_active else 0.1)
            line.set_zorder(2 if is_active else 1)
            
            # Update legend line color
            if legend:
                legend_line = legend.get_lines()[idx]
                legend_line.set_color("red" if is_active else "gray")
                legend_line.set_alpha(0.7 if is_active else 0.1)
        
        # Update title with statistics
        n_active = np.sum(np.abs(current_alphas) > 1e-10)
        active_percentage = (n_active / self.n_samples) * 100
        
        ax.set_title(
            f"Alpha Values Evolution - Iteration {frame + 1}\n"
            f"Active points: {n_active}/{self.n_samples} ({active_percentage:.1f}%)"
        )
        
        return artists


# Factory method to create components
def create_component(
    component_type: str,
    logs: Dict[str, Any],
    **kwargs
) -> AnimationComponent:
    """Factory method to create visualization components.
    
    Args:
        component_type: Type of component to create
        logs: Training logs data
        **kwargs: Additional arguments for specific component types
        
    Returns:
        AnimationComponent instance
    """
    factories = {
        "decision_boundary": DecisionBoundaryFactory,
        "alpha_evolution": AlphaEvolutionFactory,
    }
    
    if component_type not in factories:
        raise ValueError(f"Unknown component type: {component_type}")
    
    factory_class = factories[component_type]
    factory = factory_class(logs, **kwargs)
    return factory.create()