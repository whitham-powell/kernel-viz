"""Builder pattern implementation for visualization components."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .base import AnimationComponent
from .core import compute_decision_boundary


class ComponentBuilder:
    """Builder for creating visualization components step by step."""
    
    def __init__(self):
        """Initialize builder with empty state."""
        self.reset()
    
    def reset(self) -> 'ComponentBuilder':
        """Reset builder to initial state."""
        self._logs: Optional[Dict[str, Any]] = None
        self._setup_func: Optional[Callable] = None
        self._update_func: Optional[Callable] = None
        self._subplot_params: Dict[str, Any] = {"gridspec": (0, 0)}
        self._name: Optional[str] = None
        return self
    
    def with_logs(self, logs: Dict[str, Any]) -> 'ComponentBuilder':
        """Set the logs data."""
        self._logs = logs
        return self
    
    def with_setup(self, setup_func: Callable[[Axes], List[Artist]]) -> 'ComponentBuilder':
        """Set the setup function."""
        self._setup_func = setup_func
        return self
    
    def with_update(
        self, 
        update_func: Callable[[int, Axes, List[Artist]], List[Artist]]
    ) -> 'ComponentBuilder':
        """Set the update function."""
        self._update_func = update_func
        return self
    
    def with_subplot_params(self, **params) -> 'ComponentBuilder':
        """Set subplot parameters."""
        self._subplot_params = params
        return self
    
    def with_name(self, name: str) -> 'ComponentBuilder':
        """Set component name."""
        self._name = name
        return self
    
    def build(self) -> AnimationComponent:
        """Build the animation component."""
        if self._setup_func is None or self._update_func is None:
            raise ValueError("Setup and update functions are required")
        
        return AnimationComponent(
            setup_func=self._setup_func,
            update_func=self._update_func,
            subplot_params=self._subplot_params,
            name=self._name
        )


class DecisionBoundaryBuilder(ComponentBuilder):
    """Specialized builder for decision boundary components."""
    
    def __init__(self):
        """Initialize with decision boundary defaults."""
        super().__init__()
        self._plot_type: str = "contour"
        self._fixed_dims: Optional[Dict[int, float]] = None
    
    def reset(self) -> 'DecisionBoundaryBuilder':
        """Reset to initial state."""
        super().reset()
        self._plot_type = "contour"
        self._fixed_dims = None
        return self
    
    def with_plot_type(self, plot_type: str) -> 'DecisionBoundaryBuilder':
        """Set the plot type (line or contour)."""
        if plot_type not in ("line", "contour"):
            raise ValueError("Plot type must be 'line' or 'contour'")
        self._plot_type = plot_type
        return self
    
    def with_fixed_dims(self, fixed_dims: Dict[int, float]) -> 'DecisionBoundaryBuilder':
        """Set fixed dimensions for high-dimensional data."""
        self._fixed_dims = fixed_dims
        return self
    
    def auto_configure(self) -> 'DecisionBoundaryBuilder':
        """Automatically configure setup and update functions based on logs."""
        if self._logs is None:
            raise ValueError("Logs must be set before auto-configuration")
        
        # Extract data
        xs = self._logs["feature_space"]
        ys = self._logs["true_labels"]
        kernel = self._logs["kernel"]
        kernel_params = self._logs["kernel_params"] or {}
        
        # Auto-detect plot type based on kernel
        if self._plot_type == "contour" and kernel.__name__ in ("linear_kernel", "affine_kernel"):
            self._plot_type = "line"
        
        # Create setup function
        def setup(ax: Axes) -> List[Artist]:
            scatter = ax.scatter(
                xs[:, 0], xs[:, 1], c=ys, cmap="bwr", edgecolor="k", zorder=2
            )
            
            margin = 0.1
            x_min, x_max = xs[:, 0].min(), xs[:, 0].max()
            y_min, y_max = xs[:, 1].min(), xs[:, 1].max()
            ax.set_xlim([x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)])
            ax.set_ylim([y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)])
            
            ax.set_title("Decision Boundary - Iteration 1")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            
            if self._plot_type == "line":
                line = ax.plot([], [], "k-", lw=2)[0]
                return [scatter, line]
            return [scatter]
        
        # Create update function
        def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
            # Clear old contours
            for artist in ax.collections[1:]:
                artist.remove()
            
            alphas = self._logs["alphas"][frame]["alphas"]
            xx, yy, zz = compute_decision_boundary(
                xs, alphas, kernel, kernel_params, self._fixed_dims
            )
            
            if self._plot_type == "line":
                scatter, line = artists
                temp_contour = ax.contour(xx, yy, zz, levels=[0], colors="black")
                
                if temp_contour.collections[0].get_paths():
                    vertices = temp_contour.collections[0].get_paths()[0].vertices
                    line.set_data(vertices[:, 0], vertices[:, 1])
                
                for coll in temp_contour.collections:
                    coll.remove()
                
                return [scatter, line]
            else:
                scatter = artists[0]
                contour = ax.contourf(
                    xx, yy, zz, levels=[-1, 0, 1], alpha=0.3, cmap="coolwarm"
                )
                ax.set_title(f"Decision Boundary - Iteration {frame + 1}")
                return [scatter] + list(contour.collections)
        
        self._setup_func = setup
        self._update_func = update
        
        return self


class AlphaEvolutionBuilder(ComponentBuilder):
    """Specialized builder for alpha evolution components."""
    
    def __init__(self):
        """Initialize with alpha evolution defaults."""
        super().__init__()
        self._show_inactive = True
        self._color_active = "red"
        self._color_inactive = "gray"
    
    def reset(self) -> 'AlphaEvolutionBuilder':
        """Reset to initial state."""
        super().reset()
        self._show_inactive = True
        self._color_active = "red"
        self._color_inactive = "gray"
        return self
    
    def with_active_color(self, color: str) -> 'AlphaEvolutionBuilder':
        """Set color for active alpha values."""
        self._color_active = color
        return self
    
    def with_inactive_color(self, color: str) -> 'AlphaEvolutionBuilder':
        """Set color for inactive alpha values."""
        self._color_inactive = color
        return self
    
    def hide_inactive(self) -> 'AlphaEvolutionBuilder':
        """Hide inactive alpha values."""
        self._show_inactive = False
        return self
    
    def auto_configure(self) -> 'AlphaEvolutionBuilder':
        """Automatically configure setup and update functions based on logs."""
        if self._logs is None:
            raise ValueError("Logs must be set before auto-configuration")
        
        # Extract data
        alphas_history = self._logs["alphas"]
        n_samples = len(alphas_history[0]["alphas"])
        all_alphas = np.array([entry["alphas"] for entry in alphas_history])
        
        # Calculate y-axis limits
        min_alpha = np.min(all_alphas)
        max_alpha = np.max(all_alphas)
        alpha_range = max_alpha - min_alpha
        margin = max(0.1, alpha_range * 0.1)
        y_min = min_alpha - margin
        y_max = max_alpha + margin
        
        # Create setup function
        def setup(ax: Axes) -> List[Artist]:
            lines = []
            for i in range(n_samples):
                line, = ax.plot(
                    [], [],
                    label=f"$\\alpha_{{{i}}}$",
                    alpha=0.3 if self._show_inactive else 0.0,
                    linewidth=0.5,
                    color=self._color_inactive,
                )
                lines.append(line)
            
            ax.set_autoscale_on(False)
            ax.set_xlim(0, len(alphas_history))
            ax.set_ylim(y_min, y_max)
            ax.set_aspect(1)
            
            ax.set_title("Alpha Values Evolution")
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Alpha Value")
            ax.grid(True, linestyle="--", alpha=0.7)
            
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=3,
                borderaxespad=0,
                fontsize="small",
            )
            
            return lines
        
        # Create update function
        def update(frame: int, ax: Axes, artists: List[Artist]) -> List[Artist]:
            current_alphas = alphas_history[frame]["alphas"]
            legend = ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=3,
                borderaxespad=0,
                fontsize="small",
            )
            
            for idx, line in enumerate(artists):
                x_data = range(frame + 1)
                y_data = [alphas_history[j]["alphas"][idx] for j in range(frame + 1)]
                
                line.set_data(x_data, y_data)
                
                is_active = abs(current_alphas[idx]) > 1e-10
                if is_active:
                    line.set_color(self._color_active)
                    line.set_alpha(0.7)
                    line.set_zorder(2)
                else:
                    line.set_color(self._color_inactive)
                    line.set_alpha(0.1 if self._show_inactive else 0.0)
                    line.set_zorder(1)
                
                if legend:
                    legend_line = legend.get_lines()[idx]
                    legend_line.set_color(self._color_active if is_active else self._color_inactive)
                    legend_line.set_alpha(0.7 if is_active else 0.1)
            
            n_active = np.sum(np.abs(current_alphas) > 1e-10)
            active_percentage = (n_active / n_samples) * 100
            
            ax.set_title(
                f"Alpha Values Evolution - Iteration {frame + 1}\n"
                f"Active points: {n_active}/{n_samples} ({active_percentage:.1f}%)"
            )
            
            return artists
        
        self._setup_func = setup
        self._update_func = update
        
        return self