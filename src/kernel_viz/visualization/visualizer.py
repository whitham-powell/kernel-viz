"""Main visualizer class for perceptron animations."""

import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import Animation, FuncAnimation
from matplotlib.artist import Artist
from matplotlib.gridspec import GridSpec

from .base import AnimationComponent


class PerceptronVisualizer:
    def __init__(self) -> None:
        self.components: List[AnimationComponent] = []
        self.debug_mode = False
        self._animation: Optional[Animation] = None
        self.total_frames: Optional[int] = None

    def set_debug_mode(self, enabled: bool = True) -> None:
        """Enable or disable debug mode."""
        self.debug_mode = enabled

    def add_component(self, component: AnimationComponent) -> None:
        self.components.append(component)
        self._update_grid_layout()

    def remove_component(self, component: AnimationComponent) -> None:
        raise NotImplementedError("remove_component() is not implemented yet")

    def _calculate_grid_dimensions(self) -> Tuple[int, int]:
        """Calculate optimal grid dimensions based on number of components"""
        n = len(self.components)
        if n <= 1:
            return (1, 1)
        elif n == 2:
            return (1, 2)
        elif n == 3:
            return (2, 2)
        elif n == 4:
            return (2, 2)
        else:
            # For more components create a roughly square grid
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            return (rows, cols)

    def _update_grid_layout(self) -> None:
        """Update grid positions for all components based on current configuration."""
        rows, cols = self._calculate_grid_dimensions()
        n_components = len(self.components)

        if self.debug_mode:
            print(
                f"Updating grid layout: {rows} x {cols} for {n_components} components",
            )

        # Special layouts for common cases
        if n_components == 1:
            self.components[0].subplot_params["gridspec"] = (0, slice(None))
        elif n_components == 2:
            self.components[0].subplot_params["gridspec"] = (0, 0)
            self.components[1].subplot_params["gridspec"] = (0, 1)

        else:
            # General case: fill grid left to right, top to bottom
            for idx, component in enumerate(self.components):
                row = idx // cols
                col = idx % cols

                # Special handling for components that should span multiple columns
                if idx == n_components - 1 and cols > 1 and idx % cols == 0:
                    component.subplot_params["gridspec"] = (row, slice(col, cols))
                else:
                    component.subplot_params["gridspec"] = (row, col)

                if self.debug_mode:
                    print(
                        f"Component {idx} ({component.name}) position: {component.subplot_params['gridspec']}",
                    )

    def _save_animation(self, save_path: str, fps: int) -> None:
        """Save animation with error handling."""
        if self._animation is None:
            raise ValueError("No animation to save")

        try:
            file_extension = save_path.split(".")[-1].lower()
            if file_extension in ["mp4", "mov"]:
                writer = "ffmpeg"
            elif file_extension == "gif":
                writer = "pillow"
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")

            self._animation.save(save_path, writer=writer, fps=fps)

        except Exception as e:
            print(f"Error saving animation: {e}")
            raise

    def animate(  # noqa : C901
        self,
        logs: Dict[str, Any],
        figsize: Tuple[float, float] = (15, 10),
        save_path: Optional[str] = None,
        fps: int = 10,
        debug: bool = False,
    ) -> Animation:
        """Create and display/save the combined animation."""
        self.set_debug_mode(debug)
        # Determine the number of frames from the logged data
        if "misclassification_count" in logs and logs["misclassification_count"]:
            self.total_frames = len(logs["misclassification_count"])
        elif "alphas" in logs and logs["alphas"]:
            self.total_frames = len(logs["alphas"])
        else:
            raise ValueError(
                "Cannot determine number of frames from logs. No misclassification_count or alphas data found.",
            )
        begin_animate_time = time.time()

        if len(self.components) == 0:
            raise ValueError("No components added to visualizer")

        # Validate frame count
        if self.total_frames is None or self.total_frames <= 0:
            raise ValueError(
                f"Animation requires valid number of frames. Got {self.total_frames} frames. "
                "Ensure the logger has recorded data during training.",
            )

        print(f"Starting animation with {self.total_frames} frames")

        plt.close("all")  # Close any existing figures
        self.figure = plt.figure(figsize=figsize)
        rows, cols = self._calculate_grid_dimensions()

        # Only specify width_ratios for multi-column layouts
        grid_params = {}
        if cols == 2:
            grid_params["width_ratios"] = [1, 1.2]

        self.grid_spec = GridSpec(
            rows,
            cols,
            figure=self.figure,
            **grid_params,
        )

        if self.debug_mode:
            print("Debug mode enabled")
            print(f"Created {rows}x{cols} grid for {len(self.components)} components")

        # Initialize components
        component_artists = []
        for idx, component in enumerate(self.components):
            if self.debug_mode:
                print(f"Setting up component {idx}: {component.name}")

            grid_pos = component.subplot_params.get("gridspec")
            ax = self.figure.add_subplot(self.grid_spec[grid_pos])

            try:
                artists = component.setup_func(ax)
                component_artists.append((component, ax, artists))
            except Exception as e:
                print(f"Error setting up component {idx} : {component.name}: {e}")
                raise

        def update(frame: int) -> List[Artist]:
            update_frame_time = time.time()

            if self.debug_mode:
                assert (
                    self.total_frames is not None
                ), "self.total_frames is None in animate() -> update()"
                print(f"\nProcessing frame {frame} / {self.total_frames - 1}")

            all_artists = []

            for component, ax, artists in component_artists:
                try:
                    updated_artists = component.update_func(frame, ax, artists)
                    if not isinstance(updated_artists, list):
                        print(
                            f"Warning: Component {component.name} returned non-list: type={type(updated_artists)}",
                        )
                        updated_artists = list(updated_artists)

                    all_artists.extend(updated_artists)
                except Exception as e:
                    print(
                        f"Error updating component {component.name} at frame {frame}: {e}",
                    )
                    raise

            print(
                f"Frame {frame} completed in {time.time() - update_frame_time:.3f}s",
            )

            return all_artists

        # Main figure layout configuration
        self.figure.tight_layout(pad=1.75)
        # Calculate animation interval from fps (milliseconds per frame)
        interval = 1000 / fps

        self._animation = FuncAnimation(
            self.figure,
            update,
            frames=self.total_frames,
            interval=interval,
            repeat=False,
            blit=True,
        )

        if save_path:
            if self.debug_mode:
                print(f"Saving animation to {save_path}")
            self._save_animation(save_path, fps)

        plt.tight_layout()
        print(
            f"Animation configured with {self.total_frames} and ready for display or saving in {time.time() - begin_animate_time:.3f}s",
        )

        return self._animation
