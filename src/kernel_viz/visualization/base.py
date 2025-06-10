"""Base classes for visualization components."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from matplotlib.artist import Artist
from matplotlib.axes import Axes


@dataclass
class AnimationComponent:
    """Represents a single visualization component."""

    setup_func: Callable[[Axes], List[Artist]]
    update_func: Callable[[int, Axes, List[Artist]], List[Artist]]
    subplot_params: Dict[str, Any]
    name: Optional[str] = None
