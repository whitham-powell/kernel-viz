# src/logging/base.py
# Contains the base algorithm logger class and type definitions used for the logger.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, TypeVar

# Type variables for generic typing
T = TypeVar("T")  # For algorithm state
M = TypeVar("M")  # For metrics
P = TypeVar("P")  # For parameters


class LogLevel(Enum):
    INFO = auto()
    DEBUG = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class AlgorithmEvent:
    """Base dataclass for algorithm events."""

    timestamp: datetime = field(default_factory=datetime.now)
    level: LogLevel = LogLevel.INFO
    iteration: Optional[int] = None


@dataclass
class StateEvent(AlgorithmEvent, Generic[T]):
    """Event for algorithm state changes."""

    state: T  # type: ignore
    state_type: str  # type: ignore


@dataclass
class MetricEvent(AlgorithmEvent, Generic[M]):
    """Event for algorithm metric changes."""

    metric_name: str  # type: ignore
    value: M  # type: ignore


@dataclass
class ParameterEvent(AlgorithmEvent, Generic[P]):
    """Event for algorithm parameter changes."""

    parameter_name: str  # type: ignore
    value: P  # type: ignore


class StateManager(Generic[T]):
    """Manages algorithm state snapshots."""

    def __init__(self) -> None:
        self.states: List[T] = []

    def add_state(self, state: T) -> None:
        """Add a state snapshot."""
        self.states.append(state)

    def get_state(self, index: int) -> Optional[T]:
        """Retrieve a state snapshot at a specific index."""
        return self.states[index] if 0 <= index < len(self.states) else None

    def get_latest_state(self) -> Optional[T]:
        """Retrieve the latest state snapshot."""
        return self.states[-1] if self.states else None

    def get_states(self) -> List[T]:
        """Retrieve copy of all state snapshots."""
        return self.states.copy()


class MetricsTracker(Generic[M]):
    """Tracks algorithm metrics over time."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[M]] = {}

    def add_metric(self, name: str, value: M) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_metric_history(self, name: str) -> List[M]:
        """Retrieve the history of a metric."""
        return self.metrics.get(name, []).copy()

    def get_latest_metric(self, name: str) -> Optional[M]:
        """Retrieve the latest value of a metric."""
        values = self.metrics.get(name, []).copy()
        return values[-1] if values else None

    def get_metric_names(self) -> List[str]:
        """Retrieve the names of all tracked metrics."""
        return list(self.metrics.keys())


class AlgorithmLogger(ABC, Generic[T, M]):
    """Base class for algorithm loggers.
    T: Type of algorithm state
    M: Type of algorithm metrics
    """

    def __init__(self) -> None:
        self.events: List[AlgorithmEvent] = []
        self.parameters: Dict[str, Any] = {}
        self.state_manager = StateManager[T]()
        self.metrics_tracker = MetricsTracker[M]()

    def log_state(
        self,
        state: T,
        state_type: str,
        iteration: Optional[int] = None,
    ) -> None:
        """Log algorithm state changes."""
        self.events.append(
            StateEvent(
                state=state,
                state_type=state_type,
                iteration=iteration,
            ),
        )

    def log_metric(self, name: str, value: M, iteration: Optional[int] = None) -> None:
        """Log algorithm metric changes."""
        self.events.append(
            MetricEvent(
                metric_name=name,
                value=value,
                iteration=iteration,
            ),
        )

    def log_parameter(
        self,
        name: str,
        value: Any,
    ) -> None:
        """Log algorithm parameter changes."""
        self.parameters[name] = value
        self.events.append(
            ParameterEvent(
                parameter_name=name,
                value=value,
            ),
        )

    @abstractmethod
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get the data needed for visualization."""
        pass
