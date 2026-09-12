# std-lib imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# 3 party imports
import torch

# package imports


@dataclass
class TrajectoryPrediction:
    """Standard trajectory output; states has shape [samples, cells, features]."""

    states: torch.Tensor
    source_time: float
    target_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.states.ndim != 3:
            raise ValueError("states must have shape [n_samples, n_cells, n_features].")
        if 0 in self.states.shape:
            raise ValueError("states dimensions must all be non-zero.")
        if not torch.isfinite(self.states).all():
            raise ValueError("states contains non-finite values.")

    @property
    def n_samples(self) -> int:
        return self.states.shape[0]

    @property
    def n_cells(self) -> int:
        return self.states.shape[1]

    @property
    def n_features(self) -> int:
        return self.states.shape[2]


class BaseTrajectoryModel(ABC):
    def __init__(self, name: str):
        if not name.strip():
            raise ValueError("Model name must not be empty.")
        self.name = name
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, *args, **kwargs) -> "BaseTrajectoryModel":
        self._is_fitted = True
        return self

    @abstractmethod
    def predict(self, source: torch.Tensor, source_time: float, target_time: float, n_samples: int = 1) -> TrajectoryPrediction:
        raise NotImplementedError

    def _validate_prediction_input(self, source: torch.Tensor, source_time: float, target_time: float, n_samples: int) -> None:
        if source.ndim != 2 or 0 in source.shape:
            raise ValueError("source must have non-empty shape [n_cells, n_features].")
        if not torch.isfinite(source).all():
            raise ValueError("source contains non-finite values.")
        if not torch.isfinite(torch.tensor([source_time, target_time], dtype=torch.float64)).all():
            raise ValueError("source_time and target_time must be finite.")
        if target_time == source_time:
            raise ValueError("source_time and target_time must differ.")
        if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError("n_samples must be an integer >= 1.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, is_fitted={self.is_fitted})"
