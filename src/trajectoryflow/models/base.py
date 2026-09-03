# std-lib imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# 3 party imports
import torch

# package imports


@dataclass
class TrajectoryPrediction:
    """
        Standard output returned by every trajectory model.

        Shape convention
        ----------------
        states:
            [n_samples, n_cells, n_features]

        Deterministic models simply return n_samples = 1.
    """

    states: torch.Tensor

    source_time: float
    target_time: float

    metadata: dict[str, Any] = field(default_factory=dict)


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
    """
        Common interface for all trajectory prediction models.

        A trajectory model receives a population of cells at a source
        timepoint and predicts their state at a later target timepoint.
    """

    def __init__(
        self,
        name: str,
    ):
        self.name = name
        self._is_fitted = False


    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


    def fit(
        self,
        *args,
        **kwargs,
    ) -> "BaseTrajectoryModel":
        """
            Fit the model if fitting is required.

            Models without trainable parameters may simply mark
            themselves as fitted and return self.
        """
        self._is_fitted = True
        return self


    @abstractmethod
    def predict(
        self,
        source: torch.Tensor,
        source_time: float,
        target_time: float,
        n_samples: int = 1,
    ) -> TrajectoryPrediction:
        """
            Predict the population state at target_time.

            Parameters
            ----------
                source:
                    Source population with shape:

                        [n_cells, n_features]

                source_time:
                    Biological time corresponding to source.

                target_time:
                    Biological time to predict.

                n_samples:
                    Number of generated trajectories/populations.

            Returns
            -------
                TrajectoryPrediction
                    Standardized prediction object.
        """
        raise NotImplementedError


    def _validate_prediction_input(
        self,
        source: torch.Tensor,
        source_time: float,
        target_time: float,
        n_samples: int,
    ) -> None:

        if source.ndim != 2:
            raise ValueError(
                "source must have shape "
                "[n_cells, n_features]."
            )

        if target_time <= source_time:
            raise ValueError(
                "target_time must be greater "
                "than source_time."
            )

        if n_samples <= 0:
            raise ValueError(
                "n_samples must be >= 1."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"is_fitted={self.is_fitted}"
            f")"
        )