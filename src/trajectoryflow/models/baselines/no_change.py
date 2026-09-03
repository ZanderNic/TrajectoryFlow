# std-lib imports

# 3 party imports
import torch

# package imports
from trajectoryflow.models.base import BaseTrajectoryModel, TrajectoryPrediction



class NoChangeBaseline(BaseTrajectoryModel):

    def __init__(self):
        super().__init__(
            name="no_change"
        )

        self._is_fitted = True


    def predict(
        self,
        source: torch.Tensor,
        source_time: float,
        target_time: float,
        n_samples: int = 1,
    ) -> TrajectoryPrediction:

        self._validate_prediction_input(
            source=source,
            source_time=source_time,
            target_time=target_time,
            n_samples=n_samples,
        )

        states = (
            source
            .unsqueeze(0)
            .expand(
                n_samples,
                -1,
                -1,
            )
            .clone()
        )

        return TrajectoryPrediction(
            states=states,
            source_time=source_time,
            target_time=target_time,
            metadata={
                "model": self.name,
                "deterministic": True,
            },
        )