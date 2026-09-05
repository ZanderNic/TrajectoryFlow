# std-lib imports

# 3 party imports
import torch

# package imports
from trajectoryflow.models.base import BaseTrajectoryModel, TrajectoryPrediction
from trajectoryflow.models.baselines.velvet.config import (
    VelvetSDEConfig,
    VelvetVAEConfig,
)
from trajectoryflow.models.baselines.velvet.model import VelvetVAE
from trajectoryflow.models.baselines.velvet.sde import VelvetSDE


class VelvetBaseline(BaseTrajectoryModel):
    """
    TrajectoryFlow wrapper around VelvetVAE + VelvetSDE.

    `hours_per_sde_unit` is deliberately required. The original Velvet paper
    simulates in model/SDE time; it does not define a universal conversion from
    biological hours to SDE integration time for arbitrary held-out timepoints.
    """

    def __init__(
        self,
        n_genes: int,
        hours_per_sde_unit: float,
        vae_config: VelvetVAEConfig | None = None,
        sde_config: VelvetSDEConfig | None = None,
    ):
        super().__init__(name="velvet")

        if hours_per_sde_unit <= 0:
            raise ValueError("hours_per_sde_unit must be > 0.")

        self.hours_per_sde_unit = float(hours_per_sde_unit)

        self.velvet = VelvetVAE(
            n_genes=n_genes,
            config=vae_config,
        )

        self.sde = VelvetSDE(
            velvet=self.velvet,
            config=sde_config,
        )

    def predict(
        self,
        source: torch.Tensor,
        source_time: float,
        target_time: float,
        n_samples: int = 1,
    ) -> TrajectoryPrediction:
        if not self.is_fitted:
            raise RuntimeError("VelvetBaseline must be fitted before prediction.")
        if source.ndim != 2:
            raise ValueError("source must have shape [n_cells, n_genes].")
        if target_time <= source_time:
            raise ValueError("target_time must be later than source_time.")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        if (source < 0).any():
            raise ValueError(
                "Velvet expects non-negative raw/estimated total RNA values, "
                "not log-normalized expression."
            )

        delta_hours = target_time - source_time
        t_max = delta_hours / self.hours_per_sde_unit

        states = self.sde.predict_expression(
            source_total=source,
            n_samples=n_samples,
            t_max=t_max,
        )

        return TrajectoryPrediction(
            states=states,
            source_time=source_time,
            target_time=target_time,
            metadata={
                "model": self.name,
                "stochastic": True,
                "hours_per_sde_unit": self.hours_per_sde_unit,
                "sde_t_max": t_max,
            },
        )
