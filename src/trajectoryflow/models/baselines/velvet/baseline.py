# std-lib imports

# 3 party imports
import torch

# package imports
from trajectoryflow.models.base import BaseTrajectoryModel, TrajectoryPrediction
from trajectoryflow.models.baselines.velvet.config import VelvetSDEConfig, VelvetVAEConfig
from trajectoryflow.models.baselines.velvet.model import VelvetVAE
from trajectoryflow.models.baselines.velvet.sde import VelvetSDE


class VelvetBaseline(BaseTrajectoryModel):
    """TrajectoryFlow wrapper around VelvetVAE + VelvetSDE."""

    def __init__(self, n_genes: int, hours_per_sde_unit: float | None = None, vae_config: VelvetVAEConfig | None = None, sde_config: VelvetSDEConfig | None = None):
        super().__init__(name="velvet")
        if n_genes < 1:
            raise ValueError("n_genes must be >= 1.")
        if hours_per_sde_unit is not None and hours_per_sde_unit <= 0:
            raise ValueError("hours_per_sde_unit must be > 0 or None.")
        self.hours_per_sde_unit = None if hours_per_sde_unit is None else float(hours_per_sde_unit)
        self.velvet = VelvetVAE(n_genes=n_genes, config=vae_config)
        self.sde = VelvetSDE(velvet=self.velvet, config=sde_config)

    @torch.no_grad()
    def predict_velocity(self, source: torch.Tensor, sample_latent: bool = False) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("VelvetBaseline must be fitted before velocity inference.")
        if source.ndim != 2 or source.shape[1] != self.velvet.n_genes:
            raise ValueError(f"source must have shape [n_cells, {self.velvet.n_genes}].")
        if not torch.isfinite(source).all() or (source < 0).any():
            raise ValueError("Velvet expects finite, non-negative total RNA values.")
        return self.velvet.infer_gene_velocity(total=source, sample_latent=sample_latent)

    def predict(self, source: torch.Tensor, source_time: float, target_time: float, n_samples: int = 1) -> TrajectoryPrediction:
        if not self.is_fitted:
            raise RuntimeError("VelvetBaseline must be fitted before prediction.")
        self._validate_prediction_input(source, source_time, target_time, n_samples)
        if source.shape[1] != self.velvet.n_genes:
            raise ValueError(f"source must contain {self.velvet.n_genes} genes.")
        if (source < 0).any():
            raise ValueError("Velvet expects non-negative raw/estimated total RNA values, not log-normalized expression.")
        if target_time < source_time:
            raise NotImplementedError("Velvet SDE rollout does not support backcast prediction.")
        if self.hours_per_sde_unit is None:
            raise ValueError("hours_per_sde_unit is required for SDE rollout prediction, but not for predict_velocity().")

        t_max = (target_time - source_time) / self.hours_per_sde_unit
        states = self.sde.predict_expression(source_total=source, n_samples=n_samples, t_max=t_max)
        return TrajectoryPrediction(states=states, source_time=source_time, target_time=target_time, metadata={"model": self.name, "stochastic": True, "hours_per_sde_unit": self.hours_per_sde_unit, "sde_t_max": t_max})
