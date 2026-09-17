# std-lib imports
from dataclasses import dataclass
from typing import Any

# 3 party imports
import torch
from torch import nn

# package imports
from trajectoryflow.models.dual_scale.components import (
    GaussianTransition,
    LatentDiffusionTransition,
    MLPStateDecoder,
    MLPStateEncoder,
    MultiModalKineticEncoder,
    PositiveLowRankGeneHead,
)
from trajectoryflow.models.dual_scale.config import DualScaleModelConfig


@dataclass
class LocalOutputs:
    latent: torch.Tensor
    kinetic: torch.Tensor
    reconstruction: torch.Tensor
    alpha: torch.Tensor
    gamma: torch.Tensor
    gamma_base: torch.Tensor
    predicted_new: torch.Tensor


@dataclass
class GlobalOutputs:
    latent: torch.Tensor
    kinetic: torch.Tensor
    alpha: torch.Tensor
    gamma: torch.Tensor
    gamma_base: torch.Tensor
    transition: Any
    kinetic_velocity: torch.Tensor
    kinetic_direction_total: torch.Tensor
    kinetic_direction_latent: torch.Tensor
    sampled_delta: torch.Tensor
    transition_mean: torch.Tensor
    transition_std: torch.Tensor
    future_latent: torch.Tensor
    future_total: torch.Tensor
    reconstructed_past: torch.Tensor

    # Compatibility aliases for the previous mechanistic-residual plotting code.
    @property
    def kinetic_future_total(self) -> torch.Tensor:
        return self.kinetic_direction_total

    @property
    def kinetic_future_latent(self) -> torch.Tensor:
        return self.latent + self.kinetic_direction_latent

    @property
    def kinetic_delta(self) -> torch.Tensor:
        return self.kinetic_direction_latent

    @property
    def sampled_residual(self) -> torch.Tensor:
        # There is no residual decomposition anymore: diffusion predicts the full
        # long-horizon latent displacement. Keep this alias only for old tooling.
        return self.sampled_delta

    @property
    def alignment_delta(self) -> torch.Tensor:
        # The old learned alignment head has been replaced by the mechanistic
        # kinetic direction itself.
        return self.kinetic_direction_latent


class DualScaleKineticTransitionModel(nn.Module):
    """
    Local kinetics + free long-horizon stochastic population transition.

    Metabolic labeling determines a short-horizon local direction. The stochastic
    transition model remains responsible for how far the cell moves over the
    requested forecast horizon. Losses live in the trainer.
    """

    def __init__(
        self,
        n_genes: int,
        state_encoder: nn.Module,
        state_decoder: nn.Module,
        kinetic_encoder: nn.Module,
        production_head: nn.Module,
        degradation_head: nn.Module,
        transition_model: nn.Module,
        time_scale_hours: float = 2.0,
    ):
        super().__init__()
        if time_scale_hours <= 0:
            raise ValueError("time_scale_hours must be positive.")
        self.n_genes = n_genes
        self.state_encoder = state_encoder
        self.state_decoder = state_decoder
        self.kinetic_encoder = kinetic_encoder
        self.production_head = production_head
        self.degradation_head = degradation_head
        self.transition_model = transition_model
        self.time_scale_hours = float(time_scale_hours)

    @property
    def transition_name(self) -> str:
        transition = str(getattr(self.transition_model, "name", self.transition_model.__class__.__name__))
        return f"{transition}_with_kinetic_direction_prior"

    def encode_state(self, total: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(total)

    def decode_state(self, latent: torch.Tensor) -> torch.Tensor:
        return self.state_decoder(latent)

    def encode_kinetics(
        self,
        total: torch.Tensor,
        old: torch.Tensor,
        new: torch.Tensor,
        ntr: torch.Tensor,
    ) -> torch.Tensor:
        return self.kinetic_encoder(total, old, new, ntr)

    def kinetic_parameters(self, kinetic: torch.Tensor):
        alpha, _ = self.production_head(kinetic)
        gamma, gamma_base = self.degradation_head(kinetic)
        return alpha, gamma, gamma_base

    @staticmethod
    def predict_new_rna(
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        labeling_time: torch.Tensor | float,
    ) -> torch.Tensor:
        tau = torch.as_tensor(labeling_time, device=alpha.device, dtype=alpha.dtype)
        while tau.ndim < alpha.ndim:
            tau = tau.unsqueeze(-1)
        fraction = -torch.expm1(-gamma * tau)
        return alpha * fraction / gamma.clamp_min(1e-8)

    @staticmethod
    def reconstruct_past_total(
        old: torch.Tensor,
        gamma: torch.Tensor,
        labeling_time: torch.Tensor | float,
    ) -> torch.Tensor:
        tau = torch.as_tensor(labeling_time, device=old.device, dtype=old.dtype)
        while tau.ndim < old.ndim:
            tau = tau.unsqueeze(-1)
        return old * torch.exp((gamma * tau).clamp_max(8.0))

    @staticmethod
    def predict_kinetic_future_total(
        total: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        delta_time: torch.Tensor | float,
    ) -> torch.Tensor:
        """Analytic local state under dX/dt = alpha - gamma X."""
        delta = torch.as_tensor(delta_time, device=total.device, dtype=total.dtype)
        while delta.ndim < total.ndim:
            delta = delta.unsqueeze(-1)
        decay = torch.exp(-(gamma * delta).clamp_max(30.0))
        steady_state = alpha / gamma.clamp_min(1e-8)
        return (total * decay + steady_state * (1 - decay)).clamp_min(0)

    @staticmethod
    def kinetic_velocity(
        total: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Instantaneous gene-space direction dX/dt = alpha - gamma X."""
        return alpha - gamma * total

    def kinetic_latent_direction(
        self,
        total: torch.Tensor,
        latent: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        direction_step_hours: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map a short kinetic step to latent space; magnitude is not a forecast."""
        if direction_step_hours <= 0:
            raise ValueError("direction_step_hours must be positive.")
        velocity = self.kinetic_velocity(total, alpha, gamma)
        direction_total = self.predict_kinetic_future_total(
            total=total,
            alpha=alpha,
            gamma=gamma,
            delta_time=direction_step_hours,
        )
        direction_latent = self.encode_state(direction_total) - latent
        return velocity, direction_total, direction_latent

    def transition_regularization(
        self,
        transition_output: Any,
        sampled_delta: torch.Tensor,
    ) -> torch.Tensor:
        regularization = getattr(self.transition_model, "regularization", None)
        if regularization is None:
            return sampled_delta.new_zeros(())
        return regularization(transition_output, sampled_delta)

    def forward_local(
        self,
        total: torch.Tensor,
        old: torch.Tensor,
        new: torch.Tensor,
        ntr: torch.Tensor,
        labeling_time: torch.Tensor | float,
    ) -> LocalOutputs:
        latent = self.encode_state(total)
        kinetic = self.encode_kinetics(total, old, new, ntr)
        alpha, gamma, gamma_base = self.kinetic_parameters(kinetic)
        return LocalOutputs(
            latent=latent,
            kinetic=kinetic,
            reconstruction=self.decode_state(latent),
            alpha=alpha,
            gamma=gamma,
            gamma_base=gamma_base,
            predicted_new=self.predict_new_rna(alpha, gamma, labeling_time),
        )

    def forward_global(
        self,
        total: torch.Tensor,
        old: torch.Tensor,
        new: torch.Tensor,
        ntr: torch.Tensor,
        delta_time: torch.Tensor | float,
        labeling_time: torch.Tensor | float,
        n_samples: int = 1,
        deterministic: bool = False,
        kinetic_direction_step_hours: float = 0.25,
    ) -> GlobalOutputs:
        latent = self.encode_state(total)
        kinetic = self.encode_kinetics(total, old, new, ntr)
        alpha, gamma, gamma_base = self.kinetic_parameters(kinetic)
        kinetic_velocity, direction_total, direction_latent = self.kinetic_latent_direction(
            total=total,
            latent=latent,
            alpha=alpha,
            gamma=gamma,
            direction_step_hours=kinetic_direction_step_hours,
        )

        # The long-horizon generator starts from the observed source state. Kinetics
        # conditions it and constrains its direction in the trainer, but does not set
        # the displacement magnitude.
        scaled_delta = torch.as_tensor(delta_time, device=total.device, dtype=total.dtype) / self.time_scale_hours
        transition = self.transition_model(latent, kinetic, scaled_delta)
        sampled_delta = self.transition_model.sample(
            transition,
            n_samples=n_samples,
            deterministic=deterministic,
        )
        transition_mean = sampled_delta.mean(dim=0)
        transition_std = sampled_delta.std(dim=0, unbiased=False)
        future_latent = latent.unsqueeze(0) + sampled_delta
        future_total = self.decode_state(future_latent.reshape(-1, future_latent.shape[-1])).reshape(
            n_samples, total.shape[0], self.n_genes
        )
        return GlobalOutputs(
            latent=latent,
            kinetic=kinetic,
            alpha=alpha,
            gamma=gamma,
            gamma_base=gamma_base,
            transition=transition,
            kinetic_velocity=kinetic_velocity,
            kinetic_direction_total=direction_total,
            kinetic_direction_latent=direction_latent,
            sampled_delta=sampled_delta,
            transition_mean=transition_mean,
            transition_std=transition_std,
            future_latent=future_latent,
            future_total=future_total,
            reconstructed_past=self.reconstruct_past_total(old, gamma, labeling_time),
        )


def build_default_dual_scale_model(
    n_genes: int,
    config: DualScaleModelConfig | None = None,
) -> DualScaleKineticTransitionModel:
    config = config or DualScaleModelConfig()
    state_encoder = MLPStateEncoder(n_genes, config.latent_dim, config.state_hidden_dims)
    state_decoder = MLPStateDecoder(n_genes, config.latent_dim, config.decoder_hidden_dims)
    kinetic_encoder = MultiModalKineticEncoder(
        n_genes=n_genes,
        kinetic_dim=config.kinetic_dim,
        projection_dim=config.kinetic_projection_dim,
        hidden_dims=config.kinetic_hidden_dims,
    )
    production_head = PositiveLowRankGeneHead(
        kinetic_dim=config.kinetic_dim,
        n_genes=n_genes,
        initial_value=config.initial_production,
    )
    degradation_head = PositiveLowRankGeneHead(
        kinetic_dim=config.kinetic_dim,
        n_genes=n_genes,
        initial_value=config.initial_degradation,
        max_value=config.max_degradation,
    )
    if config.transition_type == "diffusion":
        transition_model = LatentDiffusionTransition(
            latent_dim=config.latent_dim,
            kinetic_dim=config.kinetic_dim,
            hidden_dims=config.transition_hidden_dims,
            time_embedding_dim=config.diffusion_time_embedding_dim,
            sample_steps=config.diffusion_sample_steps,
            cosine_offset=config.diffusion_cosine_offset,
        )
    else:
        transition_model = GaussianTransition(
            latent_dim=config.latent_dim,
            kinetic_dim=config.kinetic_dim,
            hidden_dims=config.transition_hidden_dims,
        )
    return DualScaleKineticTransitionModel(
        n_genes=n_genes,
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        kinetic_encoder=kinetic_encoder,
        production_head=production_head,
        degradation_head=degradation_head,
        transition_model=transition_model,
        time_scale_hours=config.time_scale_hours,
    )
