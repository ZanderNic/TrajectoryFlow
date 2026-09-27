# std-lib imports
from dataclasses import dataclass

# 3 party imports
import torch
from torch import nn

# package imports
from trajectoryflow.models.dual_scale.components import (
    MLPStateDecoder,
    NullKineticEncoder,
    PositiveLowRankGeneHead,
    StateWithNTREncoder,
    StochasticResidualTransition,
    StochasticResidualTransitionOutput,
    TotalKineticEncoder,
)
from trajectoryflow.models.dual_scale.config import (
    DualScaleModelConfig,
)


@dataclass
class LocalOutputs:
    latent: torch.Tensor
    kinetic: torch.Tensor
    reconstruction: torch.Tensor
    alpha: torch.Tensor
    gamma: torch.Tensor
    predicted_new: torch.Tensor


@dataclass
class GlobalOutputs:
    latent: torch.Tensor
    kinetic: torch.Tensor
    alpha: torch.Tensor
    gamma: torch.Tensor

    transition: StochasticResidualTransitionOutput
    sampled_delta: torch.Tensor

    future_latent: torch.Tensor
    future_total: torch.Tensor


class DualScaleKineticTransitionModel(nn.Module):
    """Dual-scale model with direct stochastic residual transitions.

    State path:
        (total RNA, NTR)
        -> z_t
        -> reconstructed total RNA

    Kinetic path:
        total RNA
        -> k_t
        -> alpha / gamma
        -> predicted newly synthesized RNA

    Forecast path:
        (z_t, k_t, delta_t, epsilon)
        -> stochastic residual MLP
        -> delta_z
        -> z_t + delta_z
        -> future total RNA
    """

    def __init__(
        self,
        n_genes: int,
        state_encoder: nn.Module,
        state_decoder: nn.Module,
        kinetic_encoder: nn.Module,
        production_head: nn.Module,
        degradation_head: nn.Module,
        transition_model: StochasticResidualTransition,
        time_scale_hours: float = 5.0,
    ):
        super().__init__()

        if time_scale_hours <= 0:
            raise ValueError(
                "time_scale_hours must be positive."
            )

        self.n_genes = n_genes

        self.state_encoder = state_encoder
        self.state_decoder = state_decoder

        self.kinetic_encoder = kinetic_encoder
        self.production_head = production_head
        self.degradation_head = degradation_head

        self.transition_model = transition_model

        self.time_scale_hours = float(
            time_scale_hours
        )

    @property
    def transition_name(self) -> str:
        return self.transition_model.name

    def encode_state(
        self,
        total: torch.Tensor,
        ntr: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_encoder(
            total,
            ntr,
        )

    def decode_state(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_decoder(
            latent
        )

    def encode_kinetics(
        self,
        total: torch.Tensor,
    ) -> torch.Tensor:
        return self.kinetic_encoder(
            total
        )

    def kinetic_parameters(
        self,
        kinetic: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        alpha, _ = self.production_head(
            kinetic
        )
        gamma, _ = self.degradation_head(
            kinetic
        )

        return alpha, gamma

    @staticmethod
    def predict_new_rna(
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        labeling_time: torch.Tensor | float,
    ) -> torch.Tensor:
        tau = torch.as_tensor(
            labeling_time,
            device=alpha.device,
            dtype=alpha.dtype,
        )

        while tau.ndim < alpha.ndim:
            tau = tau.unsqueeze(-1)

        fraction = -torch.expm1(
            -gamma * tau
        )

        return (
            alpha
            * fraction
            / gamma.clamp_min(1e-8)
        )

    def forward_local(
        self,
        total: torch.Tensor,
        ntr: torch.Tensor,
        labeling_time: torch.Tensor | float,
    ) -> LocalOutputs:
        latent = self.encode_state(
            total,
            ntr,
        )

        kinetic = self.encode_kinetics(
            total
        )

        alpha, gamma = self.kinetic_parameters(
            kinetic
        )

        return LocalOutputs(
            latent=latent,
            kinetic=kinetic,
            reconstruction=self.decode_state(
                latent
            ),
            alpha=alpha,
            gamma=gamma,
            predicted_new=self.predict_new_rna(
                alpha,
                gamma,
                labeling_time,
            ),
        )

    def forward_global(
        self,
        total: torch.Tensor,
        ntr: torch.Tensor,
        delta_time: torch.Tensor | float,
        n_samples: int = 1,
        deterministic: bool = False,
    ) -> GlobalOutputs:
        latent = self.encode_state(
            total,
            ntr,
        )

        kinetic = self.encode_kinetics(
            total
        )

        alpha, gamma = self.kinetic_parameters(
            kinetic
        )

        scaled_delta_time = (
            torch.as_tensor(
                delta_time,
                device=total.device,
                dtype=total.dtype,
            )
            / self.time_scale_hours
        )

        transition = self.transition_model(
            latent=latent,
            kinetic=kinetic,
            delta_time=scaled_delta_time,
        )

        sampled_delta = self.transition_model.sample(
            transition,
            n_samples=n_samples,
            deterministic=deterministic,
        )

        future_latent = (
            latent.unsqueeze(0)
            + sampled_delta
        )

        future_total = self.decode_state(
            future_latent.reshape(
                -1,
                future_latent.shape[-1],
            )
        ).reshape(
            n_samples,
            total.shape[0],
            self.n_genes,
        )

        return GlobalOutputs(
            latent=latent,
            kinetic=kinetic,
            alpha=alpha,
            gamma=gamma,
            transition=transition,
            sampled_delta=sampled_delta,
            future_latent=future_latent,
            future_total=future_total,
        )


def build_default_dual_scale_model(
    n_genes: int,
    config: DualScaleModelConfig | None = None,
) -> DualScaleKineticTransitionModel:
    config = (
        config
        or DualScaleModelConfig()
    )

    state_encoder = StateWithNTREncoder(
        n_genes=n_genes,
        latent_dim=config.latent_dim,
        projection_dim=config.state_projection_dim,
        hidden_dims=config.state_hidden_dims,
    )

    state_decoder = MLPStateDecoder(
        n_genes=n_genes,
        latent_dim=config.latent_dim,
        hidden_dims=config.decoder_hidden_dims,
    )

    effective_kinetic_dim = (
        config.kinetic_dim
        if config.use_kinetic_encoder
        else 0
    )

    kinetic_encoder = (
        TotalKineticEncoder(
            n_genes=n_genes,
            kinetic_dim=config.kinetic_dim,
            hidden_dims=config.kinetic_hidden_dims,
        )
        if config.use_kinetic_encoder
        else NullKineticEncoder()
    )

    production_head = PositiveLowRankGeneHead(
        kinetic_dim=effective_kinetic_dim,
        n_genes=n_genes,
        initial_value=config.initial_production,
    )

    degradation_head = PositiveLowRankGeneHead(
        kinetic_dim=effective_kinetic_dim,
        n_genes=n_genes,
        initial_value=config.initial_degradation,
        max_value=config.max_degradation,
    )

    transition_model = StochasticResidualTransition(
        latent_dim=config.latent_dim,
        kinetic_dim=effective_kinetic_dim,
        noise_dim=config.transition_noise_dim,
        hidden_dim=config.transition_hidden_dim,
        n_blocks=config.transition_blocks,
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
