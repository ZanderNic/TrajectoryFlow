# std-lib imports
from dataclasses import dataclass

# 3 party imports
import torch
from torch import nn

# package imports
from trajectoryflow.models.baselines.velvet.config import VelvetVAEConfig
from trajectoryflow.models.baselines.velvet.distributions import (
    standard_normal_kl,
    zinb_nll,
)
from trajectoryflow.models.baselines.velvet.dynamics import (
    LatentVectorField,
    MetabolicLabelingModel,
)
from trajectoryflow.models.baselines.velvet.neighborhood import (
    neighborhood_constraint_loss,
)
from trajectoryflow.models.baselines.velvet.vae import (
    Encoder,
    LinearZINBDecoder,
    observed_log_library,
)


@dataclass
class VelvetLoss:
    loss: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    velocity: torch.Tensor
    neighborhood: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "reconstruction": float(self.reconstruction.detach()),
            "kl": float(self.kl.detach()),
            "velocity": float(self.velocity.detach()),
            "neighborhood": float(self.neighborhood.detach()),
        }


class VelvetVAE(nn.Module):
    """
    Modern PyTorch reimplementation of the metabolic-labeling VelvetVAE model.

    It contains no scvi-tools or Lightning dependency.
    """

    def __init__(self, n_genes: int, config: VelvetVAEConfig | None = None):
        super().__init__()

        self.n_genes = n_genes
        self.config = VelvetVAEConfig() if config is None else config

        self.encoder = Encoder(
            n_genes=n_genes,
            n_hidden=self.config.n_hidden,
            n_latent=self.config.n_latent,
            n_layers=self.config.n_layers,
            dropout_rate=self.config.dropout_rate,
        )

        self.decoder = LinearZINBDecoder(
            n_latent=self.config.n_latent,
            n_genes=n_genes,
            eps=self.config.eps,
        )

        self.vector_field = LatentVectorField(
            n_latent=self.config.n_latent,
            n_hidden=self.config.vector_hidden,
            n_layers=self.config.vector_layers,
        )

        self.biophysics = MetabolicLabelingModel(
            n_genes=n_genes,
            labelling_time=self.config.labelling_time,
            gamma_default=self.config.gamma_default,
            gamma_min=self.config.gamma_min,
            eps=self.config.eps,
        )

    def encode(
        self,
        total: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if total.ndim != 2:
            raise ValueError("total must have shape [n_cells, n_genes].")

        encoder_input = torch.log1p(total.clamp_min(0.0))
        mean, log_variance = self.encoder(encoder_input)

        if sample:
            std = torch.exp(0.5 * log_variance)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        return mean, log_variance, z

    def decode(
        self,
        z: torch.Tensor,
        log_library: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.decoder(z=z, log_library=log_library)

    def latent_velocity(self, z: torch.Tensor) -> torch.Tensor:
        return self.vector_field(z)

    def gene_velocity(
        self,
        z: torch.Tensor,
        log_library: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode Velvet's high-dimensional velocity:
            V_hat = psi(z + V_z) - psi(z)
        """
        latent_velocity = self.vector_field(z)

        expression = self.decoder.decode_mean(
            z=z,
            log_library=log_library,
        )

        expression_next = self.decoder.decode_mean(
            z=z + latent_velocity,
            log_library=log_library,
        )

        velocity = expression_next - expression
        return expression, latent_velocity, velocity

    def predicted_new(
        self,
        z: torch.Tensor,
        log_library: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expression, latent_velocity, velocity = self.gene_velocity(
            z=z,
            log_library=log_library,
        )

        predicted_new = self.biophysics.predict_new(
            total_expression=expression,
            velocity=velocity,
        )

        return predicted_new, expression, latent_velocity, velocity

    def stage1_loss(
        self,
        total: torch.Tensor,
        new: torch.Tensor,
    ) -> VelvetLoss:
        if total.shape != new.shape:
            raise ValueError(f"total/new shape mismatch: {total.shape} vs {new.shape}.")
        if not torch.isfinite(total).all() or not torch.isfinite(new).all():
            raise ValueError("Velvet stage-1 inputs contain non-finite values.")
        if (total < 0).any() or (new < 0).any():
            raise ValueError("Velvet stage-1 inputs must be non-negative.")

        mean, log_variance, z = self.encode(total=total, sample=True)
        log_library = observed_log_library(total, eps=self.config.eps)

        decoded = self.decode(z=z, log_library=log_library)

        reconstruction = zinb_nll(
            x=total,
            mean=decoded["mean"],
            inverse_dispersion=decoded["inverse_dispersion"],
            dropout_logits=decoded["dropout_logits"],
            eps=self.config.eps,
        ).sum(dim=-1).mean()

        kl = standard_normal_kl(
            mean=mean,
            log_variance=log_variance,
        ).sum(dim=-1).mean()

        predicted_new, _, _, _ = self.predicted_new(
            z=z,
            log_library=log_library,
        )

        # Paper Eq. 11 uses ||log(n) - log(n_hat)||^2. log1p is the
        # zero-safe count-data version of the same objective.
        velocity = (
            torch.log1p(new.clamp_min(0.0))
            - torch.log1p(predicted_new)
        ).pow(2).sum(dim=-1).mean()

        zero = torch.zeros((), device=total.device, dtype=total.dtype)
        loss = reconstruction + kl + self.config.velocity_loss_weight * velocity

        return VelvetLoss(
            loss=loss,
            reconstruction=reconstruction,
            kl=kl,
            velocity=velocity,
            neighborhood=zero,
        )

    def stage2_loss(
        self,
        total: torch.Tensor,
        new: torch.Tensor,
        z: torch.Tensor,
        neighbor_z: torch.Tensor,
    ) -> VelvetLoss:
        """
        Stage 2: VAE frozen, vector field trained with labeling + neighborhood loss.
        """
        log_library = observed_log_library(total, eps=self.config.eps)

        predicted_new, _, latent_velocity, _ = self.predicted_new(
            z=z,
            log_library=log_library,
        )

        velocity = (
            torch.log1p(new.clamp_min(0.0))
            - torch.log1p(predicted_new)
        ).pow(2).sum(dim=-1).mean()

        neighborhood = neighborhood_constraint_loss(
            z=z,
            velocity=latent_velocity,
            neighbor_z=neighbor_z,
            sigma=self.config.transition_sigma,
        )

        zero = torch.zeros((), device=total.device, dtype=total.dtype)
        loss = (
            velocity
            + self.config.neighborhood_loss_weight * neighborhood
        )

        return VelvetLoss(
            loss=loss,
            reconstruction=zero,
            kl=zero,
            velocity=velocity,
            neighborhood=neighborhood,
        )

    def freeze_vae(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

    def stage2_parameters(self):
        yield from self.vector_field.parameters()
        yield from self.biophysics.parameters()

    @torch.no_grad()
    def latent_representation(self, total: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()

        mean, _, _ = self.encode(total=total, sample=False)

        if was_training:
            self.train()

        return mean

    @torch.no_grad()
    def decoded_expression(
        self,
        z: torch.Tensor,
        log_library: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder.decode_mean(z=z, log_library=log_library)
