# std-lib imports

# 3 party imports
import torch
from torch import nn

# package imports
from trajectoryflow.models.baselines.velvet.config import VelvetSDEConfig
from trajectoryflow.models.baselines.velvet.distributions import diagonal_gaussian_kl
from trajectoryflow.models.baselines.velvet.model import VelvetVAE
from trajectoryflow.models.baselines.velvet.neighborhood import (
    cubic_interpolate_paths,
    sample_markov_paths,
)


class VelvetSDE(nn.Module):
    """
    VelvetSDE with constant scalar diffusion and stochastic midpoint integration.

    The paper uses a Stratonovich SDE solved with the midpoint method. Because
    diffusion is constant, this implementation can use a direct differentiable
    midpoint integrator without torchsde/scvi dependencies.
    """

    def __init__(
        self,
        velvet: VelvetVAE,
        config: VelvetSDEConfig | None = None,
    ):
        super().__init__()

        self.velvet = velvet
        self.config = VelvetSDEConfig() if config is None else config

    def drift(self, z: torch.Tensor) -> torch.Tensor:
        return self.velvet.vector_field(z)

    def simulate(
        self,
        z0: torch.Tensor,
        n_simulations: int,
        n_steps: int,
        t_max: float,
        noise_scalar: float | None = None,
    ) -> torch.Tensor:
        """
        Simulate latent trajectories.

        Returns:
            [n_steps, n_simulations, n_cells, n_latent]
        """
        if z0.ndim != 2:
            raise ValueError("z0 must have shape [n_cells, n_latent].")
        if n_steps < 2:
            raise ValueError("n_steps must be >= 2.")
        if t_max <= 0:
            raise ValueError("t_max must be > 0.")

        noise = self.config.noise_scalar if noise_scalar is None else noise_scalar
        dt = t_max / (n_steps - 1)
        sqrt_dt = dt**0.5

        state = z0[None, :, :].expand(n_simulations, -1, -1).clone()
        trajectory = [state]

        for _ in range(n_steps - 1):
            brownian = torch.randn_like(state) * sqrt_dt

            drift_start = self.drift(state)
            predictor = state + drift_start * dt + noise * brownian

            midpoint = 0.5 * (state + predictor)
            drift_midpoint = self.drift(midpoint)

            state = state + drift_midpoint * dt + noise * brownian
            trajectory.append(state)

        return torch.stack(trajectory, dim=0)

    def trajectory_kl(
        self,
        sde_paths: torch.Tensor,
        markov_paths: torch.Tensor,
    ) -> torch.Tensor:
        """Paper Eq. 17, summed across simulation time."""
        if sde_paths.shape != markov_paths.shape:
            raise ValueError(
                f"SDE/Markov path shape mismatch: "
                f"{sde_paths.shape} vs {markov_paths.shape}."
            )

        losses = []

        # t=0 is identical by construction and contributes nothing useful.
        for step in range(1, sde_paths.shape[0]):
            kl = diagonal_gaussian_kl(
                p_samples=sde_paths[step],
                q_samples=markov_paths[step],
                jitter=self.config.covariance_jitter,
            )

            losses.append(kl.mean())

        return torch.stack(losses).sum()

    def training_loss(
        self,
        all_z: torch.Tensor,
        transition_matrix: torch.Tensor,
        neighbor_indices: torch.Tensor,
        start_indices: torch.Tensor,
    ) -> torch.Tensor:
        markov = sample_markov_paths(
            z=all_z.detach(),
            transition_matrix=transition_matrix.detach(),
            neighbor_indices=neighbor_indices,
            start_indices=start_indices,
            n_simulations=self.config.simulations_per_cell,
            n_steps=self.config.markov_steps,
        )

        markov = cubic_interpolate_paths(
            paths=markov,
            n_steps=self.config.n_steps,
        )

        z0 = all_z[start_indices]

        sde = self.simulate(
            z0=z0,
            n_simulations=self.config.simulations_per_cell,
            n_steps=self.config.n_steps,
            t_max=self.config.t_max,
        )

        return self.trajectory_kl(
            sde_paths=sde,
            markov_paths=markov,
        )

    @torch.no_grad()
    def predict_expression(
        self,
        source_total: torch.Tensor,
        n_samples: int,
        t_max: float,
        n_steps: int | None = None,
        noise_scalar: float | None = None,
    ) -> torch.Tensor:
        """
        Simulate and decode final expression populations.

        Returns:
            [n_samples, n_cells, n_genes]
        """
        n_steps = self.config.prediction_steps if n_steps is None else n_steps

        z0 = self.velvet.latent_representation(source_total)
        log_library = torch.log(
            source_total.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )

        paths = self.simulate(
            z0=z0,
            n_simulations=n_samples,
            n_steps=n_steps,
            t_max=t_max,
            noise_scalar=noise_scalar,
        )

        final_z = paths[-1]
        n_samples_, n_cells, n_latent = final_z.shape

        flat_z = final_z.reshape(n_samples_ * n_cells, n_latent)
        flat_library = (
            log_library[None, :, :]
            .expand(n_samples_, -1, -1)
            .reshape(n_samples_ * n_cells, 1)
        )

        expression = self.velvet.decoder.decode_mean(
            z=flat_z,
            log_library=flat_library,
        )

        return expression.reshape(n_samples_, n_cells, self.velvet.n_genes)
