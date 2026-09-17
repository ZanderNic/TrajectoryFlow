# std-lib imports
from dataclasses import dataclass
import math

# 3 party imports
import torch
from torch import nn
from torch.nn import functional as F

# package imports


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    dims = (input_dim, *hidden_dims)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend((nn.Linear(in_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim)))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class MLPStateEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        self.n_genes, self.latent_dim = n_genes, latent_dim
        self.network = _mlp(n_genes, hidden_dims, latent_dim)

    def forward(self, total: torch.Tensor) -> torch.Tensor:
        return self.network(torch.log1p(total.clamp_min(0)))


class MLPStateDecoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (256, 512),
    ):
        super().__init__()
        self.n_genes, self.latent_dim = n_genes, latent_dim
        self.network = _mlp(latent_dim, hidden_dims, n_genes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.network(latent))


class MultiModalKineticEncoder(nn.Module):
    """Encode total/old/new/NTR without first concatenating four gene matrices."""

    def __init__(
        self,
        n_genes: int,
        kinetic_dim: int = 32,
        projection_dim: int = 96,
        hidden_dims: tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        self.n_genes, self.kinetic_dim = n_genes, kinetic_dim
        self.total_projection = nn.Linear(n_genes, projection_dim)
        self.old_projection = nn.Linear(n_genes, projection_dim)
        self.new_projection = nn.Linear(n_genes, projection_dim)
        self.ntr_projection = nn.Linear(n_genes, projection_dim)
        self.network = _mlp(4 * projection_dim, hidden_dims, kinetic_dim)

    def forward(
        self,
        total: torch.Tensor,
        old: torch.Tensor,
        new: torch.Tensor,
        ntr: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                self.total_projection(torch.log1p(total.clamp_min(0))),
                self.old_projection(torch.log1p(old.clamp_min(0))),
                self.new_projection(torch.log1p(new.clamp_min(0))),
                self.ntr_projection(ntr.clamp(0, 1)),
            ),
            dim=-1,
        )
        return self.network(features)


class PositiveLowRankGeneHead(nn.Module):
    """Gene-wise positive parameter with a gene baseline + low-rank state correction."""

    def __init__(
        self,
        kinetic_dim: int,
        n_genes: int,
        initial_value: float,
        min_value: float = 1e-5,
        max_value: float | None = None,
    ):
        super().__init__()
        if initial_value <= 0:
            raise ValueError("initial_value must be positive.")
        self.min_value, self.max_value = min_value, max_value
        initial_raw = torch.log(torch.expm1(torch.tensor(float(initial_value))))
        self.base_raw = nn.Parameter(torch.full((n_genes,), float(initial_raw)))
        self.loading = nn.Parameter(torch.empty(n_genes, kinetic_dim))
        nn.init.normal_(self.loading, std=1e-3)

    def forward(self, kinetic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.base_raw + kinetic @ self.loading.T
        value = F.softplus(raw) + self.min_value
        if self.max_value is not None:
            value = value.clamp_max(self.max_value)
        base = F.softplus(self.base_raw) + self.min_value
        if self.max_value is not None:
            base = base.clamp_max(self.max_value)
        return value, base


@dataclass
class GaussianTransitionOutput:
    delta_mean: torch.Tensor
    delta_log_std: torch.Tensor

    @property
    def delta_std(self) -> torch.Tensor:
        return self.delta_log_std.exp()


class GaussianTransition(nn.Module):
    """Original stochastic Gaussian latent transition kept as an ablation option."""

    name = "gaussian_latent_delta"

    def __init__(
        self,
        latent_dim: int,
        kinetic_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.network = _mlp(latent_dim + kinetic_dim + 1, hidden_dims, 2 * latent_dim)

    def forward(
        self,
        latent: torch.Tensor,
        kinetic: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> GaussianTransitionOutput:
        if delta_time.ndim == 0:
            delta_time = delta_time.expand(latent.shape[0])
        if delta_time.ndim == 1:
            delta_time = delta_time[:, None]
        output = self.network(torch.cat((latent, kinetic, delta_time), dim=-1))
        delta_mean, delta_log_std = output.chunk(2, dim=-1)
        return GaussianTransitionOutput(
            delta_mean=delta_mean,
            delta_log_std=delta_log_std.clamp(self.min_log_std, self.max_log_std),
        )

    @staticmethod
    def sample(
        output: GaussianTransitionOutput,
        n_samples: int = 1,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            return output.delta_mean.unsqueeze(0).expand(n_samples, -1, -1)
        noise = torch.randn(
            (n_samples, *output.delta_mean.shape),
            device=output.delta_mean.device,
            dtype=output.delta_mean.dtype,
        )
        return output.delta_mean.unsqueeze(0) + output.delta_std.unsqueeze(0) * noise

    @staticmethod
    def regularization(
        output: GaussianTransitionOutput,
        sampled_delta: torch.Tensor,
    ) -> torch.Tensor:
        del sampled_delta
        return 0.5 * (
            output.delta_mean.square()
            + output.delta_log_std.mul(2).exp()
            - 1
            - 2 * output.delta_log_std
        ).mean()


class SinusoidalTimeEmbedding(nn.Module):
    """Small Fourier-style embedding for continuous diffusion time t in [0, 1]."""

    def __init__(self, dim: int = 32):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("time embedding dimension must be even and >= 4.")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1)
        half = self.dim // 2
        frequencies = torch.exp(
            torch.linspace(0, -math.log(10_000), half, device=time.device, dtype=time.dtype)
        )
        angles = 2 * math.pi * time[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class DiffusionResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear_1(F.silu(self.norm(x)))
        x = self.linear_2(F.silu(x))
        return residual + x


class ConditionalVelocityDenoiser(nn.Module):
    """Tiny conditional MLP denoiser predicting diffusion v in latent-delta space."""

    def __init__(
        self,
        latent_dim: int,
        kinetic_dim: int,
        hidden_dims: tuple[int, ...] = (192, 192),
        time_embedding_dim: int = 32,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        if len(set(hidden_dims)) != 1:
            raise ValueError("Diffusion residual MLP currently expects equal hidden dimensions.")
        hidden_dim = hidden_dims[0]
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.input_projection = nn.Linear(
            2 * latent_dim + kinetic_dim + 1 + time_embedding_dim,
            hidden_dim,
        )
        self.blocks = nn.ModuleList(DiffusionResidualBlock(hidden_dim) for _ in hidden_dims)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_delta: torch.Tensor,
        source_latent: torch.Tensor,
        kinetic: torch.Tensor,
        delta_time: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        if delta_time.ndim == 1:
            delta_time = delta_time[:, None]
        time_embedding = self.time_embedding(diffusion_time)
        x = torch.cat(
            (noisy_delta, source_latent, kinetic, delta_time, time_embedding),
            dim=-1,
        )
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.output_projection(F.silu(self.output_norm(x)))


@dataclass
class DiffusionTransitionOutput:
    source_latent: torch.Tensor
    kinetic: torch.Tensor
    delta_time: torch.Tensor


class LatentDiffusionTransition(nn.Module):
    """
    Lightweight conditional latent diffusion transition.

    It generates a latent change Δz with a cosine schedule, v-prediction and a
    deterministic DDIM-style reverse update. Stochasticity comes from the initial
    Gaussian noise. In this project it is trained through population-level losses,
    so no artificial cell-to-cell target pairing is required.
    """

    name = "latent_diffusion_delta"

    def __init__(
        self,
        latent_dim: int,
        kinetic_dim: int,
        hidden_dims: tuple[int, ...] = (192, 192),
        time_embedding_dim: int = 32,
        sample_steps: int = 8,
        cosine_offset: float = 0.008,
    ):
        super().__init__()
        if sample_steps < 2:
            raise ValueError("sample_steps must be >= 2.")
        if not 0 <= cosine_offset < 1:
            raise ValueError("cosine_offset must be in [0, 1).")
        self.latent_dim = latent_dim
        self.sample_steps = int(sample_steps)
        self.cosine_offset = float(cosine_offset)
        self.denoiser = ConditionalVelocityDenoiser(
            latent_dim=latent_dim,
            kinetic_dim=kinetic_dim,
            hidden_dims=hidden_dims,
            time_embedding_dim=time_embedding_dim,
        )

    def forward(
        self,
        latent: torch.Tensor,
        kinetic: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> DiffusionTransitionOutput:
        if delta_time.ndim == 0:
            delta_time = delta_time.expand(latent.shape[0])
        if delta_time.ndim == 1:
            delta_time = delta_time[:, None]
        return DiffusionTransitionOutput(
            source_latent=latent,
            kinetic=kinetic,
            delta_time=delta_time,
        )

    def _alpha_sigma(self, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.cosine_offset
        angle = (time + s) / (1 + s) * math.pi / 2
        base = math.cos(s / (1 + s) * math.pi / 2) ** 2
        alpha_bar = (torch.cos(angle).square() / base).clamp(1e-5, 1.0)
        return alpha_bar.sqrt(), (1 - alpha_bar).clamp_min(0).sqrt()

    def sample(
        self,
        output: DiffusionTransitionOutput,
        n_samples: int = 1,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        batch_size = output.source_latent.shape[0]
        shape = (n_samples, batch_size, self.latent_dim)
        delta = torch.zeros(shape, device=output.source_latent.device, dtype=output.source_latent.dtype)
        if not deterministic:
            delta = torch.randn_like(delta)

        source = output.source_latent.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, self.latent_dim)
        kinetic = output.kinetic.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, output.kinetic.shape[-1])
        delta_time = output.delta_time.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, 1)
        times = torch.linspace(
            1.0,
            0.0,
            self.sample_steps + 1,
            device=delta.device,
            dtype=delta.dtype,
        )

        for current_time, next_time in zip(times[:-1], times[1:]):
            flat_delta = delta.reshape(-1, self.latent_dim)
            t = current_time.expand(flat_delta.shape[0])
            velocity = self.denoiser(flat_delta, source, kinetic, delta_time, t)
            alpha, sigma = self._alpha_sigma(current_time)
            next_alpha, next_sigma = self._alpha_sigma(next_time)
            clean_delta = alpha * flat_delta - sigma * velocity
            noise = sigma * flat_delta + alpha * velocity
            flat_delta = next_alpha * clean_delta + next_sigma * noise
            delta = flat_delta.reshape(shape)
        return delta

    @staticmethod
    def regularization(
        output: DiffusionTransitionOutput,
        sampled_delta: torch.Tensor,
    ) -> torch.Tensor:
        del output
        return sampled_delta.new_zeros(())


class KineticAlignmentHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        kinetic_dim: int,
        hidden_dims: tuple[int, ...] = (128,),
    ):
        super().__init__()
        self.network = _mlp(latent_dim + kinetic_dim, hidden_dims, latent_dim)

    def forward(self, latent: torch.Tensor, kinetic: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((latent, kinetic), dim=-1))
