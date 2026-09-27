# std-lib imports
from dataclasses import dataclass

# 3 party imports
import torch
from torch import nn
from torch.nn import functional as F


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    dims = (input_dim, *hidden_dims)
    layers: list[nn.Module] = []

    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend(
            (
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
            )
        )

    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class StateWithNTREncoder(nn.Module):
    """Encode total RNA together with the observed new-to-total ratio."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        projection_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512, 256),
    ):
        super().__init__()

        self.n_genes = n_genes
        self.latent_dim = latent_dim

        self.total_projection = nn.Linear(n_genes, projection_dim)
        self.ntr_projection = nn.Linear(n_genes, projection_dim)

        self.network = _mlp(
            input_dim=2 * projection_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
        )

    def forward(
        self,
        total: torch.Tensor,
        ntr: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                self.total_projection(
                    torch.log1p(total.clamp_min(0))
                ),
                self.ntr_projection(
                    ntr.clamp(0, 1)
                ),
            ),
            dim=-1,
        )

        return self.network(features)


class MLPStateDecoder(nn.Module):
    """Decode a latent cell state into non-negative total RNA expression."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (256, 512),
    ):
        super().__init__()

        self.n_genes = n_genes
        self.latent_dim = latent_dim

        self.network = _mlp(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=n_genes,
        )

    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return F.softplus(
            self.network(latent)
        )


class TotalKineticEncoder(nn.Module):
    """Infer a kinetic representation from total RNA only.

    New RNA is deliberately excluded so that the local kinetic objective
    remains a genuine prediction target.
    """

    def __init__(
        self,
        n_genes: int,
        kinetic_dim: int = 32,
        hidden_dims: tuple[int, ...] = (256, 128),
    ):
        super().__init__()

        self.n_genes = n_genes
        self.kinetic_dim = kinetic_dim

        self.network = _mlp(
            input_dim=n_genes,
            hidden_dims=hidden_dims,
            output_dim=kinetic_dim,
        )

    def forward(
        self,
        total: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            torch.log1p(total.clamp_min(0))
        )


class NullKineticEncoder(nn.Module):
    """Ablation encoder that removes cell-specific kinetic conditioning."""

    def __init__(self):
        super().__init__()
        self.kinetic_dim = 0

    def forward(
        self,
        total: torch.Tensor,
    ) -> torch.Tensor:
        return total.new_zeros(
            (total.shape[0], 0)
        )


class PositiveLowRankGeneHead(nn.Module):
    """Positive gene-wise parameter with baseline + low-rank cell correction."""

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
            raise ValueError(
                "initial_value must be positive."
            )

        self.min_value = min_value
        self.max_value = max_value

        initial_raw = torch.log(
            torch.expm1(
                torch.tensor(float(initial_value))
            )
        )

        self.base_raw = nn.Parameter(
            torch.full(
                (n_genes,),
                float(initial_raw),
            )
        )

        self.loading = nn.Parameter(
            torch.empty(
                n_genes,
                kinetic_dim,
            )
        )
        nn.init.normal_(
            self.loading,
            std=1e-3,
        )

    def forward(
        self,
        kinetic: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = (
            self.base_raw
            + kinetic @ self.loading.T
        )

        value = (
            F.softplus(raw)
            + self.min_value
        )

        if self.max_value is not None:
            value = value.clamp_max(
                self.max_value
            )

        base = (
            F.softplus(self.base_raw)
            + self.min_value
        )

        if self.max_value is not None:
            base = base.clamp_max(
                self.max_value
            )

        return value, base


class ResidualMLPBlock(nn.Module):
    """Simple residual MLP block used inside the stochastic transition."""

    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            hidden_dim
        )
        self.linear_1 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )
        self.linear_2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x = self.linear_1(
            F.gelu(
                self.norm(x)
            )
        )
        x = self.linear_2(
            F.gelu(x)
        )

        return residual + x


@dataclass
class StochasticResidualTransitionOutput:
    """Conditioning information required to sample latent changes."""

    source_latent: torch.Tensor
    kinetic: torch.Tensor
    delta_time: torch.Tensor


class StochasticResidualTransition(nn.Module):
    """Direct stochastic generator for latent cell-state changes.

    For every source cell, the transition samples epsilon ~ N(0, I) and predicts

        Delta z = Delta t * G(z_t, k_t, Delta t, epsilon)

    The multiplication by the scaled time interval gives an identity condition:
    Delta t = 0 implies Delta z = 0.

    Sampling requires one generator forward pass per future sample.
    """

    name = "stochastic_residual_mlp"

    def __init__(
        self,
        latent_dim: int,
        kinetic_dim: int,
        noise_dim: int = 32,
        hidden_dim: int = 256,
        n_blocks: int = 2,
    ):
        super().__init__()

        if latent_dim < 1:
            raise ValueError(
                "latent_dim must be >= 1."
            )
        if kinetic_dim < 0:
            raise ValueError(
                "kinetic_dim must be >= 0."
            )
        if noise_dim < 1:
            raise ValueError(
                "noise_dim must be >= 1."
            )
        if hidden_dim < 1:
            raise ValueError(
                "hidden_dim must be >= 1."
            )
        if n_blocks < 1:
            raise ValueError(
                "n_blocks must be >= 1."
            )

        self.latent_dim = latent_dim
        self.kinetic_dim = kinetic_dim
        self.noise_dim = noise_dim

        input_dim = (
            latent_dim
            + kinetic_dim
            + noise_dim
            + 1
        )

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim)
            for _ in range(n_blocks)
        )

        self.output_norm = nn.LayerNorm(
            hidden_dim
        )
        self.output_projection = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        # Begin close to an identity transition instead of producing
        # large arbitrary displacements at initialization.
        nn.init.normal_(
            self.output_projection.weight,
            std=1e-3,
        )
        nn.init.zeros_(
            self.output_projection.bias
        )

    def forward(
        self,
        latent: torch.Tensor,
        kinetic: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> StochasticResidualTransitionOutput:
        if delta_time.ndim == 0:
            delta_time = delta_time.expand(
                latent.shape[0]
            )

        if delta_time.ndim == 1:
            delta_time = delta_time[:, None]

        if delta_time.shape != (
            latent.shape[0],
            1,
        ):
            raise ValueError(
                "delta_time must be scalar, [batch], or [batch, 1]."
            )

        return StochasticResidualTransitionOutput(
            source_latent=latent,
            kinetic=kinetic,
            delta_time=delta_time,
        )

    def _generate_delta(
        self,
        source_latent: torch.Tensor,
        kinetic: torch.Tensor,
        delta_time: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            (
                source_latent,
                kinetic,
                delta_time,
                noise,
            ),
            dim=-1,
        )

        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        raw_delta = self.output_projection(
            F.gelu(
                self.output_norm(x)
            )
        )

        # Residual transition with exact identity at delta_time = 0.
        return delta_time * raw_delta

    def sample(
        self,
        output: StochasticResidualTransitionOutput,
        n_samples: int = 1,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if n_samples < 1:
            raise ValueError(
                "n_samples must be >= 1."
            )

        batch_size = (
            output.source_latent.shape[0]
        )

        source = (
            output.source_latent
            .unsqueeze(0)
            .expand(
                n_samples,
                -1,
                -1,
            )
            .reshape(
                n_samples * batch_size,
                self.latent_dim,
            )
        )

        kinetic = (
            output.kinetic
            .unsqueeze(0)
            .expand(
                n_samples,
                -1,
                -1,
            )
            .reshape(
                n_samples * batch_size,
                self.kinetic_dim,
            )
        )

        delta_time = (
            output.delta_time
            .unsqueeze(0)
            .expand(
                n_samples,
                -1,
                -1,
            )
            .reshape(
                n_samples * batch_size,
                1,
            )
        )

        if deterministic:
            noise = source.new_zeros(
                (
                    n_samples * batch_size,
                    self.noise_dim,
                )
            )
        else:
            noise = torch.randn(
                (
                    n_samples * batch_size,
                    self.noise_dim,
                ),
                device=source.device,
                dtype=source.dtype,
            )

        delta = self._generate_delta(
            source_latent=source,
            kinetic=kinetic,
            delta_time=delta_time,
            noise=noise,
        )

        return delta.reshape(
            n_samples,
            batch_size,
            self.latent_dim,
        )
