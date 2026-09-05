# std-lib imports
import math

# 3 party imports
import torch
from torch import nn
import torch.nn.functional as F

# package imports


class LatentVectorField(nn.Module):
    """Smooth neural vector field z -> dz/dt."""

    def __init__(
        self,
        n_latent: int,
        n_hidden: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")

        layers = []
        n_in = n_latent

        for _ in range(n_layers):
            layers.extend([nn.Linear(n_in, n_hidden), nn.SiLU()])
            n_in = n_hidden

        layers.append(nn.Linear(n_in, n_latent))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class MetabolicLabelingModel(nn.Module):
    """
    Velvet metabolic-labeling model.

    Paper equation:
        N = (1 - exp(-gamma * t)) / gamma * (dX/dt + gamma * X)

    gamma is gene-specific and shared across cells.
    """

    def __init__(
        self,
        n_genes: int,
        labelling_time: float,
        gamma_default: float = 0.1,
        gamma_min: float = 1e-5,
        eps: float = 1e-8,
    ):
        super().__init__()

        if labelling_time <= 0:
            raise ValueError("labelling_time must be > 0.")

        self.labelling_time = float(labelling_time)
        self.gamma_min = float(gamma_min)
        self.eps = float(eps)

        initial = max(gamma_default - gamma_min, eps)
        raw = math.log(math.expm1(initial))
        self.raw_gamma = nn.Parameter(torch.full((n_genes,), raw))

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma) + self.gamma_min

    @torch.no_grad()
    def set_gamma(self, gamma: torch.Tensor) -> None:
        gamma = gamma.to(self.raw_gamma)
        gamma = gamma.clamp_min(self.gamma_min + self.eps)

        if not torch.isfinite(gamma).all():
            raise ValueError("gamma initialization contains non-finite values.")

        value = gamma - self.gamma_min

        # Stable inverse softplus:
        # softplus^{-1}(y) = y + log(-expm1(-y))
        raw = value + torch.log((-torch.expm1(-value)).clamp_min(self.eps))

        if not torch.isfinite(raw).all():
            raise ValueError("raw gamma initialization contains non-finite values.")

        self.raw_gamma.copy_(raw)

    def predict_new(
        self,
        total_expression: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.gamma
        factor = -torch.expm1(-gamma * self.labelling_time) / gamma

        transcription = velocity + gamma * total_expression
        predicted_new = factor * transcription

        return predicted_new.clamp_min(self.eps)


@torch.no_grad()
def estimate_gamma_extreme_regression(
    total: torch.Tensor,
    new: torch.Tensor,
    labelling_time: float,
    quantile: float = 0.95,
    ratio_eps: float = 1e-6,
    default_gamma: float = 0.1,
) -> torch.Tensor:
    """
    Approximate the paper's extreme-value regression initialization.

    The regression is evaluated in float64 for numerical stability. The fitted
    ratio k is constrained to (0, 1), because gamma = -log(1-k) / t diverges
    at k=1.

    This is particularly important for float32, where 1 - 1e-8 rounds back to
    exactly 1.0 and would otherwise create infinite gamma values.
    """
    if total.shape != new.shape:
        raise ValueError(f"total/new shape mismatch: {total.shape} vs {new.shape}.")
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between 0 and 1.")
    if not 0 < ratio_eps < 0.5:
        raise ValueError("ratio_eps must lie between 0 and 0.5.")
    if labelling_time <= 0:
        raise ValueError("labelling_time must be > 0.")

    output_dtype = total.dtype
    device = total.device

    total64 = total.double()
    new64 = new.double()

    n_genes = total.shape[1]
    gamma = torch.full(
        (n_genes,),
        float(default_gamma),
        device=device,
        dtype=torch.float64,
    )

    for gene in range(n_genes):
        x = total64[:, gene]
        y = new64[:, gene]

        positive = x > ratio_eps

        if positive.sum() < 4:
            continue

        x_positive = x[positive]
        y_positive = y[positive]
        threshold = torch.quantile(x_positive, quantile)

        extreme = x_positive >= threshold
        x_extreme = x_positive[extreme]
        y_extreme = y_positive[extreme]

        denominator = x_extreme.square().sum()

        if not torch.isfinite(denominator) or denominator <= ratio_eps:
            continue

        k = (x_extreme * y_extreme).sum() / denominator

        if not torch.isfinite(k):
            continue

        k = k.clamp(ratio_eps, 1.0 - ratio_eps)
        value = -torch.log1p(-k) / labelling_time

        if torch.isfinite(value) and value > 0:
            gamma[gene] = value

    gamma = gamma.to(dtype=output_dtype)

    if not torch.isfinite(gamma).all():
        raise RuntimeError("Gamma initialization produced non-finite values.")

    return gamma

