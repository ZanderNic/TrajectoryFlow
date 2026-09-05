# std-lib imports

# 3 party imports
import torch
from torch import nn
import torch.nn.functional as F

# package imports


class Encoder(nn.Module):
    """scVI-style Gaussian encoder implemented directly in PyTorch."""

    def __init__(
        self,
        n_genes: int,
        n_hidden: int,
        n_latent: int,
        n_layers: int,
        dropout_rate: float,
    ):
        super().__init__()

        layers = []
        n_in = n_genes

        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(n_in, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            n_in = n_hidden

        self.network = nn.Sequential(*layers)
        self.mean = nn.Linear(n_in, n_latent)
        self.log_variance = nn.Linear(n_in, n_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(x)
        return self.mean(hidden), self.log_variance(hidden).clamp(-12.0, 12.0)


class LinearZINBDecoder(nn.Module):
    """
    Linearly decoded count model.

    The latent-to-expression mapping is linear, while the count likelihood is
    zero-inflated negative binomial. Observed library size is retained per cell.
    """

    def __init__(self, n_latent: int, n_genes: int, eps: float = 1e-8):
        super().__init__()

        self.factor_loadings = nn.Linear(n_latent, n_genes, bias=False)
        self.dropout = nn.Linear(n_latent, n_genes)
        self.raw_inverse_dispersion = nn.Parameter(torch.zeros(n_genes))
        self.eps = eps

    def forward(
        self,
        z: torch.Tensor,
        log_library: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expression_logits = self.factor_loadings(z)
        scale = torch.softmax(expression_logits, dim=-1)

        library = torch.exp(log_library)
        mean = library * scale

        inverse_dispersion = F.softplus(self.raw_inverse_dispersion) + self.eps
        inverse_dispersion = inverse_dispersion.expand_as(mean)

        return {
            "mean": mean,
            "scale": scale,
            "inverse_dispersion": inverse_dispersion,
            "dropout_logits": self.dropout(z),
        }

    def decode_mean(self, z: torch.Tensor, log_library: torch.Tensor) -> torch.Tensor:
        return self.forward(z=z, log_library=log_library)["mean"]


def observed_log_library(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.sum(dim=-1, keepdim=True).clamp_min(eps))
