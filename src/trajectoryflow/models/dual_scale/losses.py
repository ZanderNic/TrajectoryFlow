"""Losses for the three-objective dual-scale stochastic transition model."""

import torch
from torch.nn import functional as F


def log1p_expression(
    x: torch.Tensor,
) -> torch.Tensor:
    """Map non-negative RNA expression to log1p expression space."""
    return torch.log1p(
        x.clamp_min(0)
    )


def reconstruction_loss(
    reconstructed_total: torch.Tensor,
    observed_total: torch.Tensor,
) -> torch.Tensor:
    """L_rec = MSE(log1p(X_hat_t), log1p(X_t))."""
    return F.mse_loss(
        log1p_expression(
            reconstructed_total
        ),
        log1p_expression(
            observed_total
        ),
    )


def kinetic_loss(
    predicted_new: torch.Tensor,
    observed_new: torch.Tensor,
) -> torch.Tensor:
    """L_kin = MSE(log1p(N_hat_t), log1p(N_t))."""
    return F.mse_loss(
        log1p_expression(
            predicted_new
        ),
        log1p_expression(
            observed_new
        ),
    )


def _sliced_wasserstein(
    predicted: torch.Tensor,
    target: torch.Tensor,
    n_projections: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable empirical Sliced-Wasserstein distance."""

    if (
        predicted.ndim != 2
        or target.ndim != 2
    ):
        raise ValueError(
            "predicted and target must have shape [n_cells, n_features]."
        )

    if (
        predicted.shape[1]
        != target.shape[1]
    ):
        raise ValueError(
            "predicted and target must contain the same number of features."
        )

    if n_projections < 1:
        raise ValueError(
            "n_projections must be >= 1."
        )

    if (
        predicted.shape[0] < 1
        or target.shape[0] < 1
    ):
        raise ValueError(
            "populations must be non-empty."
        )

    n = min(
        predicted.shape[0],
        target.shape[0],
    )

    if predicted.shape[0] != n:
        indices = torch.randperm(
            predicted.shape[0],
            device=predicted.device,
        )[:n]

        predicted = predicted[
            indices
        ]

    if target.shape[0] != n:
        indices = torch.randperm(
            target.shape[0],
            device=target.device,
        )[:n]

        target = target[
            indices
        ]

    n_features = (
        predicted.shape[1]
    )

    projections = torch.randn(
        n_features,
        n_projections,
        device=predicted.device,
        dtype=predicted.dtype,
    )

    projections = (
        projections
        / torch.linalg.vector_norm(
            projections,
            dim=0,
            keepdim=True,
        ).clamp_min(eps)
    )

    predicted_projected = (
        predicted @ projections
    )

    target_projected = (
        target @ projections
    )

    predicted_sorted = torch.sort(
        predicted_projected,
        dim=0,
    ).values

    target_sorted = torch.sort(
        target_projected,
        dim=0,
    ).values

    return torch.sqrt(
        torch.mean(
            (
                predicted_sorted
                - target_sorted
            ).square()
        ).clamp_min(eps)
    )


def population_sliced_wasserstein_loss(
    predicted_future: torch.Tensor,
    observed_future: torch.Tensor,
    n_projections: int = 16,
) -> torch.Tensor:
    """L_SW = SW(log1p(X_hat_future), log1p(X_future))."""
    return _sliced_wasserstein(
        log1p_expression(
            predicted_future
        ),
        log1p_expression(
            observed_future
        ),
        n_projections=n_projections,
    )
