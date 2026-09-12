# std-lib imports
import math
from dataclasses import dataclass

# 3 party imports

# package imports


@dataclass(frozen=True)
class VelvetVAEConfig:
    """VelvetVAE hyperparameters; defaults follow the released benchmark where known."""

    n_hidden: int = 128
    n_latent: int = 50
    n_layers: int = 1
    dropout_rate: float = 0.1
    vector_hidden: int = 128
    vector_layers: int = 3
    labelling_time: float = 2.0
    kl_loss_weight: float = 0.1
    velocity_loss_weight: float = 10.0
    neighborhood_loss_weight: float = 1.0
    n_neighbors: int = 100
    transition_sigma: float = 0.31622776601683794
    stage1_epochs: int = 200
    stage2_epochs: int = 800
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int | None = None
    microbatch_size: int | None = 256
    latent_batch_size: int = 2048
    initialize_gamma: bool = True
    gamma_init_cells: int = 5000
    gamma_extreme_quantile: float = 0.95
    gamma_ratio_eps: float = 1e-6
    gamma_default: float = 0.1
    gamma_min: float = 1e-5
    eps: float = 1e-8
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("n_hidden", "n_latent", "n_layers", "vector_hidden", "vector_layers", "n_neighbors", "latent_batch_size", "gamma_init_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("stage1_epochs", "stage2_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("labelling_time", "transition_sigma", "lr", "gamma_ratio_eps", "gamma_default", "gamma_min", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        for name in ("kl_loss_weight", "velocity_loss_weight", "neighborhood_loss_weight", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number.")
        if isinstance(self.dropout_rate, bool) or not isinstance(self.dropout_rate, (int, float)) or not math.isfinite(self.dropout_rate) or not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must lie in [0, 1).")
        if isinstance(self.gamma_extreme_quantile, bool) or not isinstance(self.gamma_extreme_quantile, (int, float)) or not math.isfinite(self.gamma_extreme_quantile) or not 0 < self.gamma_extreme_quantile < 1:
            raise ValueError("gamma_extreme_quantile must lie in (0, 1).")
        for name in ("batch_size", "microbatch_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None.")
        if not isinstance(self.initialize_gamma, bool):
            raise ValueError("initialize_gamma must be boolean.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")


@dataclass(frozen=True)
class VelvetSDEConfig:
    """VelvetSDE hyperparameters used by the released benchmark."""

    noise_scalar: float = 0.15
    epochs: int = 250
    cells_per_epoch: int = 200
    simulations_per_cell: int = 50
    n_steps: int = 30
    markov_steps: int = 15
    t_max: float = 25.0
    markov_neighbors: int = 10
    transition_sigma: float = 0.31622776601683794
    lr: float = 1e-3
    weight_decay: float = 1e-3
    covariance_jitter: float = 1e-4
    prediction_steps: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("epochs", "cells_per_epoch", "simulations_per_cell", "n_steps", "markov_steps", "markov_neighbors", "prediction_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("transition_sigma", "t_max", "lr", "covariance_jitter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        for name in ("noise_scalar", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
