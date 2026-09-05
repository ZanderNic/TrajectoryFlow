# std-lib imports
from dataclasses import dataclass


@dataclass(frozen=True)
class VelvetVAEConfig:
    """
    Configuration for the VelvetVAE reimplementation.

    Paper-derived defaults:
        n_latent=50
        labelling_time=2 h
        velocity_loss_weight=10
        n_neighbors=100
        stage1_epochs=200
        stage2_epochs=800
        lr=1e-3
        weight_decay=1e-3

    The encoder defaults n_hidden=128, n_layers=1 and dropout_rate=0.1 follow
    standard scVI defaults. The Velvet paper states that the VAE builds on
    scvi-tools but does not enumerate every hidden-layer default.
    """

    n_hidden: int = 128
    n_latent: int = 50
    n_layers: int = 1
    dropout_rate: float = 0.1

    vector_hidden: int = 128
    vector_layers: int = 2

    labelling_time: float = 2.0
    velocity_loss_weight: float = 10.0
    neighborhood_loss_weight: float = 1.0

    n_neighbors: int = 100
    transition_sigma: float = 1.0

    stage1_epochs: int = 200
    stage2_epochs: int = 800
    lr: float = 1e-3
    weight_decay: float = 1e-3

    batch_size: int | None = None
    latent_batch_size: int = 2048

    initialize_gamma: bool = True
    gamma_init_cells: int = 5000
    gamma_extreme_quantile: float = 0.95
    gamma_ratio_eps: float = 1e-6
    gamma_default: float = 0.1
    gamma_min: float = 1e-5

    eps: float = 1e-8
    seed: int = 0


@dataclass(frozen=True)
class VelvetSDEConfig:
    """
    Configuration for VelvetSDE.

    The paper reports 250 training epochs, 200 starting cells per epoch,
    50 simulations per cell, and scalar diffusion magnitude 0.2.

    n_steps, markov_steps and t_max are exposed because these are integration
    choices rather than biological timepoint labels.
    """

    noise_scalar: float = 0.2
    epochs: int = 250
    cells_per_epoch: int = 200
    simulations_per_cell: int = 50

    n_steps: int = 30
    markov_steps: int = 15
    t_max: float = 25.0

    markov_neighbors: int = 10
    transition_sigma: float = 1.0

    lr: float = 1e-3
    weight_decay: float = 1e-3
    covariance_jitter: float = 1e-4

    prediction_steps: int = 100
    seed: int = 0
