# std-lib imports
from dataclasses import dataclass


@dataclass(frozen=True)
class DualScaleModelConfig:
    """Configuration for the dual-scale stochastic residual model."""

    latent_dim: int = 64
    kinetic_dim: int = 32
    use_kinetic_encoder: bool = True

    state_hidden_dims: tuple[int, ...] = (
        512,
        256,
    )
    decoder_hidden_dims: tuple[int, ...] = (
        256,
        512,
    )
    state_projection_dim: int = 128

    kinetic_hidden_dims: tuple[int, ...] = (
        256,
        128,
    )

    transition_noise_dim: int = 32
    transition_hidden_dim: int = 256
    transition_blocks: int = 2

    initial_production: float = 0.1
    initial_degradation: float = 0.1
    max_degradation: float = 3.0

    # This scale belongs to the population-transition model and is independent
    # of the metabolic-labeling interval used by the kinetic objective.
    # The SCI-FATE2 early snapshots used here are 5 h apart, so 5 h maps to 1.
    time_scale_hours: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(
            self.use_kinetic_encoder,
            bool,
        ):
            raise ValueError(
                "use_kinetic_encoder must be boolean."
            )

        if self.latent_dim < 1:
            raise ValueError(
                "latent_dim must be >= 1."
            )

        if self.kinetic_dim < 1:
            raise ValueError(
                "kinetic_dim must be >= 1."
            )

        if self.transition_noise_dim < 1:
            raise ValueError(
                "transition_noise_dim must be >= 1."
            )

        if self.transition_hidden_dim < 1:
            raise ValueError(
                "transition_hidden_dim must be >= 1."
            )

        if self.transition_blocks < 1:
            raise ValueError(
                "transition_blocks must be >= 1."
            )

        if self.initial_production <= 0:
            raise ValueError(
                "initial_production must be > 0."
            )

        if self.initial_degradation <= 0:
            raise ValueError(
                "initial_degradation must be > 0."
            )

        if self.max_degradation <= 0:
            raise ValueError(
                "max_degradation must be > 0."
            )

        if self.time_scale_hours <= 0:
            raise ValueError(
                "time_scale_hours must be > 0."
            )
