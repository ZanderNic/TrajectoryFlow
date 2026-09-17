# std-lib imports
from dataclasses import dataclass

# 3 party imports

# package imports


@dataclass(frozen=True)
class DualScaleModelConfig:
    latent_dim: int = 64
    kinetic_dim: int = 32
    state_hidden_dims: tuple[int, ...] = (512, 256)
    decoder_hidden_dims: tuple[int, ...] = (256, 512)
    kinetic_projection_dim: int = 96
    kinetic_hidden_dims: tuple[int, ...] = (256, 128)
    transition_type: str = "diffusion"
    transition_hidden_dims: tuple[int, ...] = (192, 192)
    diffusion_time_embedding_dim: int = 32
    diffusion_sample_steps: int = 8
    diffusion_cosine_offset: float = 0.008
    alignment_hidden_dims: tuple[int, ...] = (128,)
    initial_production: float = 0.1
    initial_degradation: float = 0.1
    max_degradation: float = 3.0
    time_scale_hours: float = 2.0

    def __post_init__(self) -> None:
        if self.transition_type not in {"diffusion", "gaussian"}:
            raise ValueError("transition_type must be 'diffusion' or 'gaussian'.")
        if self.diffusion_sample_steps < 2:
            raise ValueError("diffusion_sample_steps must be >= 2.")
