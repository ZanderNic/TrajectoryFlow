# std-lib imports

# 3 party imports
import pytest
import torch

# package imports
from trajectoryflow.models.baselines.velvet.config import (
    VelvetVAEConfig,
)
from trajectoryflow.models.baselines.velvet.model import VelvetVAE


def test_infer_gene_velocity_shape_and_finiteness():
    model = VelvetVAE(
        n_genes=5,
        config=VelvetVAEConfig(
            n_hidden=8,
            n_latent=3,
            vector_hidden=8,
            vector_layers=1,
        ),
    )

    total = torch.poisson(
        torch.full((7, 5), 3.0)
    )

    velocity = model.infer_gene_velocity(total)

    assert velocity.shape == total.shape
    assert torch.isfinite(velocity).all()


def test_infer_gene_velocity_is_deterministic_by_default():
    model = VelvetVAE(
        n_genes=5,
        config=VelvetVAEConfig(
            n_hidden=8,
            n_latent=3,
            vector_hidden=8,
            vector_layers=1,
            dropout_rate=0.5,
        ),
    )
    model.train()

    total = torch.poisson(
        torch.full((7, 5), 3.0)
    )

    first = model.infer_gene_velocity(total)
    second = model.infer_gene_velocity(total)

    torch.testing.assert_close(first, second)
    assert model.training


def test_velvet_rollout_rejects_backcast_explicitly():
    from trajectoryflow.models.baselines.velvet import VelvetBaseline, VelvetVAEConfig

    model = VelvetBaseline(n_genes=3, hours_per_sde_unit=1.0, vae_config=VelvetVAEConfig(n_hidden=4, n_latent=2, vector_hidden=4, n_neighbors=1))
    model._is_fitted = True
    with pytest.raises(NotImplementedError, match="backcast"):
        model.predict(torch.ones((2, 3)), source_time=10.0, target_time=5.0)
