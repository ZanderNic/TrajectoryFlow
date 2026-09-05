# std-lib imports

# 3 party imports
import torch

# package imports
from trajectoryflow.models.baselines.velvet.config import (
    VelvetSDEConfig,
    VelvetVAEConfig,
)
from trajectoryflow.models.baselines.velvet.dynamics import (
    MetabolicLabelingModel,
    estimate_gamma_extreme_regression,
)
from trajectoryflow.models.baselines.velvet.model import VelvetVAE
from trajectoryflow.models.baselines.velvet.neighborhood import (
    neighborhood_constraint_loss,
)
from trajectoryflow.models.baselines.velvet.sde import VelvetSDE


def test_metabolic_model_shape():
    model = MetabolicLabelingModel(
        n_genes=4,
        labelling_time=2.0,
    )

    total = torch.ones(3, 4)
    velocity = torch.zeros(3, 4)

    predicted = model.predict_new(total, velocity)

    assert predicted.shape == total.shape
    assert torch.isfinite(predicted).all()
    assert (predicted > 0).all()


def test_velvet_stage1_loss_is_finite():
    config = VelvetVAEConfig(
        n_hidden=8,
        n_latent=3,
        vector_hidden=8,
        vector_layers=1,
    )

    model = VelvetVAE(n_genes=5, config=config)

    total = torch.poisson(torch.full((12, 5), 3.0))
    new = torch.minimum(
        total,
        torch.poisson(torch.full((12, 5), 1.0)),
    )

    result = model.stage1_loss(total=total, new=new)

    assert result.loss.ndim == 0
    assert torch.isfinite(result.loss)


def test_neighborhood_constraint_is_finite():
    z = torch.randn(6, 3)
    velocity = torch.randn(6, 3)
    neighbor_z = torch.randn(6, 4, 3)

    loss = neighborhood_constraint_loss(
        z=z,
        velocity=velocity,
        neighbor_z=neighbor_z,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_sde_shape():
    vae_config = VelvetVAEConfig(
        n_hidden=8,
        n_latent=3,
        vector_hidden=8,
        vector_layers=1,
    )

    velvet = VelvetVAE(n_genes=5, config=vae_config)
    sde = VelvetSDE(
        velvet=velvet,
        config=VelvetSDEConfig(n_steps=5),
    )

    z0 = torch.randn(7, 3)

    paths = sde.simulate(
        z0=z0,
        n_simulations=4,
        n_steps=5,
        t_max=2.0,
    )

    assert paths.shape == (5, 4, 7, 3)
    assert torch.isfinite(paths).all()



def test_gamma_initialization_stays_finite_when_new_exceeds_total():
    total = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ]
    )

    new = total * 2.0

    gamma = estimate_gamma_extreme_regression(
        total=total,
        new=new,
        labelling_time=2.0,
    )

    assert torch.isfinite(gamma).all()
    assert (gamma > 0).all()
