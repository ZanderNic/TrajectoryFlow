# std-lib imports

# 3 party imports
import torch
import torch.nn.functional as F

# package imports
from trajectoryflow.models.baselines.velvet.config import VelvetVAEConfig
from trajectoryflow.models.baselines.velvet.dynamics import LatentVectorField
from trajectoryflow.models.baselines.velvet.model import VelvetVAE
from trajectoryflow.models.baselines.velvet.neighborhood import (
    neighborhood_constraint_loss,
    neighborhood_projection,
)


def test_vector_field_matches_released_architecture_semantics():
    field = LatentVectorField(
        n_latent=4,
        n_hidden=8,
        n_layers=3,
    )

    linear = [
        module
        for module in field.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    relu = [
        module
        for module in field.modules()
        if isinstance(module, torch.nn.ReLU)
    ]

    assert len(linear) == 4
    assert len(relu) == 3

    assert linear[0].in_features == 4
    assert linear[0].out_features == 8

    assert linear[1].in_features == 8
    assert linear[1].out_features == 8

    assert linear[2].in_features == 8
    assert linear[2].out_features == 8

    assert linear[3].in_features == 8
    assert linear[3].out_features == 4


def test_neighborhood_projection_matches_released_formula():
    x = torch.tensor(
        [
            [1.0, 0.2, -0.5, 0.7],
            [0.3, 1.1, 0.4, -0.2],
            [-0.7, 0.5, 1.2, 0.1],
            [0.8, -0.4, 0.6, 1.3],
            [-0.2, 0.9, -1.0, 0.4],
            [1.1, -0.8, 0.2, -0.6],
        ]
    )

    velocity = torch.tensor(
        [
            [0.4, -0.2, 0.7, 0.1],
            [-0.3, 0.8, 0.2, -0.4],
            [0.6, 0.1, -0.5, 0.9],
            [-0.7, 0.3, 0.4, 0.2],
            [0.2, 0.5, -0.8, 0.6],
            [0.9, -0.1, 0.3, -0.5],
        ]
    )

    neighbors = torch.tensor(
        [
            [1, 2, 3],
            [0, 2, 4],
            [0, 1, 5],
            [0, 2, 5],
            [1, 2, 3],
            [0, 2, 4],
        ]
    )

    neighbor_x = x[neighbors]
    sigma = 1.0 / (10.0**0.5)

    displacements = neighbor_x - x[:, None, :]
    cosine = F.cosine_similarity(
        velocity[:, None, :],
        displacements,
        dim=-1,
    )

    weights = torch.expm1(cosine * 10.0)
    weights = weights / weights.abs().sum(dim=1, keepdim=True)

    expected = (
        torch.einsum("bkd,bk->bd", displacements, weights)
        - weights.mean(dim=1, keepdim=True)
        * displacements.sum(dim=1)
    )

    actual = neighborhood_projection(
        z=x,
        velocity=velocity,
        neighbor_z=neighbor_x,
        sigma=sigma,
    )

    torch.testing.assert_close(actual, expected)

    expected_loss = (
        1.0
        - F.cosine_similarity(
            velocity,
            expected,
            dim=1,
        ).mean()
    )

    actual_loss = neighborhood_constraint_loss(
        z=x,
        velocity=velocity,
        neighbor_z=neighbor_x,
        sigma=sigma,
    )

    torch.testing.assert_close(actual_loss, expected_loss)


def test_stage1_loss_uses_official_kl_and_velocity_scaling(monkeypatch):
    total = torch.tensor(
        [[10.0, 0.0, 5.0], [0.0, 15.0, 9.0]]
    )
    new = torch.tensor(
        [[3.0, 0.0, 1.0], [0.0, 4.0, 3.0]]
    )
    predicted_new = torch.tensor(
        [[2.5, 0.2, 1.2], [0.3, 3.5, 2.7]]
    )
    mean = torch.tensor(
        [[0.2, -0.3], [0.7, 0.1]]
    )
    log_var = torch.tensor(
        [[-0.2, 0.1], [0.2, -0.1]]
    )
    rate = torch.tensor(
        [[8.0, 2.0, 6.0], [2.0, 14.0, 8.0]]
    )
    theta = torch.tensor(
        [[2.0, 3.0, 1.5], [2.0, 3.0, 1.5]]
    )
    dropout = torch.tensor(
        [[-1.0, -0.5, -1.5], [-0.7, -1.1, -0.9]]
    )

    model = VelvetVAE(
        n_genes=3,
        config=VelvetVAEConfig(
            n_hidden=8,
            n_latent=2,
            vector_hidden=8,
            vector_layers=1,
            dropout_rate=0.0,
            initialize_gamma=False,
        ),
    )

    def encode(total, sample=True):
        return mean, log_var, torch.zeros((2, 2))

    def decode(z, log_library):
        return {
            "mean": rate,
            "scale": torch.softmax(rate, dim=-1),
            "inverse_dispersion": theta,
            "dropout_logits": dropout,
        }

    def predicted(z, log_library):
        return (
            predicted_new,
            rate,
            torch.zeros_like(z),
            torch.zeros_like(rate),
        )

    monkeypatch.setattr(model, "encode", encode)
    monkeypatch.setattr(model, "decode", decode)
    monkeypatch.setattr(model, "predicted_new", predicted)

    result = model.stage1_loss(total, new)

    from trajectoryflow.models.baselines.velvet.distributions import (
        standard_normal_kl,
        zinb_nll,
    )

    reconstruction = zinb_nll(
        x=total,
        mean=rate,
        inverse_dispersion=theta,
        dropout_logits=dropout,
        eps=model.config.eps,
    ).sum(dim=-1).mean()

    kl = (
        standard_normal_kl(mean, log_var)
        .sum(dim=-1)
        .mean()
        * 0.1
    )

    velocity = F.mse_loss(
        torch.log1p(new),
        torch.log1p(predicted_new),
    )

    expected = reconstruction + kl + 10.0 * velocity

    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(result.kl, kl)
    torch.testing.assert_close(result.velocity, velocity)


def test_paper_faithful_defaults():
    config = VelvetVAEConfig()

    assert config.vector_layers == 3
    assert config.kl_loss_weight == 0.1
    assert config.velocity_loss_weight == 10.0
    assert config.transition_sigma == 1.0 / (10.0**0.5)
