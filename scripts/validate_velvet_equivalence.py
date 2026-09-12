"""
Validate TrajectoryFlow's Velvet implementation at two levels.

Run the complete validation:
    python scripts/validate_velvet_equivalence.py

Run only TrajectoryFlow API/benchmark regression checks:
    python scripts/validate_velvet_equivalence.py --self-only

Rebuild the isolated pinned official environment:
    python scripts/validate_velvet_equivalence.py --rebuild-reference

Level 1: official numerical equivalence
---------------------------------------
The pinned official VelvetVAE implementation and TrajectoryFlow receive
identical toy inputs, weights and hyperparameters. Core equations/components
are compared numerically within RTOL/ATOL.

Level 2: TrajectoryFlow integration checks
------------------------------------------
Checks the new local-velocity API and benchmark plumbing:
- deterministic finite gene-velocity inference and training-mode restoration
- VelvetBaseline.predict_velocity() and ExperimentModelAdapter integration
- PCA velocity projection (linear and finite-difference)
- cosine PTS/CRS semantics, zero-velocity validity handling
- CBD mean-neighbour cosine semantics
- velocity-reference save/load roundtrip
- train_sde=False really skips the SDE-training branch

This validator deliberately does not compare independently trained models,
paper PTS/CRS/CBD scores, or long-run training convergence. Those require the
paper data/reference trajectories and are separate benchmark-level tests.
"""

# std-lib imports
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

# 3 party imports
import numpy as np

# package imports


OFFICIAL_REPO = "https://github.com/rorymaizels/velvetVAE.git"
OFFICIAL_REF = "e493643de70c2dd10b247831ad8c2c81cf858895"
REFERENCE_SETUPTOOLS = "80.9.0"
TORCHCUBICSPLINE_REPO = "git+https://github.com/patrick-kidger/torchcubicspline.git"

RTOL = 1e-5
ATOL = 1e-6


# ------------------------------------------------------------------
# Shared deterministic setup
# ------------------------------------------------------------------


def toy_data():
    x = np.array(
        [
            [1.0, 0.2, 0.5, 2.0],
            [0.3, 1.1, 1.5, 0.5],
            [2.0, 1.0, 0.4, 1.0],
            [0.5, 2.0, 1.0, 0.2],
            [1.5, 0.5, 2.0, 1.0],
        ],
        dtype=np.float32,
    )

    velocity = np.array(
        [
            [0.5, 0.2, 0.1, 0.3],
            [0.1, 0.5, 0.3, 0.2],
            [0.3, 0.1, 0.5, 0.2],
            [0.2, 0.4, 0.2, 0.5],
            [0.4, 0.2, 0.3, 0.1],
        ],
        dtype=np.float32,
    )

    neighbors = np.array(
        [[1, 2, 3], [0, 2, 4], [0, 1, 4], [0, 1, 4], [0, 1, 3]],
        dtype=np.int64,
    )

    z = np.array(
        [
            [0.10, -0.20, 0.30, -0.40],
            [0.50, -0.60, 0.70, -0.80],
            [1.00, -1.00, 0.25, -0.75],
        ],
        dtype=np.float32,
    )

    return x, velocity, neighbors, z


def linear_layers(module):
    import torch.nn as nn

    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


def batch_norm_layers(module):
    import torch.nn as nn

    return [layer for layer in module.modules() if isinstance(layer, nn.BatchNorm1d)]


def fill_weights(module):
    import torch

    with torch.no_grad():
        for i, layer in enumerate(linear_layers(module)):
            values = torch.arange(
                layer.weight.numel(),
                dtype=layer.weight.dtype,
            ).reshape_as(layer.weight)

            layer.weight.copy_(torch.sin(values + i + 1) * 0.08)

            if layer.bias is not None:
                values = torch.arange(
                    layer.bias.numel(),
                    dtype=layer.bias.dtype,
                )
                layer.bias.copy_(torch.cos(values + i + 1) * 0.03)

        for layer in batch_norm_layers(module):
            n = layer.num_features
            layer.weight.copy_(torch.linspace(0.8, 1.2, n))
            layer.bias.copy_(torch.linspace(-0.1, 0.1, n))
            layer.running_mean.copy_(torch.linspace(-0.2, 0.2, n))
            layer.running_var.copy_(torch.linspace(0.7, 1.3, n))


def decoder_weights(decoder, official):
    import torch

    weight = torch.tensor(
        [
            [0.02, 0.03, 0.04, 0.05],
            [0.06, 0.07, 0.08, 0.09],
            [0.10, 0.11, 0.12, 0.13],
            [0.14, 0.15, 0.16, 0.17],
            [0.18, 0.19, 0.20, 0.21],
        ]
    )

    layer = (
        decoder.factor_regressor.fc_layers[0][0]
        if official
        else decoder.factor_loadings
    )

    with torch.no_grad():
        layer.weight.copy_(weight)
        if layer.bias is not None:
            layer.bias.zero_()


def midpoint(z, drift, steps=6, t_max=2.0):
    import torch

    dt = t_max / (steps - 1)
    path = [z.clone()]

    for _ in range(steps - 1):
        predictor = z + drift(z) * dt
        z = z + drift(0.5 * (z + predictor)) * dt
        path.append(z.clone())

    return torch.stack(path)


def to_json(values):
    return {
        key: (
            value.detach().cpu().numpy().tolist()
            if hasattr(value, "detach")
            else np.asarray(value).tolist()
        )
        for key, value in values.items()
    }


# ------------------------------------------------------------------
# Official implementation
# ------------------------------------------------------------------


def official():
    import torch
    import torch.nn.functional as F

    import anndata as ad
    import scvi
    from scvelo.core import LinearRegression
    from velvetvae.module import Encoder
    from velvetvae.submodule import MarkovProcess, NeighborhoodConstraint, SDE, VectorField

    try:
        from scvi.nn import LinearDecoderSCVI
    except ImportError:
        from scvi.module._vae import LinearDecoderSCVI

    try:
        from scvi.distributions import ZeroInflatedNegativeBinomial
    except ImportError:
        from scvi.distributions._negative_binomial import ZeroInflatedNegativeBinomial

    torch.manual_seed(0)
    x_np, v_np, k_np, z_np = toy_data()
    out = {}

    # 1. Metabolic labeling.
    total = torch.tensor([[1, 2, 4, 8]], dtype=torch.float32)
    velocity = torch.tensor([[0.5, -0.2, 1, 2]], dtype=torch.float32)
    gamma = torch.tensor([0.2, 0.5, 1, 2])

    factor = (1 - torch.exp(-2 * gamma)) / gamma
    out["metabolic"] = torch.clip(
        factor * (velocity + gamma * total),
        0,
        100,
    )

    # 2. Gamma initialization.
    gamma_x = np.arange(1, 19, dtype=np.float32).reshape(6, 3)
    gamma_y = gamma_x * np.array([0.2, 0.3, 0.4], dtype=np.float32)

    regression = LinearRegression(fit_intercept=False, percentile=[5, 95])
    regression.fit(gamma_x, gamma_y)
    slope = np.asarray(regression.coef_).reshape(-1)

    out["gamma"] = np.clip(-np.log(1 - slope) / 2, 0.01, 10)

    # 3. Encoder.
    encoder = Encoder(
        n_input=4,
        n_output=3,
        n_layers=1,
        n_hidden=8,
        dropout_rate=0,
    )
    fill_weights(encoder)
    encoder.eval()

    with torch.no_grad():
        hidden = encoder.encoder(torch.log1p(torch.from_numpy(x_np)))
        out["encoder"] = encoder.mean_encoder(hidden)

    # 4. Decoder + 5. velocity.
    z = torch.from_numpy(z_np)
    library = torch.tensor([[2.0], [2.5], [3.0]])

    decoder = LinearDecoderSCVI(
        n_input=4,
        n_output=5,
        n_cat_list=None,
        use_batch_norm=False,
        use_layer_norm=False,
        bias=False,
    )
    decoder_weights(decoder, official=True)
    decoder.eval()

    field = VectorField(latent_dim=4, n_hidden=8, n_layers=3)
    fill_weights(field)
    field.eval()

    with torch.no_grad():
        rate = decoder(dispersion="gene", z=z, library=library)[2]
        latent_velocity = field(z)
        future = decoder(
            dispersion="gene",
            z=z + latent_velocity,
            library=library,
        )[2]

    out["decoder"] = rate
    out["latent_velocity"] = latent_velocity
    out["gene_velocity"] = future - rate

    # 6. Transition probabilities.
    neighbors = torch.from_numpy(k_np)
    adata = ad.AnnData(X=x_np)
    adata.layers["total"] = x_np
    adata.layers["velocity"] = v_np
    adata.obsm["knn_index"] = k_np

    process = object.__new__(MarkovProcess)
    process.device = torch.device("cpu")
    process.deterministic_scaling = 10.0

    transition = process.velocity_transition_matrix(adata)
    out["transition"] = torch.gather(transition, 1, neighbors)

    # 7. Neighborhood constraint.
    x = torch.from_numpy(x_np)
    velocity = torch.from_numpy(v_np)

    constraint = NeighborhoodConstraint(
        X=x_np,
        inverse_sigma=10.0,
        similarity_strength=0.2,
    )

    projection = constraint.project(x=x, v=velocity, k=neighbors)
    out["neighborhood"] = projection
    out["neighborhood_loss"] = 1 - F.cosine_similarity(
        velocity,
        projection,
        dim=1,
    ).mean()

    # 8. Stage-1 loss on fixed intermediate values.
    total = torch.tensor([[10, 0, 5], [0, 15, 9]], dtype=torch.float32)
    new = torch.tensor([[3, 0, 1], [0, 4, 3]], dtype=torch.float32)
    predicted_new = torch.tensor([[2.5, 0.2, 1.2], [0.3, 3.5, 2.7]])
    mean = torch.tensor([[0.2, -0.3], [0.7, 0.1]])
    log_var = torch.tensor([[-0.2, 0.1], [0.2, -0.1]])
    rate = torch.tensor([[8, 2, 6], [2, 14, 8]], dtype=torch.float32)
    theta = torch.tensor([[2, 3, 1.5], [2, 3, 1.5]])
    dropout = torch.tensor([[-1, -0.5, -1.5], [-0.7, -1.1, -0.9]])

    zinb = ZeroInflatedNegativeBinomial(
        mu=rate,
        theta=theta,
        zi_logits=dropout,
    )

    reconstruction = -zinb.log_prob(total).sum(-1).mean()
    kl = 0.5 * (
        mean.square() + log_var.exp() - 1 - log_var
    ).sum(-1).mean()
    velocity_loss = F.mse_loss(
        torch.log1p(new),
        torch.log1p(predicted_new),
    )

    out["stage1_loss"] = (
        reconstruction
        + 0.1 * kl
        + 10 * velocity_loss
    )

    # 9. SDE coefficients and deterministic midpoint behavior.
    z0 = torch.tensor([[0.2, 0.5, -0.1], [0.7, -0.2, 0.4]])
    sde = SDE(
        latent_dim=3,
        noise_scalar=0.15,
        n_layers=3,
        n_hidden=8,
        device="cpu",
    )
    fill_weights(sde.drift)
    sde.eval()

    with torch.no_grad():
        out["sde_drift"] = sde.f(torch.tensor(0.0), z0)
        out["sde_diffusion"] = sde.g(torch.tensor(0.0), z0)
        out["sde_zero_noise"] = midpoint(
            z0,
            lambda state: sde.f(torch.tensor(0.0), state),
        )

    return {
        "meta": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "scvi": getattr(scvi, "__version__", "unknown"),
        },
        "values": to_json(out),
    }


# ------------------------------------------------------------------
# TrajectoryFlow implementation
# ------------------------------------------------------------------


def matching_field(n_latent):
    from trajectoryflow.models.baselines.velvet.dynamics import LatentVectorField

    return LatentVectorField(
        n_latent=n_latent,
        n_hidden=8,
        n_layers=3,
    )


def ours():
    import torch

    from trajectoryflow.models.baselines.velvet.config import VelvetSDEConfig, VelvetVAEConfig
    from trajectoryflow.models.baselines.velvet.dynamics import (
        MetabolicLabelingModel,
        estimate_gamma_extreme_regression,
    )
    from trajectoryflow.models.baselines.velvet.model import VelvetVAE
    from trajectoryflow.models.baselines.velvet.neighborhood import (
        neighborhood_constraint_loss,
        neighborhood_projection,
        transition_probabilities,
    )
    from trajectoryflow.models.baselines.velvet.sde import VelvetSDE
    from trajectoryflow.models.baselines.velvet.vae import Encoder, LinearZINBDecoder

    torch.manual_seed(0)
    x_np, v_np, k_np, z_np = toy_data()
    out = {}

    # 1. Metabolic labeling.
    total = torch.tensor([[1, 2, 4, 8]], dtype=torch.float32)
    velocity = torch.tensor([[0.5, -0.2, 1, 2]], dtype=torch.float32)
    gamma = torch.tensor([0.2, 0.5, 1, 2])

    metabolic = MetabolicLabelingModel(
        n_genes=4,
        labelling_time=2,
    )
    metabolic.set_gamma(gamma)
    out["metabolic"] = metabolic.predict_new(total, velocity)

    # 2. Gamma initialization.
    gamma_x = torch.arange(1, 19, dtype=torch.float32).reshape(6, 3)
    gamma_y = gamma_x * torch.tensor([0.2, 0.3, 0.4])

    parameters = inspect.signature(
        estimate_gamma_extreme_regression
    ).parameters

    kwargs = {
        "total": gamma_x,
        "new": gamma_y,
        "labelling_time": 2.0,
        "quantile": 0.95,
    }

    if "ratio_eps" in parameters:
        kwargs["ratio_eps"] = 1e-6
    elif "eps" in parameters:
        kwargs["eps"] = 1e-6

    out["gamma"] = estimate_gamma_extreme_regression(**kwargs)

    # 3. Encoder.
    encoder = Encoder(
        n_genes=4,
        n_hidden=8,
        n_latent=3,
        n_layers=1,
        dropout_rate=0,
    )
    fill_weights(encoder)
    encoder.eval()

    with torch.no_grad():
        out["encoder"] = encoder(
            torch.log1p(torch.from_numpy(x_np))
        )[0]

    # 4. Decoder + 5. velocity.
    z = torch.from_numpy(z_np)
    library = torch.tensor([[2.0], [2.5], [3.0]])

    decoder = LinearZINBDecoder(n_latent=4, n_genes=5)
    decoder_weights(decoder, official=False)
    decoder.eval()

    field = matching_field(4)
    fill_weights(field)
    field.eval()

    with torch.no_grad():
        rate = decoder(z=z, log_library=library)["mean"]
        latent_velocity = field(z)
        future = decoder(
            z=z + latent_velocity,
            log_library=library,
        )["mean"]

    out["decoder"] = rate
    out["latent_velocity"] = latent_velocity
    out["gene_velocity"] = future - rate

    # 6. Transition probabilities.
    x = torch.from_numpy(x_np)
    velocity = torch.from_numpy(v_np)
    neighbors = torch.from_numpy(k_np)
    sigma = 1 / math.sqrt(10)

    out["transition"] = transition_probabilities(
        z=x,
        velocity=velocity,
        neighbor_indices=neighbors,
        sigma=sigma,
    )

    # 7. Neighborhood constraint.
    neighbor_x = x[neighbors]

    out["neighborhood"] = neighborhood_projection(
        z=x,
        velocity=velocity,
        neighbor_z=neighbor_x,
        sigma=sigma,
    )
    out["neighborhood_loss"] = neighborhood_constraint_loss(
        z=x,
        velocity=velocity,
        neighbor_z=neighbor_x,
        sigma=sigma,
    )

    # 8. Actual TrajectoryFlow stage-1 loss with fixed intermediates.
    total = torch.tensor([[10, 0, 5], [0, 15, 9]], dtype=torch.float32)
    new = torch.tensor([[3, 0, 1], [0, 4, 3]], dtype=torch.float32)
    predicted_new = torch.tensor([[2.5, 0.2, 1.2], [0.3, 3.5, 2.7]])
    mean = torch.tensor([[0.2, -0.3], [0.7, 0.1]])
    log_var = torch.tensor([[-0.2, 0.1], [0.2, -0.1]])
    rate = torch.tensor([[8, 2, 6], [2, 14, 8]], dtype=torch.float32)
    theta = torch.tensor([[2, 3, 1.5], [2, 3, 1.5]])
    dropout = torch.tensor([[-1, -0.5, -1.5], [-0.7, -1.1, -0.9]])

    model = VelvetVAE(
        n_genes=3,
        config=VelvetVAEConfig(
            n_hidden=8,
            n_latent=2,
            vector_hidden=8,
            vector_layers=1,
            dropout_rate=0,
            velocity_loss_weight=10,
        ),
    )

    def encode(self, total, sample=True):
        z = torch.zeros((len(total), 2))
        return mean, log_var, z

    def decode(self, z, log_library):
        return {
            "mean": rate,
            "scale": torch.softmax(rate, dim=-1),
            "inverse_dispersion": theta,
            "dropout_logits": dropout,
        }

    def predicted(self, z, log_library):
        return predicted_new, rate, torch.zeros_like(z), torch.zeros_like(rate)

    model.encode = types.MethodType(encode, model)
    model.decode = types.MethodType(decode, model)
    model.predicted_new = types.MethodType(predicted, model)

    with torch.no_grad():
        out["stage1_loss"] = model.stage1_loss(
            total=total,
            new=new,
        ).loss

    # 9. SDE.
    z0 = torch.tensor([[0.2, 0.5, -0.1], [0.7, -0.2, 0.4]])
    field = matching_field(3)
    fill_weights(field)
    field.eval()

    class Holder(torch.nn.Module):
        def __init__(self, vector_field):
            super().__init__()
            self.vector_field = vector_field

    sde = VelvetSDE(
        Holder(field),
        VelvetSDEConfig(noise_scalar=0.15),
    )

    with torch.no_grad():
        out["sde_drift"] = sde.drift(z0)
        out["sde_diffusion"] = torch.full_like(z0, 0.15)
        out["sde_zero_noise"] = sde.simulate(
            z0=z0,
            n_simulations=1,
            n_steps=6,
            t_max=2.0,
            noise_scalar=0,
        )[:, 0]

    return {
        "meta": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "values": to_json(out),
    }


# ------------------------------------------------------------------
# Compare
# ------------------------------------------------------------------


CHECKS = (
    ("metabolic", "Metabolic labeling"),
    ("gamma", "Gamma initialization"),
    ("encoder", "Encoder"),
    ("decoder", "Decoder"),
    ("latent_velocity", "Latent vector field"),
    ("gene_velocity", "Gene velocity"),
    ("transition", "Transition probabilities"),
    ("neighborhood", "Neighborhood projection"),
    ("neighborhood_loss", "Neighborhood loss"),
    ("stage1_loss", "Stage-1 loss"),
    ("sde_drift", "SDE drift"),
    ("sde_diffusion", "SDE diffusion"),
    ("sde_zero_noise", "SDE zero-noise integration"),
)


def compare(reference, current):
    failed = []

    print()
    print("VELVET EQUIVALENCE")
    print("=" * 68)
    print(
        f"official: Python {reference['meta']['python']}, "
        f"Torch {reference['meta']['torch']}, scvi {reference['meta']['scvi']}"
    )
    print(
        f"ours:     Python {current['meta']['python']}, "
        f"Torch {current['meta']['torch']}"
    )
    print()

    for key, name in CHECKS:
        expected = np.asarray(reference["values"][key])
        actual = np.asarray(current["values"][key])

        atol = 2e-5 if key == "encoder" else ATOL
        rtol = RTOL

        passed = (
            expected.shape == actual.shape
            and np.isfinite(expected).all()
            and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=rtol, atol=atol)
        )

        max_error = (
            np.max(np.abs(actual - expected))
            if expected.shape == actual.shape
            else np.inf
        )

        print(
            f"{'PASS' if passed else 'FAIL':<4}  "
            f"{name:<32} max_abs={max_error:.3e}"
        )

        if not passed:
            failed.append(name)

    print("-" * 68)

    if failed:
        print(f"OVERALL: FAIL — {len(failed)} checks differ")
        for name in failed:
            print(f"  - {name}")
        return False

    print(f"OVERALL: PASS — all {len(CHECKS)} checks match")
    return True




# ------------------------------------------------------------------
# TrajectoryFlow public-API / benchmark regression checks
# ------------------------------------------------------------------


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _expect_raises(exception_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception_type:
        return
    except Exception as error:
        raise AssertionError(
            f"Expected {exception_type.__name__}, got "
            f"{type(error).__name__}: {error}"
        ) from error

    raise AssertionError(
        f"Expected {exception_type.__name__}, but no exception was raised."
    )


def _mark_fitted(model):
    """
    Support both BaseTrajectoryModel implementations used during development:
    direct `is_fitted` state and a property backed by `_is_fitted`.
    """
    try:
        model._is_fitted = True
    except Exception:
        pass

    try:
        model.is_fitted = True
    except Exception:
        pass


def _velocity_api_checks():
    import torch

    from trajectoryflow.models.baselines.velvet.config import VelvetVAEConfig
    from trajectoryflow.models.baselines.velvet.model import VelvetVAE
    from trajectoryflow.models.baselines.velvet.vae import observed_log_library

    torch.manual_seed(7)

    model = VelvetVAE(
        n_genes=4,
        config=VelvetVAEConfig(
            n_hidden=8,
            n_latent=3,
            vector_hidden=8,
            vector_layers=1,
            dropout_rate=0.5,
            initialize_gamma=False,
            seed=7,
        ),
    )
    fill_weights(model)

    total = torch.tensor(toy_data()[0], dtype=torch.float32)

    # Deliberately start in training mode: infer_gene_velocity() must switch
    # to eval internally and restore the original state afterwards.
    model.train()

    first = model.infer_gene_velocity(
        total,
        sample_latent=False,
    )
    second = model.infer_gene_velocity(
        total,
        sample_latent=False,
    )

    _require(first.shape == total.shape, "Velocity shape differs from total RNA.")
    _require(torch.isfinite(first).all(), "Velocity contains non-finite values.")
    _require(not first.requires_grad, "Inference velocity must not require gradients.")
    _require(model.training, "infer_gene_velocity() did not restore training mode.")
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    # Public inference must be exactly the same deterministic calculation as
    # encode(posterior mean) -> gene_velocity().
    was_training = model.training
    model.eval()

    with torch.no_grad():
        _, _, z = model.encode(
            total=total,
            sample=False,
        )
        log_library = observed_log_library(
            total,
            eps=model.config.eps,
        )
        expected = model.gene_velocity(
            z=z,
            log_library=log_library,
        )[2]

    model.train(was_training)
    torch.testing.assert_close(first, expected, rtol=RTOL, atol=ATOL)

    # Invalid count-like inputs must fail loudly.
    _expect_raises(
        ValueError,
        model.infer_gene_velocity,
        torch.ones(4),
    )
    _expect_raises(
        ValueError,
        model.infer_gene_velocity,
        torch.ones((3, 5)),
    )

    negative = total.clone()
    negative[0, 0] = -1
    _expect_raises(
        ValueError,
        model.infer_gene_velocity,
        negative,
    )

    non_finite = total.clone()
    non_finite[0, 0] = float("nan")
    _expect_raises(
        ValueError,
        model.infer_gene_velocity,
        non_finite,
    )


def _baseline_velocity_checks():
    import torch

    from trajectoryflow.experiment.models import (
        BuiltExperimentModel,
        VelvetExperimentAdapter,
    )
    from trajectoryflow.models.baselines.velvet import (
        VelvetBaseline,
        VelvetSDEConfig,
        VelvetVAEConfig,
    )

    torch.manual_seed(11)

    baseline = VelvetBaseline(
        n_genes=4,
        hours_per_sde_unit=2.0,
        vae_config=VelvetVAEConfig(
            n_hidden=8,
            n_latent=3,
            vector_hidden=8,
            vector_layers=1,
            dropout_rate=0.4,
            initialize_gamma=False,
            seed=11,
        ),
        sde_config=VelvetSDEConfig(
            epochs=1,
            seed=11,
        ),
    )

    total = torch.tensor(toy_data()[0], dtype=torch.float32)

    _expect_raises(
        RuntimeError,
        baseline.predict_velocity,
        total,
    )

    _mark_fitted(baseline)

    expected = baseline.velvet.infer_gene_velocity(total)
    actual = baseline.predict_velocity(total)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _require(actual.shape == total.shape, "Baseline velocity has wrong shape.")
    _require(torch.isfinite(actual).all(), "Baseline velocity is non-finite.")

    adapter = VelvetExperimentAdapter()
    built = BuiltExperimentModel(
        model=baseline,
        trainer=None,
        device=torch.device("cpu"),
    )

    _require(
        adapter.supports_velocity(built),
        "VelvetExperimentAdapter does not advertise velocity support.",
    )

    adapter_velocity = adapter.predict_velocity(
        built=built,
        source=total,
    )

    torch.testing.assert_close(
        adapter_velocity,
        actual,
        rtol=0,
        atol=0,
    )


def _velocity_reference_checks():
    import numpy as np

    from trajectoryflow.experiment.evaluation import (
        VelocityReference,
        load_velocity_reference,
        project_velocity_to_reference,
        save_velocity_reference,
        velocity_alignment_metrics,
    )

    expression = np.array(
        [
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )

    velocity = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 2.0, 0.0, 0.5],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    components = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    reference = VelocityReference(
        name="pts",
        cell_ids=np.array(["c0", "c1", "c2"]),
        vectors=velocity[:, :2],
        pca_components=components,
        pca_mean=np.zeros(4),
        positions=expression[:, :2],
        groups=np.array(["A", "A", "B"]),
        genes=np.array(["g0", "g1", "g2", "g3"]),
        expression_transform="none",
        projection_mode="linear",
    )

    positions, projected = project_velocity_to_reference(
        expression=expression,
        velocity=velocity,
        reference=reference,
    )

    np.testing.assert_allclose(
        positions,
        expression[:, :2],
        rtol=0,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        projected,
        velocity[:, :2],
        rtol=0,
        atol=ATOL,
    )

    # Finite-difference projection must reduce to the same result for an
    # identity/no-transform expression space.
    finite_difference = VelocityReference(
        name="pts_fd",
        cell_ids=reference.cell_ids,
        vectors=reference.vectors,
        pca_components=reference.pca_components,
        pca_mean=reference.pca_mean,
        positions=reference.positions,
        groups=reference.groups,
        genes=reference.genes,
        expression_transform="none",
        projection_mode="finite_difference",
        velocity_epsilon=1e-4,
    )

    _, projected_fd = project_velocity_to_reference(
        expression=expression,
        velocity=velocity,
        reference=finite_difference,
    )

    np.testing.assert_allclose(
        projected_fd,
        velocity[:, :2],
        rtol=1e-8,
        atol=1e-8,
    )

    # Cosine semantics: scaling does not matter, opposite direction is -1,
    # and zero predicted velocity is invalid rather than being scored as 0.
    predicted = np.array(
        [
            [2.0, 0.0],
            [0.0, -3.0],
            [0.0, 0.0],
        ]
    )
    target = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    metrics, cells = velocity_alignment_metrics(
        predicted=predicted,
        reference=target,
        cell_ids=np.array(["a", "b", "c"]),
    )

    scores = cells["cosine_similarity"].to_numpy()
    _require(np.isclose(scores[0], 1.0), "Perfect alignment must score +1.")
    _require(np.isclose(scores[1], -1.0), "Opposite alignment must score -1.")
    _require(np.isnan(scores[2]), "Zero velocity must have an undefined score.")
    _require(
        cells["valid"].tolist() == [True, True, False],
        "Velocity validity mask is incorrect.",
    )

    overall_valid = metrics[
        (metrics["metric"] == "valid_fraction")
        & metrics["group"].isna()
    ]["value"].iloc[0]

    _require(
        np.isclose(overall_valid, 2 / 3),
        "valid_fraction does not reflect zero-velocity cells.",
    )

    # CBD mode: reference vectors are means of unit source->target neighbour
    # directions. Dotting a normalized model velocity with that vector is
    # exactly the mean neighbour-wise cosine.
    cbd_predicted = np.array(
        [
            [2.0, 0.0],
            [0.0, 4.0],
        ]
    )
    cbd_reference = np.array(
        [
            [0.75, 0.25],
            [-0.50, 0.50],
        ]
    )

    _, cbd_cells = velocity_alignment_metrics(
        predicted=cbd_predicted,
        reference=cbd_reference,
        cell_ids=np.array(["x", "y"]),
        groups=np.array(["A_to_B", "C_to_D"]),
        alignment_mode="mean_unit_direction",
    )

    np.testing.assert_allclose(
        cbd_cells["cosine_similarity"].to_numpy(),
        [0.75, 0.50],
        rtol=0,
        atol=ATOL,
    )

    # Reference persistence must preserve the PCA basis and alignment mode.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reference.npz"
        save_velocity_reference(path, reference)
        loaded = load_velocity_reference(path)

    _require(loaded.name == reference.name, "Reference name changed on roundtrip.")
    _require(
        loaded.alignment_mode == reference.alignment_mode,
        "Reference alignment mode changed on roundtrip.",
    )
    np.testing.assert_array_equal(loaded.cell_ids, reference.cell_ids)
    np.testing.assert_allclose(
        loaded.pca_components,
        reference.pca_components,
        rtol=0,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        loaded.vectors,
        reference.vectors,
        rtol=0,
        atol=ATOL,
    )


def _trainer_velocity_only_check():
    import numpy as np
    import pandas as pd
    import torch
    from scipy import sparse

    from trajectoryflow.models.baselines.velvet import (
        VelvetBaseline,
        VelvetSDEConfig,
        VelvetVAEConfig,
    )
    from trajectoryflow.models.baselines.velvet.data import VelvetData
    from trajectoryflow.training.velvet import VelvetTrainer

    data = VelvetData(
        total=sparse.csr_matrix(
            np.ones((4, 3), dtype=np.float32)
        ),
        new=sparse.csr_matrix(
            np.ones((4, 3), dtype=np.float32)
        ),
        obs=pd.DataFrame(
            {
                "cell_id": ["a", "b", "c", "d"],
                "timepoint": ["5h"] * 4,
            }
        ),
        timepoints=("5h",),
    )

    baseline = VelvetBaseline(
        n_genes=3,
        hours_per_sde_unit=2.0,
        vae_config=VelvetVAEConfig(
            n_hidden=4,
            n_latent=2,
            vector_hidden=4,
            vector_layers=1,
            n_neighbors=2,
            stage1_epochs=1,
            stage2_epochs=1,
            initialize_gamma=False,
            seed=3,
        ),
        sde_config=VelvetSDEConfig(
            epochs=1,
            seed=3,
        ),
    )

    neighbors = np.array(
        [
            [1, 2],
            [0, 2],
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    trainer = VelvetTrainer(
        data=data,
        neighbor_indices=neighbors,
        device="cpu",
        train_sde=False,
    )

    calls = {
        "gamma": 0,
        "stage1": 0,
        "latent": 0,
        "stage2": 0,
        "sde": 0,
    }

    def initialize_gamma(self, model):
        calls["gamma"] += 1

    def stage1(self, model):
        calls["stage1"] += 1

    def latent_all(self, model):
        calls["latent"] += 1
        return torch.zeros(
            (self.data.n_cells, model.velvet.config.n_latent),
            dtype=torch.float32,
        )

    def stage2(
        self,
        baseline,
        all_z_cpu,
        neighbor_indices,
    ):
        calls["stage2"] += 1

    def sde(
        self,
        baseline,
        all_z_cpu,
    ):
        calls["sde"] += 1

    trainer._initialize_gamma = types.MethodType(
        initialize_gamma,
        trainer,
    )
    trainer._train_stage1 = types.MethodType(
        stage1,
        trainer,
    )
    trainer._latent_all = types.MethodType(
        latent_all,
        trainer,
    )
    trainer._train_stage2 = types.MethodType(
        stage2,
        trainer,
    )
    trainer._train_sde = types.MethodType(
        sde,
        trainer,
    )

    trainer.fit(baseline)

    _require(calls["gamma"] == 1, "Gamma initialization path was not entered.")
    _require(calls["stage1"] == 1, "Stage 1 path was not entered.")
    _require(calls["latent"] == 1, "Latent embedding path was not entered.")
    _require(calls["stage2"] == 1, "Stage 2 path was not entered.")
    _require(
        calls["sde"] == 0,
        "train_sde=False still executed SDE training.",
    )


SELF_CHECKS = (
    ("Velocity inference API", _velocity_api_checks),
    ("VelvetBaseline + adapter API", _baseline_velocity_checks),
    ("Velocity reference/projection/scoring", _velocity_reference_checks),
    ("Velocity-only training skips SDE", _trainer_velocity_only_check),
)


def run_trajectoryflow_checks(verbose=True):
    failed = []

    if verbose:
        print()
        print("TRAJECTORYFLOW VELOCITY / BENCHMARK CHECKS")
        print("=" * 68)

    for name, function in SELF_CHECKS:
        try:
            function()
            passed = True
            message = ""
        except Exception as error:
            passed = False
            message = f"{type(error).__name__}: {error}"
            failed.append((name, message))

        if verbose:
            print(
                f"{'PASS' if passed else 'FAIL':<4}  "
                f"{name:<40} {message}"
            )

    if verbose:
        print("-" * 68)

        if failed:
            print(
                f"OVERALL: FAIL — {len(failed)} "
                "TrajectoryFlow integration check(s) failed"
            )
            for name, message in failed:
                print(f"  - {name}: {message}")
        else:
            print(
                f"OVERALL: PASS — all {len(SELF_CHECKS)} "
                "TrajectoryFlow integration checks pass"
            )

    return not failed


# ------------------------------------------------------------------
# Isolated official installation
# ------------------------------------------------------------------


def _installed_reference_commit(python):
    code = r"""
import json
from importlib.metadata import distribution

for name in ("velvetvae", "velvetVAE"):
    try:
        raw = distribution(name).read_text("direct_url.json")
    except Exception:
        continue

    if not raw:
        continue

    try:
        data = json.loads(raw)
    except Exception:
        continue

    print(data.get("vcs_info", {}).get("commit_id", ""))
    break
"""

    process = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
    )

    if process.returncode:
        return None

    value = process.stdout.strip().splitlines()
    return value[-1].strip() if value else None


def _reference_import_check(python):
    """
    Check the complete runtime import chain needed by official Velvet.

    pytorch-lightning 1.7.7 imports pkg_resources. pkg_resources was removed
    from setuptools >=82, so the reference environment intentionally pins an
    older setuptools version.
    """
    return subprocess.run(
        [
            str(python),
            "-c",
            (
                "import pkg_resources; "
                "import velvetvae, scvi, scvelo; "
                "print('reference imports OK')"
            ),
        ],
        capture_output=True,
        text=True,
    )


def _uv_executable():
    uv = shutil.which("uv")

    if uv is None:
        raise RuntimeError(
            "Install uv once so the script can create/repair the isolated "
            "official Velvet environment, or run with --self-only."
        )

    return uv


def _install_legacy_setuptools(python):
    uv = _uv_executable()

    print(
        "Installing reference-compatible setuptools "
        f"{REFERENCE_SETUPTOOLS} ..."
    )

    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--reinstall",
            f"setuptools=={REFERENCE_SETUPTOOLS}",
        ],
        check=True,
    )


def _install_torchcubicspline(python):
    uv = _uv_executable()

    print("Installing missing official dependency torchcubicspline ...")

    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            TORCHCUBICSPLINE_REPO,
        ],
        check=True,
    )


def _ensure_reference_imports(python, repair=True):
    """
    Verify the complete import chain required by official Velvet and repair
    known omissions/incompatibilities in the upstream reference environment.

    Repairs are restricted to dependencies required by the pinned official
    code; they never modify the TrajectoryFlow environment.
    """
    repaired = set()

    for _ in range(4):
        result = _reference_import_check(python)

        if result.returncode == 0:
            return

        stderr = result.stderr

        missing_pkg_resources = (
            "No module named 'pkg_resources'" in stderr
            or 'No module named "pkg_resources"' in stderr
        )
        missing_torchcubicspline = (
            "No module named 'torchcubicspline'" in stderr
            or 'No module named "torchcubicspline"' in stderr
        )

        if not repair:
            break

        if missing_pkg_resources and "setuptools" not in repaired:
            print(
                "Reference environment uses a modern setuptools without "
                "pkg_resources; repairing it in place."
            )
            _install_legacy_setuptools(python)
            repaired.add("setuptools")
            continue

        if (
            missing_torchcubicspline
            and "torchcubicspline" not in repaired
        ):
            print(
                "Official Velvet imports torchcubicspline but its package "
                "metadata did not install it; repairing the reference "
                "environment in place."
            )
            _install_torchcubicspline(python)
            repaired.add("torchcubicspline")
            continue

        break

    raise RuntimeError(
        "Official Velvet reference environment failed its import check.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def reference_python(root, rebuild=False):
    env = root / ".venv-velvet-reference"
    python = env / "bin" / "python"

    if python.exists() and not rebuild:
        try:
            _ensure_reference_imports(
                python,
                repair=True,
            )

            commit = _installed_reference_commit(python)

            if commit == OFFICIAL_REF:
                return python

            print(
                "Existing Velvet reference environment is not verifiably "
                f"pinned to {OFFICIAL_REF}; rebuilding it."
            )

        except RuntimeError as error:
            print(
                "Existing Velvet reference environment is unusable after "
                f"repair: {error}"
            )
            print("Rebuilding the reference environment.")

    uv = _uv_executable()

    print("Creating pinned .venv-velvet-reference ...")

    if env.exists():
        shutil.rmtree(env)

    subprocess.run(
        [uv, "venv", "--python", "3.10", str(env)],
        check=True,
    )

    # Install the legacy setuptools pin in the same resolution transaction as
    # official Velvet so a newly resolved modern setuptools cannot remove
    # pkg_resources again.
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            f"setuptools=={REFERENCE_SETUPTOOLS}",
            TORCHCUBICSPLINE_REPO,
            f"git+{OFFICIAL_REPO}@{OFFICIAL_REF}",
        ],
        check=True,
    )

    _ensure_reference_imports(
        python,
        repair=False,
    )

    commit = _installed_reference_commit(python)

    if commit not in (None, "", OFFICIAL_REF):
        raise RuntimeError(
            "Installed Velvet reference commit does not match the pinned "
            f"commit: expected {OFFICIAL_REF}, got {commit}."
        )

    return python


def _parse_worker_json(stdout, worker):
    # Dependencies occasionally print informational messages to stdout.
    # The worker JSON itself is emitted as one line, so parse from the end.
    for line in reversed(stdout.splitlines()):
        line = line.strip()

        if not line:
            continue

        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        f"{worker} worker produced no parseable JSON output:\n{stdout}"
    )


def run_worker(python, script, worker, root):
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)

    if worker == "official":
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(root / "src")

    process = subprocess.run(
        [str(python), str(script), worker],
        env=env,
        capture_output=True,
        text=True,
    )

    if process.returncode:
        raise RuntimeError(
            f"{worker} worker failed:\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    return _parse_worker_json(
        process.stdout,
        worker=worker,
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def _project_root():
    script = Path(__file__).resolve()
    candidate = script.parent.parent

    if (candidate / "src" / "trajectoryflow").exists():
        return candidate

    cwd = Path.cwd()

    if (cwd / "src" / "trajectoryflow").exists():
        return cwd

    raise RuntimeError(
        "Could not locate the TrajectoryFlow project root. Run this script "
        "from the repository or place it under scripts/."
    )


def main():
    # Internal worker modes.
    if len(sys.argv) == 2 and sys.argv[1] == "official":
        print(json.dumps(official()))
        return 0

    if len(sys.argv) == 2 and sys.argv[1] == "ours":
        print(json.dumps(ours()))
        return 0

    allowed = {"--self-only", "--rebuild-reference"}
    unknown = set(sys.argv[1:]) - allowed

    if unknown:
        print(
            "Usage: python scripts/validate_velvet_equivalence.py "
            "[--self-only] [--rebuild-reference]",
            file=sys.stderr,
        )
        print(
            f"Unknown argument(s): {sorted(unknown)}",
            file=sys.stderr,
        )
        return 2

    try:
        root = _project_root()

        self_ok = run_trajectoryflow_checks(
            verbose=True,
        )

        if "--self-only" in sys.argv:
            return 0 if self_ok else 1

        python = reference_python(
            root,
            rebuild="--rebuild-reference" in sys.argv,
        )

        print()
        print("Running official Velvet...")
        reference = run_worker(
            python,
            Path(__file__).resolve(),
            "official",
            root,
        )

        print("Running TrajectoryFlow Velvet...")
        current = run_worker(
            Path(sys.executable),
            Path(__file__).resolve(),
            "ours",
            root,
        )

        official_ok = compare(reference, current)

        print()
        print("=" * 68)

        if official_ok and self_ok:
            print(
                "FINAL: PASS — official numerical equivalence and "
                "TrajectoryFlow integration checks pass"
            )
            return 0

        print("FINAL: FAIL")

        if not official_ok:
            print("  - one or more official Velvet equivalence checks failed")
        if not self_ok:
            print("  - one or more TrajectoryFlow integration checks failed")

        return 1

    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
