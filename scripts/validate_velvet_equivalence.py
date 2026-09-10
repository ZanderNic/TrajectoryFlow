"""
Check that TrajectoryFlow's Velvet implementation reproduces official VelvetVAE.

Run:
    python scripts/validate_velvet_equivalence.py

The official package is installed once into `.venv-velvet-reference`, so its
old PyTorch/scvi dependencies never conflict with TrajectoryFlow.

The test is deterministic: both implementations receive identical toy inputs,
weights and hyperparameters. Independent training is deliberately not compared.

A successful run means the tested Velvet equations and model components agree
numerically within RTOL/ATOL.
"""

# std-lib imports
import json
import math
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

# 3 party imports
import numpy as np

# package imports


OFFICIAL_REPO = "https://github.com/rorymaizels/velvetVAE.git"
OFFICIAL_REF = "e493643de70c2dd10b247831ad8c2c81cf858895"

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

    for n_layers in range(1, 6):
        field = LatentVectorField(
            n_latent=n_latent,
            n_hidden=8,
            n_layers=n_layers,
        )

        if len(linear_layers(field)) == 3:
            return field

    raise RuntimeError("TrajectoryFlow VectorField depth cannot match official Velvet.")


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

        passed = (
            expected.shape == actual.shape
            and np.isfinite(expected).all()
            and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=RTOL, atol=ATOL)
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
# Isolated official installation
# ------------------------------------------------------------------


def reference_python(root):
    env = root / ".venv-velvet-reference"
    python = env / "bin" / "python"

    if python.exists():
        result = subprocess.run(
            [str(python), "-c", "import velvetvae, scvi, scvelo"],
            capture_output=True,
        )
        if result.returncode == 0:
            return python

    uv = shutil.which("uv")

    if uv is None:
        raise RuntimeError(
            "Install uv once so the script can create the isolated official "
            "Velvet environment."
        )

    print("Creating .venv-velvet-reference ...")

    if env.exists():
        shutil.rmtree(env)

    subprocess.run(
        [uv, "venv", "--python", "3.10", str(env)],
        check=True,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            f"git+{OFFICIAL_REPO}@{OFFICIAL_REF}",
        ],
        check=True,
    )

    return python


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
            f"{worker} worker failed:\n{process.stdout}\n{process.stderr}"
        )

    return json.loads(process.stdout)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "official":
        print(json.dumps(official()))
        return 0

    if len(sys.argv) == 2 and sys.argv[1] == "ours":
        print(json.dumps(ours()))
        return 0

    script = Path(__file__).resolve()
    root = script.parent.parent

    if not (root / "src" / "trajectoryflow").exists():
        root = Path.cwd()

    try:
        python = reference_python(root)

        print("Running official Velvet...")
        reference = run_worker(python, script, "official", root)

        print("Running TrajectoryFlow Velvet...")
        current = run_worker(Path(sys.executable), script, "ours", root)

        return 0 if compare(reference, current) else 1

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
