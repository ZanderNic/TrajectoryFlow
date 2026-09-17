# std-lib imports
from pathlib import Path

# 3 party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# package imports
from trajectoryflow.plotting.dual_scale import (
    plot_alignment_cosine,
    plot_gene_baselines,
    plot_gene_mean_agreement,
    plot_latent_transition,
    plot_loss_group,
    plot_total_loss_history,
    plot_transition_uncertainty,
)


def training_frame() -> pd.DataFrame:
    rows = []
    step = 0
    for stage, metrics in (
        ("local_pretrain", {"total": 3.0, "reconstruction": 2.0, "new_rna": 1.0}),
        ("joint/local", {"total": 2.0, "reconstruction": 1.2, "new_rna": 0.8}),
        ("joint/global", {"total": 1.5, "future_sliced_wasserstein": 1.0, "future_mmd": 0.5}),
    ):
        step += 1
        for metric, value in metrics.items():
            rows.append({"granularity": "step", "stage": stage, "step": step, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def test_training_plots_write_files(tmp_path: Path):
    metrics = training_frame()
    total = tmp_path / "total.png"
    local = tmp_path / "local.png"
    fig = plot_total_loss_history(metrics, total, smooth_window=1)
    plt.close(fig)
    fig = plot_loss_group(metrics, ("reconstruction", "new_rna"), "local", "Local", local, smooth_window=1)
    plt.close(fig)
    assert total.exists() and local.exists()


def test_checkpoint_diagnostic_plots_write_files(tmp_path: Path):
    genes = pd.DataFrame({"production_base": [0.2, 0.4], "degradation_base": [0.1, 0.3]})
    fig = plot_gene_baselines(genes, tmp_path / "genes.png")
    plt.close(fig)

    observed = np.array([[1.0, 2.0], [2.0, 3.0]])
    predicted = observed * 0.9
    fig = plot_gene_mean_agreement(observed, predicted, "Agreement", "x", "y", tmp_path / "agreement.png")
    plt.close(fig)

    cells = pd.DataFrame(
        {
            "transition_mean_norm": [1.0, 2.0],
            "transition_std_norm": [0.4, 0.6],
            "alignment_cosine": [0.2, 0.8],
        }
    )
    fig = plot_transition_uncertainty(cells, tmp_path / "uncertainty.png")
    plt.close(fig)
    fig = plot_alignment_cosine(cells, tmp_path / "alignment.png")
    plt.close(fig)

    source = np.array([[0.0, 0.0, 0.1], [0.2, 0.0, 0.2]])
    target = np.array([[1.0, 0.0, 0.1], [1.2, 0.0, 0.2]])
    future = np.array([[0.8, 0.0, 0.1], [1.0, 0.0, 0.2]])
    fig = plot_latent_transition(source, target, future, tmp_path / "latent.png")
    plt.close(fig)

    assert all((tmp_path / name).exists() for name in ("genes.png", "agreement.png", "uncertainty.png", "alignment.png", "latent.png"))
