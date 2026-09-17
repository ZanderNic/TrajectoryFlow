# std-lib imports
from pathlib import Path
from typing import Iterable

# 3 party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

# package imports


LOCAL_METRICS = ("reconstruction", "new_rna", "gamma_regularization")
GLOBAL_METRICS = (
    "past_direction_alignment",
    "future_sliced_wasserstein",
    "future_mmd",
    "future_gene_sliced_wasserstein",
    "kinetic_direction_alignment",
    "transition_regularization",
    "gamma_regularization",
)


def _save(fig, path: str | Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    return fig


def _steps(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"granularity", "stage", "step", "metric", "value"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Training metrics are missing columns: {sorted(missing)}")
    return metrics.loc[metrics["granularity"] == "step"].copy()


def _smooth(values: pd.Series, window: int) -> pd.Series:
    return values if window <= 1 else values.rolling(window, min_periods=1).mean()


def plot_total_loss_history(
    metrics: pd.DataFrame,
    path: str | Path | None = None,
    smooth_window: int = 20,
):
    frame = _steps(metrics)
    frame = frame.loc[frame["metric"] == "total"].sort_values("step")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for stage, group in frame.groupby("stage", sort=False):
        group = group.sort_values("step")
        ax.plot(group["step"], _smooth(group["value"], smooth_window), label=str(stage))
    ax.set(title="Dual-scale training loss", xlabel="Optimizer step", ylabel="Loss")
    ax.grid(alpha=0.2)
    if frame["stage"].nunique() > 1:
        ax.legend(frameon=False)
    return _save(fig, path)


def plot_loss_group(
    metrics: pd.DataFrame,
    names: Iterable[str],
    stage_contains: str,
    title: str,
    path: str | Path | None = None,
    smooth_window: int = 20,
):
    frame = _steps(metrics)
    frame = frame.loc[frame["stage"].astype(str).str.contains(stage_contains, regex=False)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in names:
        group = frame.loc[frame["metric"] == name].sort_values("step")
        if group.empty:
            continue
        ax.plot(group["step"], _smooth(group["value"], smooth_window), label=name)
    ax.set(title=title, xlabel="Optimizer step", ylabel="Loss")
    ax.grid(alpha=0.2)
    if ax.lines:
        ax.legend(frameon=False)
    return _save(fig, path)


def plot_gene_baselines(
    gene_table: pd.DataFrame,
    path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(gene_table["production_base"].to_numpy(), bins=50, alpha=0.55, label="production α")
    ax.hist(gene_table["degradation_base"].to_numpy(), bins=50, alpha=0.55, label="degradation γ")
    ax.set(title="Learned gene-wise kinetic baselines", xlabel="Rate", ylabel="Genes")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_gene_mean_agreement(
    observed: np.ndarray,
    predicted: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    path: str | Path | None = None,
):
    x = np.log1p(np.asarray(observed, dtype=np.float64).mean(axis=0))
    y = np.log1p(np.asarray(predicted, dtype=np.float64).mean(axis=0))
    finite = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(x[finite], y[finite], s=8, alpha=0.35)
    if finite.any():
        lo = float(min(x[finite].min(), y[finite].min()))
        hi = float(max(x[finite].max(), y[finite].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        corr = np.corrcoef(x[finite], y[finite])[0, 1] if finite.sum() > 1 else np.nan
        ax.text(0.03, 0.97, f"gene-mean r = {corr:.3f}", transform=ax.transAxes, va="top")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.2)
    return _save(fig, path)


def plot_cell_kinetics(
    cells: pd.DataFrame,
    path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if "alpha_mean" in cells:
        ax.hist(cells["alpha_mean"], bins=40, alpha=0.55, label="mean α")
    if "gamma_mean" in cells:
        ax.hist(cells["gamma_mean"], bins=40, alpha=0.55, label="mean γ")
    ax.set(title="Cell-wise inferred kinetics", xlabel="Mean gene-wise rate", ylabel="Cells")
    if ax.patches:
        ax.legend(frameon=False)
    return _save(fig, path)


def plot_transition_uncertainty(
    cells: pd.DataFrame,
    path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(cells["transition_mean_norm"], bins=40, alpha=0.55, label="||E[Δz]||")
    ax.hist(cells["transition_std_norm"], bins=40, alpha=0.55, label="||SD[Δz]||")
    ax.set(title="Latent transition magnitude and uncertainty", xlabel="Latent norm", ylabel="Cells")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_alignment_cosine(
    cells: pd.DataFrame,
    path: str | Path | None = None,
):
    if "alignment_cosine" not in cells or cells["alignment_cosine"].dropna().empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cells["alignment_cosine"].dropna(), bins=40)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set(title="Kinetic direction / transition alignment", xlabel="Cosine similarity", ylabel="Cells", xlim=(-1, 1))
    return _save(fig, path)


def plot_latent_transition(
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    predicted_latent: np.ndarray,
    path: str | Path | None = None,
    title: str = "Latent future transition",
):
    source_latent = np.asarray(source_latent)
    target_latent = np.asarray(target_latent)
    predicted_latent = np.asarray(predicted_latent)
    reference = np.concatenate((source_latent, target_latent), axis=0)
    pca = PCA(n_components=2, random_state=0).fit(reference)
    source_2d = pca.transform(source_latent)
    target_2d = pca.transform(target_latent)
    predicted_2d = pca.transform(predicted_latent)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(source_2d[:, 0], source_2d[:, 1], s=10, alpha=0.35, label="source")
    ax.scatter(target_2d[:, 0], target_2d[:, 1], s=10, alpha=0.35, label="real target")
    ax.scatter(predicted_2d[:, 0], predicted_2d[:, 1], s=10, alpha=0.35, label="predicted target")
    ax.set(title=title, xlabel="PCA 1", ylabel="PCA 2")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_past_reconstruction(
    real_past_latent: np.ndarray,
    reconstructed_past_latent: np.ndarray,
    path: str | Path | None = None,
    title: str = "Past-population reconstruction",
):
    real = np.asarray(real_past_latent)
    reconstructed = np.asarray(reconstructed_past_latent)
    pca = PCA(n_components=2, random_state=0).fit(real)
    real_2d, reconstructed_2d = pca.transform(real), pca.transform(reconstructed)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(real_2d[:, 0], real_2d[:, 1], s=10, alpha=0.4, label="real past")
    ax.scatter(reconstructed_2d[:, 0], reconstructed_2d[:, 1], s=10, alpha=0.4, label="reconstructed past")
    ax.set(title=title, xlabel="PCA 1", ylabel="PCA 2")
    ax.legend(frameon=False)
    return _save(fig, path)
