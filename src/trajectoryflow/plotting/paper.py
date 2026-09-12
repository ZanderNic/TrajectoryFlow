# std-lib imports
from collections.abc import Mapping, Sequence
from pathlib import Path

# 3 party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

# package imports


def _array(x) -> np.ndarray:
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    if x.ndim != 2 or 0 in x.shape:
        raise ValueError("expression matrices must have non-empty shape [n_cells, n_genes].")
    if not np.isfinite(x).all():
        raise ValueError("expression matrices contain non-finite values.")
    return np.asarray(x, dtype=np.float32)


def _sample_indices(n: int, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or n <= maximum:
        return np.arange(n, dtype=np.int64)
    if maximum < 1:
        raise ValueError("maximum must be >= 1 or None.")
    return np.sort(np.random.default_rng(seed).choice(n, size=maximum, replace=False))


def _transform_expression(x, transform: str, library_size: float) -> np.ndarray:
    x = _array(x)
    if transform == "none":
        return x
    if transform != "library_log1p":
        raise ValueError("transform must be 'none' or 'library_log1p'.")
    if library_size <= 0:
        raise ValueError("library_size must be > 0.")
    if np.any(x < 0):
        raise ValueError("library_log1p requires non-negative expression values.")
    totals = x.sum(axis=1, keepdims=True)
    scaled = np.divide(x, totals, out=np.zeros_like(x), where=totals > 0) * library_size
    return np.log1p(scaled, out=scaled)


def _save(fig, path: str | Path | None, dpi: int = 250) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _scatter(ax, coordinates: np.ndarray, labels=None, size: float = 8, alpha: float = 0.55) -> None:
    if labels is None:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=size, alpha=alpha)
        return
    labels = np.asarray(labels).astype(str)
    if labels.shape != (len(coordinates),):
        raise ValueError("labels must have one value per coordinate row.")
    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=size, alpha=alpha, label=label)


def _as_bool(value) -> bool:
    return value.strip().lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


def _filter_frame(frame: pd.DataFrame, **filters) -> pd.DataFrame:
    result = frame.copy()
    for column, value in filters.items():
        if value is not None:
            if column not in result.columns:
                raise KeyError(f"Missing column {column!r}.")
            result = result.loc[result[column] == value]
    return result


def plot_forecast_comparison(
    source,
    target,
    predictions: Mapping[str, np.ndarray],
    path: str | Path | None = None,
    source_labels=None,
    target_labels=None,
    source_title: str = "Source",
    target_title: str = "True target",
    transform: str = "library_log1p",
    library_size: float = 10_000.0,
    max_cells: int | None = 3000,
    seed: int = 0,
    title: str | None = None,
):
    """Plot source, true target and model predictions in one fixed two-dimensional PCA space.

    PCA is fitted on real source and target cells only. Predictions are transformed afterwards, so
    a model cannot improve its visual appearance by changing the embedding used for comparison.
    Each prediction must have shape ``[n_cells, n_genes]``; for stochastic benchmark output choose
    one population sample before calling this function rather than averaging generated cells across
    samples, because generated cell indices do not represent paired cells across stochastic draws.
    """
    if not predictions:
        raise ValueError("predictions must contain at least one model.")

    source, target = _array(source), _array(target)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same number of genes.")
    prediction_arrays = {name: _array(values) for name, values in predictions.items()}
    if any(values.shape[1] != source.shape[1] for values in prediction_arrays.values()):
        raise ValueError("all predictions must use the same genes as source and target.")

    source_idx = _sample_indices(len(source), max_cells, seed)
    target_idx = _sample_indices(len(target), max_cells, seed + 1)
    source_labels = None if source_labels is None else np.asarray(source_labels)[source_idx]
    target_labels = None if target_labels is None else np.asarray(target_labels)[target_idx]
    source = _transform_expression(source[source_idx], transform, library_size)
    target = _transform_expression(target[target_idx], transform, library_size)

    real = np.concatenate((source, target), axis=0)
    pca = PCA(n_components=2, random_state=seed).fit(real)
    source_xy, target_xy = pca.transform(source), pca.transform(target)

    panels = [(source_title, source_xy, source_labels), (target_title, target_xy, target_labels)]
    for offset, (name, values) in enumerate(prediction_arrays.items(), start=2):
        indices = _sample_indices(len(values), max_cells, seed + offset)
        transformed = _transform_expression(values[indices], transform, library_size)
        panels.append((name, pca.transform(transformed), None))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (panel_title, coordinates, labels) in zip(axes, panels):
        _scatter(ax, coordinates, labels)
        ax.set_title(panel_title)
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    legend_ax = axes[1] if target_labels is not None else axes[0]
    handles, names = legend_ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="center right", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.subplots_adjust(right=0.88)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig, axes


def plot_velocity_fields(
    positions: np.ndarray,
    fields: Mapping[str, np.ndarray],
    path: str | Path | None = None,
    labels=None,
    component_pairs: Sequence[tuple[int, int]] = ((0, 1), (2, 3)),
    max_cells: int = 4000,
    max_arrows: int = 300,
    seed: int = 0,
    title: str | None = None,
):
    """Plot multiple velocity/direction fields directly in PCA space.

    This intentionally uses PCA rather than transforming vectors through UMAP. PCA is linear, so
    vectors can be projected consistently; a nonlinear UMAP transform does not preserve velocity
    vectors as ordinary point displacements. ``fields`` can contain Velvet, a new model and the
    reference direction to create paper-style side-by-side comparisons on identical coordinates.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2 or not np.isfinite(positions).all():
        raise ValueError("positions must be finite with shape [n_cells, n_components>=2].")
    if not fields:
        raise ValueError("fields must contain at least one velocity field.")
    fields = {name: np.asarray(values, dtype=np.float64) for name, values in fields.items()}
    if any(values.shape != positions.shape for values in fields.values()):
        raise ValueError("every velocity field must have the same shape as positions.")
    pairs = [pair for pair in component_pairs if max(pair) < positions.shape[1]]
    if not pairs:
        raise ValueError("No requested component pair is available in positions.")
    labels = None if labels is None else np.asarray(labels)
    if labels is not None and labels.shape != (len(positions),):
        raise ValueError("labels must have one value per cell.")

    cell_idx = _sample_indices(len(positions), max_cells, seed)
    valid_arrow = np.flatnonzero(np.isfinite(np.stack(list(fields.values()))).all(axis=(0, 2)))
    arrow_idx = valid_arrow[_sample_indices(len(valid_arrow), min(max_arrows, len(valid_arrow)), seed + 1)]
    fig, axes = plt.subplots(len(pairs), len(fields), figsize=(5 * len(fields), 4.5 * len(pairs)), squeeze=False)

    for row, (x_pc, y_pc) in enumerate(pairs):
        for col, (name, vectors) in enumerate(fields.items()):
            ax = axes[row, col]
            _scatter(ax, positions[cell_idx][:, [x_pc, y_pc]], None if labels is None else labels[cell_idx])
            planar = vectors[arrow_idx][:, [x_pc, y_pc]]
            norms = np.linalg.norm(planar, axis=1)
            keep = np.isfinite(planar).all(axis=1) & (norms > 1e-12)
            if np.any(keep):
                arrows, points = planar[keep] / norms[keep, None], positions[arrow_idx[keep]]
                ax.quiver(points[:, x_pc], points[:, y_pc], arrows[:, 0], arrows[:, 1], angles="xy", scale_units="xy", scale=None, width=0.0025, alpha=0.8)
            ax.set_xlabel(f"PC{x_pc + 1}")
            ax.set_ylabel(f"PC{y_pc + 1}")
            ax.set_title(name if len(pairs) == 1 else f"{name} | PC{x_pc + 1}/PC{y_pc + 1}")

    handles, names = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="center right", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.subplots_adjust(right=0.88)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig, axes


def plot_metric_comparison(
    metrics: pd.DataFrame,
    metric: str,
    path: str | Path | None = None,
    split: str | None = None,
    task: str | None = None,
    task_type: str | None = None,
    reference: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    connect_paired_seeds: bool = True,
):
    """Compare models for one metric using individual seeds plus mean ± sample SD."""
    frame = _filter_frame(metrics, metric=metric, split=split, task=task, task_type=task_type, reference=reference)
    if "group" in frame.columns:
        frame = frame.loc[frame["group"].isna() | frame["group"].astype(str).isin(("", "nan"))]
    frame = frame.assign(value=pd.to_numeric(frame["value"], errors="coerce")).dropna(subset=["value", "model"])
    if frame.empty:
        raise ValueError(f"No rows found for metric {metric!r} and the requested filters.")

    models = list(dict.fromkeys(frame["model"].astype(str)))
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(models) + 2), 5))
    x = np.arange(len(models), dtype=float)

    if connect_paired_seeds and "seed" in frame.columns:
        pivot = frame.pivot_table(index="seed", columns="model", values="value", aggfunc="first")
        for _, row in pivot.reindex(columns=models).dropna().iterrows():
            ax.plot(x, row.to_numpy(dtype=float), linewidth=0.8, alpha=0.25, zorder=1)

    for index, model in enumerate(models):
        values = frame.loc[frame["model"].astype(str) == model, "value"].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(len(values))
        ax.scatter(index + jitter, values, s=28, alpha=0.7, zorder=2)
        mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
        ax.errorbar(index, mean, yerr=std, fmt="o", capsize=5, linewidth=2, markersize=7, zorder=3)

    higher = frame["higher_is_better"].dropna() if "higher_is_better" in frame.columns else pd.Series(dtype=bool)
    direction = "↑" if len(higher) and _as_bool(higher.iloc[0]) else "↓" if len(higher) else ""
    ax.set_xticks(x, models)
    ax.set_ylabel(ylabel or f"{metric} {direction}".strip())
    ax.set_title(title or metric.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, path)
    return fig, ax


def plot_cbd_transitions(
    metrics: pd.DataFrame,
    path: str | Path | None = None,
    split: str | None = None,
    task: str | None = None,
    metric: str = "cosine_similarity",
    transition_order: Sequence[str] | None = None,
    title: str = "Cross-boundary direction",
):
    """Plot mean ± SD CBD score for every biological transition and model."""
    frame = _filter_frame(metrics, metric=metric, split=split, task=task, reference="cbd")
    if "group" not in frame.columns:
        raise KeyError("CBD plotting requires a 'group' column containing source->target transitions.")
    frame = frame.loc[frame["group"].notna() & (frame["group"].astype(str).str.len() > 0)].copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value", "model", "group"])
    if frame.empty:
        raise ValueError("No per-transition CBD metric rows found.")

    models = list(dict.fromkeys(frame["model"].astype(str)))
    observed = list(dict.fromkeys(frame["group"].astype(str)))
    transitions = [group for group in (transition_order or observed) if group in set(observed)]
    transitions += [group for group in observed if group not in transitions]
    y = np.arange(len(transitions), dtype=float)
    offsets = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else np.array([0.0])

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.65 * len(transitions) + 2)))
    for offset, model in zip(offsets, models):
        means, stds = [], []
        for transition in transitions:
            values = frame.loc[(frame["model"].astype(str) == model) & (frame["group"].astype(str) == transition), "value"].to_numpy(dtype=float)
            means.append(values.mean() if len(values) else np.nan)
            stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        ax.errorbar(means, y + offset, xerr=stds, fmt="o", capsize=3, label=model)

    ax.axvline(0.0, linewidth=1, alpha=0.5)
    ax.set_yticks(y, [transition.replace("->", " → ") for transition in transitions])
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Mean CBD cosine similarity ↑")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, path)
    return fig, ax


def plot_cell_type_composition(
    obs: pd.DataFrame,
    path: str | Path | None = None,
    timepoint_column: str = "timepoint",
    cell_type_column: str = "cell_type",
    normalize: bool = True,
    include_unannotated: bool = True,
    title: str = "Cell-type composition over time",
):
    """Plot observed cell-type composition at every timepoint as stacked bars."""
    missing = {timepoint_column, cell_type_column} - set(obs.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    frame = obs[[timepoint_column, cell_type_column]].copy()
    if include_unannotated:
        frame[cell_type_column] = frame[cell_type_column].fillna("Unannotated").replace("", "Unannotated")
    else:
        frame = frame.loc[frame[cell_type_column].notna() & (frame[cell_type_column].astype(str).str.len() > 0)]
    if frame.empty:
        raise ValueError("No annotated cells available for composition plotting.")

    table = pd.crosstab(frame[timepoint_column].astype(str), frame[cell_type_column].astype(str))
    def time_key(value: str):
        try:
            return 0, float(value.removesuffix("h"))
        except ValueError:
            return 1, value
    table = table.reindex(sorted(table.index, key=time_key))
    if normalize:
        table = table.div(table.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(table)), 5))
    table.plot(kind="bar", stacked=True, ax=ax, width=0.82)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Cell fraction" if normalize else "Cells")
    ax.set_ylim((0, 1)) if normalize else None
    ax.set_title(title)
    ax.legend(title="Cell type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, path)
    return fig, ax


def plot_efficiency_summary(
    experiments: pd.DataFrame,
    path: str | Path | None = None,
    split: str | None = None,
    title: str = "Model efficiency",
):
    """Compare training wall time and trainable parameter count across models."""
    frame = _filter_frame(experiments, split=split)
    required = {"model", "training_seconds", "trainable_parameters"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing experiment columns: {sorted(missing)}")
    frame = frame.loc[frame.get("status", "completed") == "completed"].copy() if "status" in frame.columns else frame.copy()
    frame["training_seconds"] = pd.to_numeric(frame["training_seconds"], errors="coerce")
    frame["trainable_parameters"] = pd.to_numeric(frame["trainable_parameters"], errors="coerce")
    if frame.empty:
        raise ValueError("No completed experiments available.")

    models = list(dict.fromkeys(frame["model"].astype(str)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, column, label in ((axes[0], "training_seconds", "Training time [s]"), (axes[1], "trainable_parameters", "Trainable parameters")):
        for index, model in enumerate(models):
            values = frame.loc[frame["model"].astype(str) == model, column].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(len(values))
            ax.scatter(index + jitter, values, s=28, alpha=0.7)
            ax.errorbar(index, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0.0, fmt="o", capsize=5, linewidth=2)
        ax.set_xticks(np.arange(len(models)), models, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig, axes
