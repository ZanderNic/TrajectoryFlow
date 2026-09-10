# std-lib imports
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# 3 party imports
import numpy as np
import pandas as pd
import torch

# package imports
from trajectoryflow.experiment.models import EvaluationAdapter
from trajectoryflow.experiment.runtime import ParameterStats, PhaseStats


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float
    std: float | None = None
    higher_is_better: bool | None = None
    value_range: str | None = None
    group: str | None = None


@dataclass
class TaskResult:
    phase: str
    task_name: str
    task_kind: str
    source: str
    target: str
    delta_hours: float
    n_source_cells: int
    n_target_cells: int
    n_samples: int
    metrics: list[MetricValue] = field(default_factory=list)
    data_prep: PhaseStats | None = None
    prediction_runs: list[PhaseStats] = field(default_factory=list)
    evaluation: PhaseStats | None = None
    source_indices_fingerprint: str | None = None
    target_indices_fingerprint: str | None = None

    @property
    def mean_prediction_seconds(self) -> float | None:
        if not self.prediction_runs:
            return None

        return sum(run.wall_seconds for run in self.prediction_runs) / len(
            self.prediction_runs
        )

    @property
    def prediction_cells_per_second(self) -> float | None:
        seconds = self.mean_prediction_seconds

        if seconds is None or seconds <= 0:
            return None

        return self.n_source_cells * self.n_samples / seconds


@dataclass
class VelocityTaskResult:
    phase: str
    task_name: str
    reference: str
    timepoints: tuple[str, ...]
    n_cells: int
    n_reference_components: int
    status: str = "completed"
    metrics: list[MetricValue] = field(default_factory=list)
    prediction: PhaseStats | None = None
    evaluation: PhaseStats | None = None
    cell_indices_fingerprint: str | None = None
    plot_paths: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ExperimentResult:
    experiment_id: str
    model: str
    split: str
    split_fingerprint: str
    seed: int
    status: str
    fit_timepoints: tuple[str, ...]
    training_cell_counts: dict[str, int]
    parameters: ParameterStats
    system: dict[str, Any]
    training_data_prep: PhaseStats | None = None
    training: PhaseStats | None = None
    total: PhaseStats | None = None
    training_summary: dict[str, Any] = field(default_factory=dict)
    tasks: list[TaskResult] = field(default_factory=list)
    velocity_tasks: list[VelocityTaskResult] = field(default_factory=list)
    error: str | None = None

    def metrics_frame(self) -> pd.DataFrame:
        rows = []

        for task in self.tasks:
            for metric in task.metrics:
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "model": self.model,
                        "split": self.split,
                        "split_fingerprint": self.split_fingerprint,
                        "seed": self.seed,
                        "phase": task.phase,
                        "task_type": "forecast",
                        "task": task.task_name,
                        "task_kind": task.task_kind,
                        "reference": None,
                        "group": metric.group,
                        "timepoints": None,
                        "source": task.source,
                        "target": task.target,
                        "delta_hours": task.delta_hours,
                        "n_source_cells": task.n_source_cells,
                        "n_target_cells": task.n_target_cells,
                        "n_velocity_cells": None,
                        "n_samples": task.n_samples,
                        "metric": metric.name,
                        "value": metric.value,
                        "std": metric.std,
                        "higher_is_better": metric.higher_is_better,
                        "value_range": metric.value_range,
                    }
                )

        for task in self.velocity_tasks:
            for metric in task.metrics:
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "model": self.model,
                        "split": self.split,
                        "split_fingerprint": self.split_fingerprint,
                        "seed": self.seed,
                        "phase": task.phase,
                        "task_type": "velocity",
                        "task": task.task_name,
                        "task_kind": "local_direction",
                        "reference": task.reference,
                        "group": metric.group,
                        "timepoints": ",".join(task.timepoints),
                        "source": None,
                        "target": None,
                        "delta_hours": None,
                        "n_source_cells": None,
                        "n_target_cells": None,
                        "n_velocity_cells": task.n_cells,
                        "n_samples": None,
                        "metric": metric.name,
                        "value": metric.value,
                        "std": metric.std,
                        "higher_is_better": metric.higher_is_better,
                        "value_range": metric.value_range,
                    }
                )

        return pd.DataFrame(rows)

    def runtime_frame(self) -> pd.DataFrame:
        rows = []

        if self.training_data_prep is not None:
            rows.append(
                {
                    "experiment_id": self.experiment_id,
                    "model": self.model,
                    "split": self.split,
                    "seed": self.seed,
                    "phase": "training_data_prep",
                    "task": None,
                    **self.training_data_prep.to_dict(),
                }
            )

        if self.training is not None:
            rows.append(
                {
                    "experiment_id": self.experiment_id,
                    "model": self.model,
                    "split": self.split,
                    "seed": self.seed,
                    "phase": "training",
                    "task": None,
                    **self.training.to_dict(),
                }
            )

        for task in self.tasks:
            if task.data_prep is not None:
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "model": self.model,
                        "split": self.split,
                        "seed": self.seed,
                        "phase": f"{task.phase}_data_prep",
                        "task": task.task_name,
                        **task.data_prep.to_dict(),
                    }
                )

            for run_id, timing in enumerate(task.prediction_runs):
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "model": self.model,
                        "split": self.split,
                        "seed": self.seed,
                        "phase": f"{task.phase}_prediction",
                        "task": task.task_name,
                        "timing_run": run_id,
                        **timing.to_dict(),
                    }
                )

            if task.evaluation is not None:
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "model": self.model,
                        "split": self.split,
                        "seed": self.seed,
                        "phase": f"{task.phase}_evaluation",
                        "task": task.task_name,
                        **task.evaluation.to_dict(),
                    }
                )

        for task in self.velocity_tasks:
            if task.prediction is not None:
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "model": self.model,
                        "split": self.split,
                        "seed": self.seed,
                        "phase": f"{task.phase}_velocity_prediction",
                        "task": task.task_name,
                        **task.prediction.to_dict(),
                    }
                )

            if task.evaluation is not None:
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "model": self.model,
                        "split": self.split,
                        "seed": self.seed,
                        "phase": f"{task.phase}_velocity_evaluation",
                        "task": task.task_name,
                        **task.evaluation.to_dict(),
                    }
                )

        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return asdict(self)


class CallableEvaluationAdapter(EvaluationAdapter):
    """Wrap the existing TrajectoryFlow evaluator without coupling experiment code to it."""

    def __init__(self, evaluate: Callable):
        self._evaluate = evaluate

    def evaluate(self, prediction, target: torch.Tensor) -> pd.DataFrame:
        result = self._evaluate(prediction, target)

        if isinstance(result, pd.DataFrame):
            return result

        if hasattr(result, "frame"):
            frame = result.frame
            return frame() if callable(frame) else frame

        if hasattr(result, "to_frame"):
            return result.to_frame()

        raise TypeError(
            "Evaluator must return a DataFrame or an object exposing frame/to_frame."
        )


def metric_values_from_frame(frame: pd.DataFrame) -> list[MetricValue]:
    if frame.empty:
        return []

    value_column = "value" if "value" in frame.columns else "mean"
    metrics = []

    for _, row in frame.iterrows():
        metrics.append(
            MetricValue(
                name=str(row["metric"]),
                value=float(row[value_column]),
                std=(
                    float(row["std"])
                    if "std" in frame.columns and pd.notna(row["std"])
                    else None
                ),
                higher_is_better=(
                    bool(row["higher_is_better"])
                    if "higher_is_better" in frame.columns
                    and pd.notna(row["higher_is_better"])
                    else None
                ),
                value_range=(
                    str(row["value_range"])
                    if "value_range" in frame.columns
                    and pd.notna(row["value_range"])
                    else None
                ),
                group=(
                    str(row["group"])
                    if "group" in frame.columns
                    and pd.notna(row["group"])
                    else None
                ),
            )
        )

    return metrics


def library_log1p(
    expression: torch.Tensor,
    library_size: float = 10_000.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if expression.ndim not in (2, 3):
        raise ValueError(
            "expression must have shape [cells, genes] or [samples, cells, genes]."
        )

    if not torch.isfinite(expression).all():
        raise ValueError("expression contains non-finite values.")

    if (expression < 0).any():
        raise ValueError("library_log1p requires non-negative expression values.")

    library = expression.sum(dim=-1, keepdim=True).clamp_min(eps)
    normalized = expression / library * library_size

    return torch.log1p(normalized)


def transform_evaluation_space(
    expression: torch.Tensor,
    transform: str,
    library_size: float,
) -> torch.Tensor:
    if transform == "none":
        return expression

    if transform == "library_log1p":
        return library_log1p(expression, library_size=library_size)

    raise ValueError(f"Unknown evaluation transform {transform!r}.")

@dataclass(frozen=True)
class VelocityReference:
    name: str
    cell_ids: np.ndarray
    vectors: np.ndarray
    pca_components: np.ndarray
    pca_mean: np.ndarray
    positions: np.ndarray | None = None
    groups: np.ndarray | None = None
    genes: np.ndarray | None = None
    expression_transform: str = "none"
    library_size: float = 10_000.0
    projection_mode: str = "linear"
    velocity_epsilon: float = 1e-3
    alignment_mode: str = "cosine"

    def __post_init__(self) -> None:
        cell_ids = np.asarray(self.cell_ids).astype(str)
        vectors = np.asarray(self.vectors, dtype=np.float64)
        components = np.asarray(self.pca_components, dtype=np.float64)
        mean = np.asarray(self.pca_mean, dtype=np.float64)

        if cell_ids.ndim != 1:
            raise ValueError("VelocityReference cell_ids must be one-dimensional.")
        # Duplicate cell IDs are allowed because one boundary cell can
        # contribute to multiple CBD source->target transitions.
        if vectors.ndim != 2 or vectors.shape[0] != len(cell_ids):
            raise ValueError(
                "VelocityReference vectors must have shape [n_cells, n_components]."
            )
        if components.ndim != 2:
            raise ValueError(
                "VelocityReference pca_components must have shape "
                "[n_components, n_genes]."
            )
        if vectors.shape[1] != components.shape[0]:
            raise ValueError(
                "Reference vector dimension must match the PCA component count."
            )
        if mean.shape != (components.shape[1],):
            raise ValueError("pca_mean must have shape [n_genes].")
        if self.expression_transform not in ("none", "library_log1p"):
            raise ValueError(
                "expression_transform must be 'none' or 'library_log1p'."
            )
        if self.projection_mode not in ("linear", "finite_difference"):
            raise ValueError(
                "projection_mode must be 'linear' or 'finite_difference'."
            )
        if self.alignment_mode not in ("cosine", "mean_unit_direction"):
            raise ValueError(
                "alignment_mode must be 'cosine' or 'mean_unit_direction'."
            )
        if self.library_size <= 0:
            raise ValueError("library_size must be > 0.")
        if self.velocity_epsilon <= 0:
            raise ValueError("velocity_epsilon must be > 0.")

        positions = self.positions
        if positions is not None:
            positions = np.asarray(positions, dtype=np.float64)
            if positions.shape != vectors.shape:
                raise ValueError(
                    "positions must have the same shape as reference vectors."
                )

        groups = self.groups
        if groups is not None:
            groups = np.asarray(groups).astype(str)
            if groups.shape != cell_ids.shape:
                raise ValueError("groups must have shape [n_cells].")

        genes = self.genes
        if genes is not None:
            genes = np.asarray(genes).astype(str)
            if genes.shape != (components.shape[1],):
                raise ValueError("genes must have shape [n_genes].")

        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "pca_components", components)
        object.__setattr__(self, "pca_mean", mean)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "genes", genes)

    @property
    def n_components(self) -> int:
        return int(self.pca_components.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.pca_components.shape[1])

    def indices_for(self, cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lookup: dict[str, list[int]] = {}
        for reference_index, cell_id in enumerate(self.cell_ids):
            lookup.setdefault(cell_id, []).append(reference_index)

        query_indices = []
        reference_indices = []

        for query_index, cell_id in enumerate(np.asarray(cell_ids).astype(str)):
            for reference_index in lookup.get(cell_id, ()):
                query_indices.append(query_index)
                reference_indices.append(reference_index)

        return (
            np.asarray(query_indices, dtype=np.int64),
            np.asarray(reference_indices, dtype=np.int64),
        )


def _npz_scalar(values, key: str, default):
    if key not in values:
        return default

    value = values[key]

    if np.asarray(value).ndim == 0:
        return np.asarray(value).item()

    if np.asarray(value).size == 1:
        return np.asarray(value).reshape(-1)[0].item()

    raise ValueError(f"Expected scalar value for {key!r}.")


def load_velocity_reference(path: str | Path) -> VelocityReference:
    path = Path(path)

    with np.load(path, allow_pickle=False) as values:
        required = {"cell_ids", "vectors", "pca_components", "pca_mean"}
        missing = required - set(values.files)

        if missing:
            raise KeyError(
                f"Velocity reference {path} is missing arrays: {sorted(missing)}."
            )

        return VelocityReference(
            name=str(_npz_scalar(values, "name", path.stem)),
            cell_ids=values["cell_ids"],
            vectors=values["vectors"],
            pca_components=values["pca_components"],
            pca_mean=values["pca_mean"],
            positions=values["positions"] if "positions" in values else None,
            groups=values["groups"] if "groups" in values else None,
            genes=values["genes"] if "genes" in values else None,
            expression_transform=str(
                _npz_scalar(values, "expression_transform", "none")
            ),
            library_size=float(
                _npz_scalar(values, "library_size", 10_000.0)
            ),
            projection_mode=str(
                _npz_scalar(values, "projection_mode", "linear")
            ),
            velocity_epsilon=float(
                _npz_scalar(values, "velocity_epsilon", 1e-3)
            ),
            alignment_mode=str(
                _npz_scalar(values, "alignment_mode", "cosine")
            ),
        )


def save_velocity_reference(
    path: str | Path,
    reference: VelocityReference,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "name": np.asarray(reference.name),
        "cell_ids": reference.cell_ids,
        "vectors": reference.vectors.astype(np.float32),
        "pca_components": reference.pca_components.astype(np.float32),
        "pca_mean": reference.pca_mean.astype(np.float32),
        "expression_transform": np.asarray(reference.expression_transform),
        "library_size": np.asarray(reference.library_size, dtype=np.float64),
        "projection_mode": np.asarray(reference.projection_mode),
        "velocity_epsilon": np.asarray(
            reference.velocity_epsilon,
            dtype=np.float64,
        ),
        "alignment_mode": np.asarray(reference.alignment_mode),
    }

    if reference.positions is not None:
        arrays["positions"] = reference.positions.astype(np.float32)
    if reference.groups is not None:
        arrays["groups"] = reference.groups
    if reference.genes is not None:
        arrays["genes"] = reference.genes

    np.savez_compressed(path, **arrays)


def _transform_numpy_expression(
    expression: np.ndarray,
    transform: str,
    library_size: float,
    eps: float = 1e-8,
) -> np.ndarray:
    expression = np.asarray(expression, dtype=np.float64)

    if transform == "none":
        return expression

    if transform != "library_log1p":
        raise ValueError(f"Unknown expression transform {transform!r}.")

    if (expression < 0).any():
        raise ValueError("library_log1p requires non-negative expression.")

    totals = expression.sum(axis=-1, keepdims=True)
    scaled = expression / np.maximum(totals, eps) * library_size
    return np.log1p(scaled)


def project_velocity_to_reference(
    expression: np.ndarray,
    velocity: np.ndarray,
    reference: VelocityReference,
) -> tuple[np.ndarray, np.ndarray]:
    expression = np.asarray(expression, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    if expression.shape != velocity.shape:
        raise ValueError("expression and velocity must have the same shape.")
    if expression.ndim != 2:
        raise ValueError("expression/velocity must have shape [n_cells, n_genes].")
    if expression.shape[1] != reference.n_genes:
        raise ValueError(
            f"Reference expects {reference.n_genes} genes, "
            f"got {expression.shape[1]}."
        )

    transformed = _transform_numpy_expression(
        expression,
        transform=reference.expression_transform,
        library_size=reference.library_size,
    )
    positions = (transformed - reference.pca_mean) @ reference.pca_components.T

    if reference.projection_mode == "linear":
        projected_velocity = velocity @ reference.pca_components.T
    else:
        epsilon = reference.velocity_epsilon
        next_expression = np.maximum(expression + epsilon * velocity, 0.0)
        transformed_next = _transform_numpy_expression(
            next_expression,
            transform=reference.expression_transform,
            library_size=reference.library_size,
        )
        projected_velocity = (
            (transformed_next - transformed) @ reference.pca_components.T
        ) / epsilon

    return positions, projected_velocity


def velocity_alignment_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    cell_ids: np.ndarray,
    groups: np.ndarray | None = None,
    alignment_mode: str = "cosine",
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    cell_ids = np.asarray(cell_ids).astype(str)

    if predicted.shape != reference.shape:
        raise ValueError("predicted and reference vectors must have the same shape.")
    if predicted.ndim != 2:
        raise ValueError("velocity vectors must have shape [n_cells, n_components].")
    if len(cell_ids) != len(predicted):
        raise ValueError("cell_ids length must match velocity rows.")

    if alignment_mode not in ("cosine", "mean_unit_direction"):
        raise ValueError(f"Unknown alignment mode {alignment_mode!r}.")

    predicted_norm = np.linalg.norm(predicted, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    finite = np.isfinite(predicted).all(axis=1) & np.isfinite(reference).all(axis=1)

    score = np.full(len(predicted), np.nan, dtype=np.float64)

    if alignment_mode == "cosine":
        valid = finite & (predicted_norm > eps) & (reference_norm > eps)
        score[valid] = np.sum(
            predicted[valid] * reference[valid],
            axis=1,
        ) / (predicted_norm[valid] * reference_norm[valid])
    else:
        # For CBD, `reference` stores the mean of unit displacement vectors
        # from a boundary cell to its target-cluster neighbours. Therefore
        # v_hat dot reference is exactly the mean neighbour-wise cosine score.
        valid = finite & (predicted_norm > eps)
        score[valid] = np.sum(
            (predicted[valid] / predicted_norm[valid, None]) * reference[valid],
            axis=1,
        )

    score = np.clip(score, -1.0, 1.0, out=score, where=np.isfinite(score))

    if groups is None:
        groups = np.full(len(predicted), "", dtype=str)
    else:
        groups = np.asarray(groups).astype(str)
        if len(groups) != len(predicted):
            raise ValueError("groups length must match velocity rows.")

    per_cell = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "group": groups,
            "cosine_similarity": score,
            "predicted_norm": predicted_norm,
            "reference_norm": reference_norm,
            "valid": valid,
        }
    )

    rows = []

    def add_summary(frame: pd.DataFrame, group: str | None) -> None:
        values = frame.loc[frame["valid"], "cosine_similarity"].to_numpy()
        valid_fraction = float(frame["valid"].mean()) if len(frame) else 0.0

        if len(values):
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            median = float(np.median(values))
            positive = float((values > 0).mean())
        else:
            mean = float("nan")
            std = float("nan")
            median = float("nan")
            positive = float("nan")

        rows.extend(
            [
                {
                    "metric": "cosine_similarity",
                    "value": mean,
                    "std": std,
                    "higher_is_better": True,
                    "value_range": "[-1, 1]",
                    "group": group,
                },
                {
                    "metric": "cosine_median",
                    "value": median,
                    "std": None,
                    "higher_is_better": True,
                    "value_range": "[-1, 1]",
                    "group": group,
                },
                {
                    "metric": "positive_fraction",
                    "value": positive,
                    "std": None,
                    "higher_is_better": True,
                    "value_range": "[0, 1]",
                    "group": group,
                },
                {
                    "metric": "valid_fraction",
                    "value": valid_fraction,
                    "std": None,
                    "higher_is_better": True,
                    "value_range": "[0, 1]",
                    "group": group,
                },
            ]
        )

    add_summary(per_cell, group=None)

    non_empty_groups = sorted(group for group in np.unique(groups) if group)
    for group in non_empty_groups:
        add_summary(per_cell.loc[per_cell["group"] == group], group=group)

    return pd.DataFrame(rows), per_cell

