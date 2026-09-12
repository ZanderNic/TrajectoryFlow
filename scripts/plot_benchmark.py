#!/usr/bin/env python3

# std-lib imports
import argparse
import json
import re
from pathlib import Path

# 3 party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# package imports
from trajectoryflow.data.store import ScifateStore
from trajectoryflow.experiment.evaluation import load_velocity_reference
from trajectoryflow.plotting.paper import (
    plot_cbd_transitions,
    plot_cell_type_composition,
    plot_efficiency_summary,
    plot_forecast_comparison,
    plot_metric_comparison,
    plot_velocity_fields,
)

PRIMARY_FORECAST_METRICS = ("sliced_wasserstein", "mmd", "centroid_distance", "mean_correlation")
SCIFATE2_CBD_ORDER = ("Early_Neural->Neural", "NMP->Early_Neural", "NMP->Mesoderm", "Neural->pMN", "pMN->MN", "pMN->p3", "p3->V3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready figures from a TrajectoryFlow benchmark directory.")
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Seed used for qualitative forecast/velocity figures. Defaults to the first benchmark seed.")
    parser.add_argument("--phase", choices=("validation", "test"), default="test")
    parser.add_argument("--prediction-sample", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=3000)
    return parser.parse_args()


def _safe(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_manifest(root: Path) -> dict:
    path = root / "benchmark_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_result(run_dir: Path) -> dict:
    path = run_dir / "result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _task_file_name(phase: str, task: str) -> str:
    return f"{phase}__{task}".replace("/", "_").replace(" ", "_")


def _close(result) -> None:
    fig = result[0] if isinstance(result, tuple) else result
    plt.close(fig)


def _load_prediction_sample(run_dir: Path, safe: str, sample: int) -> np.ndarray | None:
    legacy = run_dir / "predictions" / f"{safe}.npz"
    if legacy.exists():
        with np.load(legacy) as values:
            states = values["states"]
        if states.ndim == 3:
            if not 0 <= sample < states.shape[0]:
                raise IndexError(f"Prediction sample {sample} is unavailable; n_samples={states.shape[0]}.")
            return states[sample]
        return states

    path = run_dir / "predictions" / safe / f"sample_{sample:03d}.npz"
    if not path.exists():
        return None
    with np.load(path) as values:
        return values["state"]


def make_metric_figures(metrics: pd.DataFrame, output: Path) -> None:
    if metrics.empty:
        return
    forecast = metrics.loc[metrics.get("task_type", "") == "forecast"]
    for metric in PRIMARY_FORECAST_METRICS:
        frame = forecast.loc[forecast["metric"] == metric]
        for (split, task), _ in frame.groupby(["split", "task"], dropna=False):
            path = output / "metrics" / f"{_safe(split)}__{_safe(task)}__{metric}.png"
            _close(plot_metric_comparison(metrics, metric, path=path, split=split, task=task, task_type="forecast"))

    direction = metrics.loc[(metrics.get("task_type", "") == "velocity") & (metrics["metric"] == "cosine_similarity")]
    overall = direction.loc[direction["group"].isna() | direction["group"].astype(str).isin(("", "nan"))] if "group" in direction else direction
    for (split, task, reference), _ in overall.groupby(["split", "task", "reference"], dropna=False):
        path = output / "direction" / f"{_safe(split)}__{_safe(task)}__{_safe(reference)}.png"
        _close(plot_metric_comparison(metrics, "cosine_similarity", path=path, split=split, task=task, task_type="velocity", reference=reference, ylabel="Mean cosine similarity ↑"))

    cbd = direction.loc[direction.get("reference", "") == "cbd"]
    order = list(SCIFATE2_CBD_ORDER)
    for (split, task), frame in cbd.groupby(["split", "task"], dropna=False):
        groups = frame["group"].dropna().astype(str) if "group" in frame else pd.Series(dtype=str)
        if not len(groups):
            continue
        path = output / "direction" / f"{_safe(split)}__{_safe(task)}__cbd_transitions.png"
        _close(plot_cbd_transitions(metrics, path=path, split=split, task=task, transition_order=order))


def make_efficiency_figures(experiments: pd.DataFrame, output: Path) -> None:
    if experiments.empty:
        return
    for split in experiments["split"].dropna().unique():
        path = output / "efficiency" / f"{_safe(split)}.png"
        _close(plot_efficiency_summary(experiments, path=path, split=split, title=f"Model efficiency | {split}"))


def make_composition_figure(store: ScifateStore, output: Path) -> None:
    frames = []
    for timepoint in store.timepoints:
        obs = store.load(timepoint).obs.copy()
        if "timepoint" not in obs.columns:
            obs["timepoint"] = timepoint
        frames.append(obs)
    obs = pd.concat(frames, ignore_index=True)
    if "cell_type" in obs.columns:
        _close(plot_cell_type_composition(obs, path=output / "data" / "cell_type_composition.png"))


def _task_definition(result: dict, phase: str, task: str) -> dict | None:
    return next((item for item in result.get("tasks", ()) if item.get("phase") == phase and item.get("task_name") == task), None)


def make_forecast_figures(root: Path, store: ScifateStore, models: list[str], splits: list[str], seed: int, phase: str, sample: int, max_cells: int, output: Path) -> None:
    for split in splits:
        available = [(model, root / model / split / f"seed_{seed}") for model in models]
        available = [(model, run_dir) for model, run_dir in available if (run_dir / "result.json").exists()]
        if not available:
            continue
        _, template_dir = available[0]
        template_result = _load_result(template_dir)

        for task_info in template_result.get("tasks", ()):
            if task_info.get("phase") != phase:
                continue
            task, source_time, target_time = task_info["task_name"], task_info["source"], task_info["target"]
            safe = _task_file_name(phase, task)
            indices_path = template_dir / "task_indices" / f"{safe}.npz"
            if not indices_path.exists():
                continue
            with np.load(indices_path) as values:
                source_idx, target_idx = values["source_indices"], values["target_indices"]

            source_snapshot, target_snapshot = store.load_pair(source_time, target_time)
            source, target = source_snapshot.expression[source_idx], target_snapshot.expression[target_idx]
            source_labels = source_snapshot.obs.iloc[source_idx]["cell_type"].fillna("Unannotated").to_numpy() if "cell_type" in source_snapshot.obs else None
            target_labels = target_snapshot.obs.iloc[target_idx]["cell_type"].fillna("Unannotated").to_numpy() if "cell_type" in target_snapshot.obs else None
            predictions = {}

            for model, run_dir in available:
                task_definition = _task_definition(_load_result(run_dir), phase, task)
                model_indices = run_dir / "task_indices" / f"{safe}.npz"
                if task_definition is None or not model_indices.exists():
                    continue
                with np.load(model_indices) as values:
                    if not np.array_equal(values["source_indices"], source_idx) or not np.array_equal(values["target_indices"], target_idx):
                        raise RuntimeError(f"Task indices differ across models for {split}/{task}/seed_{seed}.")
                states = _load_prediction_sample(run_dir, safe, sample)
                if states is not None:
                    predictions[model] = states

            if predictions:
                path = output / "forecast" / f"{_safe(split)}__{_safe(task)}__seed_{seed}.png"
                _close(plot_forecast_comparison(source, target, predictions, path=path, source_labels=source_labels, target_labels=target_labels, max_cells=max_cells, seed=seed, title=f"{task} | seed {seed}"))


def _reference_rows(reference, cells: pd.DataFrame) -> np.ndarray:
    if reference.groups is not None and "group" in cells.columns:
        lookup = {(str(cell_id), str(group)): i for i, (cell_id, group) in enumerate(zip(reference.cell_ids, reference.groups))}
        keys = list(zip(cells["cell_id"].astype(str), cells["group"].astype(str)))
        indices = [lookup.get(key, -1) for key in keys]
    else:
        lookup = {str(cell_id): i for i, cell_id in enumerate(reference.cell_ids)}
        indices = [lookup.get(cell_id, -1) for cell_id in cells["cell_id"].astype(str)]
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices < 0):
        raise ValueError("Could not match every velocity cell to the stored reference.")
    return indices


def make_velocity_figures(root: Path, manifest: dict, models: list[str], splits: list[str], seed: int, phase: str, max_cells: int, output: Path) -> None:
    references = manifest.get("velocity_references", {})
    for split in splits:
        runs = [(model, root / model / split / f"seed_{seed}") for model in models]
        runs = [(model, run_dir) for model, run_dir in runs if (run_dir / "result.json").exists()]
        if not runs:
            continue
        template_result = _load_result(runs[0][1])

        for task in template_result.get("velocity_tasks", ()):
            if task.get("phase") != phase or task.get("status") != "completed":
                continue
            task_name, reference_name = task["task_name"], task["reference"]
            reference_info = references.get(reference_name)
            if not reference_info or not Path(reference_info["path"]).exists():
                continue
            reference = load_velocity_reference(reference_info["path"])
            safe = _task_file_name(phase, task_name)
            fields, cells = {}, None

            for model, run_dir in runs:
                folder = run_dir / "velocity" / safe
                prediction_path, cells_path = folder / "prediction.npz", folder / "cells.csv"
                if not prediction_path.exists() or not cells_path.exists():
                    continue
                model_cells = pd.read_csv(cells_path)
                with np.load(prediction_path) as values:
                    projected, prediction_ids = values["projected_velocity"], values["cell_ids"].astype(str)
                if not np.array_equal(prediction_ids, model_cells["cell_id"].astype(str).to_numpy()):
                    raise RuntimeError(f"Velocity prediction/cell order mismatch for {model}/{split}/{task_name}.")
                if cells is None:
                    cells = model_cells
                elif not np.array_equal(cells["cell_id"].astype(str).to_numpy(), model_cells["cell_id"].astype(str).to_numpy()):
                    raise RuntimeError(f"Velocity cell sampling differs across models for {split}/{task_name}/seed_{seed}.")
                fields[model] = projected

            if cells is None or not fields:
                continue
            reference_idx = _reference_rows(reference, cells)
            positions = reference.positions[reference_idx] if reference.positions is not None else None
            if positions is None:
                continue
            fields["Reference"] = reference.vectors[reference_idx]
            labels = cells["label"].fillna("Unannotated").to_numpy() if "label" in cells.columns else None
            path = output / "velocity" / f"{_safe(split)}__{_safe(task_name)}__seed_{seed}.png"
            _close(plot_velocity_fields(positions, fields, path=path, labels=labels, max_cells=max_cells, seed=seed, title=f"{task_name} | seed {seed}"))


def main() -> None:
    args = parse_args()
    root = args.benchmark_dir.resolve()
    output = (args.output_dir or root / "paper_figures").resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(root)
    models, splits = list(manifest.get("models", ())), list(manifest.get("splits", ()))
    seeds = list(manifest.get("seeds", ()))
    seed = args.seed if args.seed is not None else (int(seeds[0]) if seeds else 0)
    store = ScifateStore(manifest["dataset"]["root"])
    metrics, experiments = _read_csv(root / "metrics.csv"), _read_csv(root / "experiments.csv")

    make_metric_figures(metrics, output)
    make_efficiency_figures(experiments, output)
    make_composition_figure(store, output)
    make_forecast_figures(root, store, models, splits, seed, args.phase, args.prediction_sample, args.max_cells, output)
    make_velocity_figures(root, manifest, models, splits, seed, args.phase, args.max_cells, output)
    print(f"Paper figures saved to {output}")


if __name__ == "__main__":
    main()
