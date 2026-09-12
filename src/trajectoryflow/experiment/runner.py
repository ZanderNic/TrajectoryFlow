# std-lib imports
import copy
import gc
import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

# 3 party imports
import numpy as np
import pandas as pd
import torch

# package imports
from trajectoryflow.experiment.config import BenchmarkConfig, ModelConfig
from trajectoryflow.experiment.data import ExperimentData, PopulationSampler
from trajectoryflow.experiment.evaluation import (
    ExperimentResult,
    TaskResult,
    VelocityTaskResult,
    load_velocity_reference,
    metric_values_from_frame,
    project_velocity_to_reference,
    transform_evaluation_space,
    velocity_alignment_metrics,
)
from trajectoryflow.experiment.models import ExperimentRegistry
from trajectoryflow.experiment.plotting import plot_velocity_alignment_histogram, plot_velocity_comparison
from trajectoryflow.experiment.runtime import PhaseProfiler, parameter_stats, seed_everything, stable_seed, system_info
from trajectoryflow.experiment.split import ResolvedSplit, VelocityTask, resolve_split


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class RunArtifacts:
    def __init__(self, root: Path, model: str, split: str, seed: int):
        self.root = Path(root) / model / split / f"seed_{seed}"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def save_result(self, result: ExperimentResult) -> None:
        self._write_json("result.json", result.to_dict())
        metrics, runtimes = result.metrics_frame(), result.runtime_frame()
        if not metrics.empty:
            metrics.to_csv(self.root / "metrics.csv", index=False)
        if not runtimes.empty:
            runtimes.to_csv(self.root / "runtime.csv", index=False)

    def save_config(self, config) -> None:
        self._write_json("config.json", config)

    def save_split(self, split: ResolvedSplit) -> None:
        arrays = {}
        for timepoint, partition in split.partitions.items():
            safe = timepoint.replace(".", "_")
            arrays.update({f"{safe}__train": partition.train_indices, f"{safe}__validation": partition.validation_indices, f"{safe}__test": partition.test_indices})
        np.savez_compressed(self.root / "split_indices.npz", **arrays)
        self._write_json("split.json", {"name": split.name, "seed": split.seed, "fingerprint": split.fingerprint, "fit_timepoints": split.fit_timepoints, "cell_counts": split.cell_counts()})

    def save_checkpoint(self, state) -> Path:
        path = self.root / "checkpoint.pt"
        torch.save(state, path)
        return path

    def save_task_indices(self, phase: str, task_name: str, source_indices: np.ndarray, target_indices: np.ndarray) -> None:
        folder = self.root / "task_indices"
        folder.mkdir(exist_ok=True)
        name = f"{phase}__{task_name}".replace("/", "_")
        np.savez_compressed(folder / f"{name}.npz", source_indices=np.asarray(source_indices, dtype=np.int64), target_indices=np.asarray(target_indices, dtype=np.int64))

    def save_prediction_sample(self, phase: str, task_name: str, sample_index: int, prediction, state: torch.Tensor) -> Path:
        name = f"{phase}__{task_name}".replace("/", "_").replace(" ", "_")
        folder = self.root / "predictions" / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"sample_{sample_index:03d}.npz"
        np.savez_compressed(path, state=state.detach().cpu().numpy(), source_time=np.asarray(prediction.source_time), target_time=np.asarray(prediction.target_time))
        return path

    def velocity_folder(self, phase: str, task_name: str) -> Path:
        safe = f"{phase}__{task_name}".replace("/", "_").replace(" ", "_")
        folder = self.root / "velocity" / safe
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_velocity_evaluation(self, phase: str, task_name: str, metrics, per_cell) -> Path:
        folder = self.velocity_folder(phase, task_name)
        metrics.to_csv(folder / "metrics.csv", index=False)
        per_cell.to_csv(folder / "cells.csv", index=False)
        return folder

    def save_velocity_prediction(self, phase: str, task_name: str, projected_velocity: np.ndarray, cell_ids: np.ndarray) -> Path:
        path = self.velocity_folder(phase, task_name) / "prediction.npz"
        np.savez_compressed(path, cell_ids=np.asarray(cell_ids).astype(str), projected_velocity=np.asarray(projected_velocity, dtype=np.float32))
        return path

    def _write_json(self, name: str, value) -> None:
        with (self.root / name).open("w", encoding="utf-8") as file:
            json.dump(_jsonable(value), file, indent=2, ensure_ascii=False)


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def indices_fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:16]


def strings_fingerprint(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _reference_path(reference_dir: Path, reference: str) -> Path:
    path = Path(reference)
    return (path if path.is_absolute() else reference_dir / path) if path.suffix else reference_dir / f"{reference}.npz"


def _dataset_gene_names(data: ExperimentData) -> np.ndarray | None:
    genes = data.genes
    if hasattr(genes, "columns"):
        for column in ("gene", "gene_name", "gene_id", "feature_name"):
            if column in genes.columns and len(genes[column]) == data.n_genes:
                return genes[column].astype(str).to_numpy()
    if hasattr(genes, "index") and len(genes.index) == data.n_genes:
        values = np.asarray(genes.index).astype(str)
        if len(set(values.tolist())) == len(values):
            return values
    return None


def _validate_reference_genes(reference, data: ExperimentData) -> None:
    if reference.n_genes != data.n_genes:
        raise ValueError(f"Velocity reference expects {reference.n_genes} genes, but the experiment data has {data.n_genes}.")
    dataset_genes = _dataset_gene_names(data)
    if reference.genes is not None and dataset_genes is not None and not np.array_equal(reference.genes.astype(str), dataset_genes.astype(str)):
        raise ValueError("Velocity reference gene order does not match the experiment dataset.")


def dense_expression(selection, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(selection.expression.toarray()).float().to(device)


def _clear_store_cache(store) -> None:
    clear = getattr(store, "clear_cache", None)
    if callable(clear):
        clear()


def _replace_prediction_states(prediction, states: torch.Tensor):
    try:
        return replace(prediction, states=states)
    except TypeError:
        cloned = copy.copy(prediction)
        cloned.states = states
        return cloned


def _aggregate_metric_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    first = frames[0].copy()
    value_column = "value" if "value" in first.columns else "mean"
    identity_columns = [column for column in ("metric", "group") if column in first.columns]
    identity = first[identity_columns].fillna("").astype(str).to_numpy() if identity_columns else None
    values = []

    for frame in frames:
        column = "value" if "value" in frame.columns else "mean"
        if len(frame) != len(first) or column != value_column:
            raise ValueError("Evaluator returned inconsistent metric rows across prediction samples.")
        if identity_columns and not np.array_equal(frame[identity_columns].fillna("").astype(str).to_numpy(), identity):
            raise ValueError("Evaluator metric identities changed across prediction samples.")
        values.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64))

    stacked = np.vstack(values)
    first[value_column] = np.nanmean(stacked, axis=0)
    first["std"] = np.nanstd(stacked, axis=0, ddof=1) if len(frames) > 1 else 0.0
    return first


class ExperimentRunner:
    def __init__(self, config: BenchmarkConfig, registry: ExperimentRegistry, store):
        self.config = config
        self.registry = registry
        self.store = store
        self.device = resolve_device(config.runtime.device)
        self.sampler = PopulationSampler(n_source_cells=config.evaluation.n_source_cells, n_target_cells=config.evaluation.n_target_cells)

    def _seed_prediction(self, seed: int, split: ResolvedSplit, phase: str, task_name: str, purpose: str, run_id: int = 0) -> None:
        value = stable_seed(seed, split.name, phase, task_name, purpose, run_id)
        seed_everything(value, deterministic=self.config.runtime.deterministic)

    def run(self, model_config: ModelConfig, split: ResolvedSplit, seed: int) -> ExperimentResult:
        seed_everything(seed, deterministic=self.config.runtime.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = self.config.runtime.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.config.runtime.allow_tf32
        torch.backends.cudnn.benchmark = self.config.runtime.cudnn_benchmark

        data = ExperimentData(self.store, split)
        adapter = self.registry.model(model_config.name)
        artifacts = RunArtifacts(self.config.output_dir, model_config.name, split.name, seed)
        artifacts.save_config(self.config)
        artifacts.save_split(split)
        experiment_id = f"{model_config.name}__{split.name}__seed_{seed}__{split.fingerprint}"
        result, error_to_raise, built = None, None, None
        prep_profiler = train_profiler = None

        # Inner profilers own CUDA peak counters; the outer profiler tracks total wall/CPU/RSS only.
        with PhaseProfiler(self.device, profile_cuda_memory=False) as total_profiler:
            try:
                with PhaseProfiler(self.device) as prep_profiler:
                    built = adapter.build(config=model_config, data=data, device=self.device, seed=seed)
                with PhaseProfiler(self.device) as train_profiler:
                    training_summary = adapter.fit(built=built, config=self.config.training)

                result = ExperimentResult(experiment_id=experiment_id, model=model_config.name, split=split.name, split_fingerprint=split.fingerprint, seed=seed, status="running", fit_timepoints=split.fit_timepoints, training_cell_counts=data.training_cell_counts, parameters=parameter_stats(built.model), system=system_info(), training_data_prep=prep_profiler.stats, training=train_profiler.stats, training_summary=training_summary)

                if self.config.save_checkpoints:
                    checkpoint = adapter.checkpoint_state(built)
                    if checkpoint is not None:
                        artifacts.save_checkpoint(checkpoint)

                adapter.release_training_resources(built)
                _clear_store_cache(self.store)
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                for phase, tasks in (("validation", split.validation_tasks), ("test", split.test_tasks)):
                    for task in tasks:
                        result.tasks.append(self._evaluate_task(adapter, built, data, split, task, phase, seed, artifacts))

                if self.config.evaluation.velocity.enabled:
                    for phase, tasks in (("validation", split.validation_velocity_tasks), ("test", split.test_velocity_tasks)):
                        for task in tasks:
                            result.velocity_tasks.append(self._evaluate_velocity_task(adapter, built, data, split, task, phase, seed, artifacts))
                result.status = "completed"
            except Exception as error:
                if result is None:
                    result = ExperimentResult(experiment_id=experiment_id, model=model_config.name, split=split.name, split_fingerprint=split.fingerprint, seed=seed, status="failed", fit_timepoints=split.fit_timepoints, training_cell_counts=data.training_cell_counts, parameters=parameter_stats(built.model if built is not None else None), system=system_info(), training_data_prep=None if prep_profiler is None else prep_profiler.stats, training=None if train_profiler is None else train_profiler.stats)
                else:
                    result.status = "failed"
                result.error = f"{type(error).__name__}: {error}"
                if self.config.fail_fast:
                    error_to_raise = error

        result.total = total_profiler.stats
        artifacts.save_result(result)
        _clear_store_cache(self.store)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if error_to_raise is not None:
            raise error_to_raise
        return result

    def _predict(self, adapter, built, source, source_time: float, target_time: float, n_samples: int):
        with torch.inference_mode():
            return adapter.predict(built=built, source=source, source_time=source_time, target_time=target_time, n_samples=n_samples)

    def _prediction_batches(self, adapter, built, source, source_time: float, target_time: float):
        remaining = self.config.evaluation.n_samples
        batch_size = min(self.config.evaluation.sample_batch_size, remaining)
        while remaining:
            current = min(batch_size, remaining)
            yield self._predict(adapter, built, source, source_time, target_time, current)
            remaining -= current

    def _discard_prediction_batches(self, adapter, built, source, source_time: float, target_time: float) -> None:
        for prediction in self._prediction_batches(adapter, built, source, source_time, target_time):
            del prediction

    def _evaluate_task(self, adapter, built, data: ExperimentData, split: ResolvedSplit, task, phase: str, seed: int, artifacts: RunArtifacts) -> TaskResult:
        with PhaseProfiler(self.device) as prep_profiler:
            task_data = self.sampler.sample_task(data.task_data(task, phase=phase), split_name=split.name, seed=seed)
            source_time, target_time = task_data.source.time_hours, task_data.target.time_hours
            source_indices, target_indices = task_data.source.indices.copy(), task_data.target.indices.copy()
            n_source_cells, n_target_cells = len(task_data.source), len(task_data.target)
            source, target = dense_expression(task_data.source, self.device), dense_expression(task_data.target, self.device)
            del task_data
            _clear_store_cache(self.store)
            gc.collect()
        artifacts.save_task_indices(phase, task.task_name, source_indices, target_indices)

        for run_id in range(self.config.evaluation.prediction_warmup_runs):
            self._seed_prediction(seed, split, phase, task.task_name, "warmup", run_id)
            self._discard_prediction_batches(adapter, built, source, source_time, target_time)

        prediction_runs = []
        for run_id in range(self.config.evaluation.prediction_timing_runs):
            self._seed_prediction(seed, split, phase, task.task_name, "timing", run_id)
            with PhaseProfiler(self.device) as profiler:
                self._discard_prediction_batches(adapter, built, source, source_time, target_time)
            prediction_runs.append(profiler.stats)

        self._seed_prediction(seed, split, phase, task.task_name, "evaluation")
        metric_frames, sample_index = [], 0
        with PhaseProfiler(self.device) as evaluation_profiler:
            transformed_target = transform_evaluation_space(target, self.config.evaluation.space.transform, self.config.evaluation.space.library_size)
            del target
            for prediction in self._prediction_batches(adapter, built, source, source_time, target_time):
                for local_index in range(prediction.states.shape[0]):
                    state = prediction.states[local_index : local_index + 1]
                    if self.config.save_predictions:
                        artifacts.save_prediction_sample(phase, task.task_name, sample_index, prediction, state[0])
                    transformed_state = transform_evaluation_space(state, self.config.evaluation.space.transform, self.config.evaluation.space.library_size)
                    transformed_prediction = _replace_prediction_states(prediction, transformed_state)
                    metric_frames.append(self.registry.evaluator.evaluate(transformed_prediction, transformed_target))
                    sample_index += 1
                del prediction
            metric_frame = _aggregate_metric_frames(metric_frames)

        return TaskResult(phase=phase, task_name=task.task_name, task_kind=task.resolved_kind(split.fit_timepoints), source=task.source, target=task.target, delta_hours=task.delta_hours, n_source_cells=n_source_cells, n_target_cells=n_target_cells, n_samples=self.config.evaluation.n_samples, metrics=metric_values_from_frame(metric_frame), data_prep=prep_profiler.stats, prediction_runs=prediction_runs, evaluation=evaluation_profiler.stats, source_indices_fingerprint=indices_fingerprint(source_indices), target_indices_fingerprint=indices_fingerprint(target_indices))

    def _velocity_metadata(self, data: ExperimentData, task: VelocityTask, phase: str):
        config = self.config.evaluation.velocity
        cell_id_parts, label_parts, timepoint_parts, row_index_parts = [], [], [], []

        for timepoint in task.timepoints:
            selection = data.evaluation_snapshot(timepoint, phase, task.partition)
            obs = selection.obs
            if config.cell_id_column not in obs.columns:
                raise KeyError(f"Velocity evaluation requires obs column {config.cell_id_column!r}.")
            cell_id_parts.append(obs[config.cell_id_column].astype(str).to_numpy())
            timepoint_parts.append(np.full(len(selection), selection.timepoint, dtype=object))
            row_index_parts.append(selection.indices.copy())
            labels = obs[config.color_by].astype(str).to_numpy() if config.color_by is not None and config.color_by in obs.columns else np.full(len(selection), selection.timepoint, dtype=object)
            label_parts.append(labels)
            data.unload(timepoint)

        return np.concatenate(cell_id_parts), np.concatenate(label_parts), np.concatenate(timepoint_parts), np.concatenate(row_index_parts)

    def _match_velocity_reference(self, cell_ids, labels, timepoints, row_indices, reference, seed: int, split: ResolvedSplit, phase: str, task: VelocityTask):
        query_indices, reference_indices = reference.indices_for(cell_ids)
        if not len(query_indices):
            raise ValueError(f"Velocity task {task.task_name!r} has no cells matching reference {task.reference!r}.")

        cell_ids, labels, timepoints, row_indices = cell_ids[query_indices], labels[query_indices], timepoints[query_indices], row_indices[query_indices]
        n_cells = self.config.evaluation.velocity.n_cells
        if n_cells is not None and len(cell_ids) > n_cells:
            rng = np.random.default_rng(stable_seed(seed, split.name, phase, task.task_name, "velocity_cells"))
            keep = np.sort(rng.choice(len(cell_ids), size=n_cells, replace=False))
            cell_ids, labels, timepoints, row_indices, reference_indices = cell_ids[keep], labels[keep], timepoints[keep], row_indices[keep], reference_indices[keep]
        return cell_ids, labels, timepoints, row_indices, reference_indices

    def _velocity_expression_rows(self, timepoints: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
        expression = np.empty((len(row_indices), self.store.n_genes), dtype=np.float32)
        for timepoint in np.unique(timepoints):
            positions = np.flatnonzero(timepoints == timepoint)
            snapshot = self.store.load(timepoint)
            expression[positions] = snapshot.expression[row_indices[positions]].toarray().astype(np.float32, copy=False)
            unload = getattr(self.store, "unload", None)
            if callable(unload):
                unload(timepoint)
        return expression

    def _velocity_plots(self, folder: Path, positions: np.ndarray, projected: np.ndarray, reference_vectors: np.ndarray, labels: np.ndarray, per_cell, built, task: VelocityTask, seed: int, artifacts: RunArtifacts) -> list[str]:
        config = self.config.evaluation.velocity
        if not config.plots or reference_vectors.shape[1] < 2:
            return []
        comparison_path, histogram_path = folder / "pca_velocity.png", folder / "cosine_histogram.png"
        plot_velocity_comparison(positions=positions, predicted=projected, reference=reference_vectors, labels=labels, path=comparison_path, max_cells=config.plot_max_cells, max_arrows=config.plot_max_arrows, seed=stable_seed(seed, task.task_name, "plot"), title=f"{task.task_name} | {type(built.model).__name__}")
        plot_velocity_alignment_histogram(cosine_similarity=per_cell["cosine_similarity"].to_numpy(), path=histogram_path, title=f"{task.task_name} velocity alignment")
        return [str(path.relative_to(artifacts.root)) for path in (comparison_path, histogram_path)]

    def _evaluate_velocity_task(self, adapter, built, data: ExperimentData, split: ResolvedSplit, task: VelocityTask, phase: str, seed: int, artifacts: RunArtifacts) -> VelocityTaskResult:
        if not adapter.supports_velocity(built):
            return VelocityTaskResult(phase=phase, task_name=task.task_name, reference=task.reference, timepoints=task.timepoints, n_cells=0, n_reference_components=0, status="unsupported", error=f"{type(built.model).__name__} does not expose predict_velocity().")

        reference = load_velocity_reference(_reference_path(self.config.evaluation.velocity.reference_dir, task.reference))
        _validate_reference_genes(reference, data)
        metadata = self._velocity_metadata(data, task, phase)
        cell_ids, labels, timepoints, row_indices, reference_indices = self._match_velocity_reference(*metadata, reference, seed, split, phase, task)
        reference_vectors = reference.vectors[reference_indices]
        groups = reference.groups[reference_indices] if reference.groups is not None else None
        projected = np.empty((len(cell_ids), reference.n_components), dtype=np.float32)
        computed_positions = np.empty_like(projected)

        self._seed_prediction(seed, split, phase, task.task_name, "velocity")
        batch_size = self.config.evaluation.velocity.batch_size
        with PhaseProfiler(self.device) as prediction_profiler:
            for start in range(0, len(cell_ids), batch_size):
                stop = min(start + batch_size, len(cell_ids))
                expression = self._velocity_expression_rows(timepoints[start:stop], row_indices[start:stop])
                source = torch.from_numpy(expression).to(self.device)
                with torch.inference_mode():
                    gene_velocity = adapter.predict_velocity(built=built, source=source).detach().cpu().numpy()
                positions_batch, projected_batch = project_velocity_to_reference(expression, gene_velocity, reference)
                computed_positions[start:stop] = positions_batch
                projected[start:stop] = projected_batch
                del source, gene_velocity, expression

        positions = reference.positions[reference_indices] if reference.positions is not None else computed_positions
        with PhaseProfiler(self.device) as evaluation_profiler:
            metric_frame, per_cell = velocity_alignment_metrics(projected, reference_vectors, cell_ids, groups, reference.alignment_mode)

        per_cell.insert(1, "timepoint", timepoints)
        per_cell.insert(2, "label", labels)
        if positions.shape[1] >= 2:
            for name, values_ in (("pc1", positions[:, 0]), ("pc2", positions[:, 1]), ("predicted_pc1", projected[:, 0]), ("predicted_pc2", projected[:, 1]), ("reference_pc1", reference_vectors[:, 0]), ("reference_pc2", reference_vectors[:, 1])):
                per_cell[name] = values_

        folder = artifacts.save_velocity_evaluation(phase, task.task_name, metric_frame, per_cell)
        if self.config.save_predictions:
            artifacts.save_velocity_prediction(phase, task.task_name, projected, cell_ids)
        plot_paths = self._velocity_plots(folder, positions, projected, reference_vectors, labels, per_cell, built, task, seed, artifacts)
        return VelocityTaskResult(phase=phase, task_name=task.task_name, reference=reference.name, timepoints=task.timepoints, n_cells=len(cell_ids), n_reference_components=reference.n_components, status="completed", metrics=metric_values_from_frame(metric_frame), prediction=prediction_profiler.stats, evaluation=evaluation_profiler.stats, cell_indices_fingerprint=strings_fingerprint(cell_ids), plot_paths=plot_paths)


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, registry: ExperimentRegistry, store):
        self.config, self.registry, self.store = config, registry, store

    def run(self):
        results = []
        runner = ExperimentRunner(self.config, self.registry, self.store)
        for split_spec in self.config.splits:
            for seed in self.config.seeds:
                split = resolve_split(self.store, split_spec, seed=seed)
                results.extend(runner.run(model, split, seed) for model in self.config.models if model.enabled)
        return results
