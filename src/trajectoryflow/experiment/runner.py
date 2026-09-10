# std-lib imports
import hashlib
import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

# 3 party imports
import numpy as np
import torch

# package imports
from trajectoryflow.experiment.plotting import (
    plot_velocity_alignment_histogram,
    plot_velocity_comparison,
)
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
from trajectoryflow.experiment.runtime import (
    PhaseProfiler,
    parameter_stats,
    seed_everything,
    stable_seed,
    system_info,
)
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
        self.root.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: ExperimentResult) -> None:
        self._write_json("result.json", result.to_dict())

        metrics = result.metrics_frame()
        runtimes = result.runtime_frame()

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
            arrays[f"{safe}__train"] = partition.train_indices
            arrays[f"{safe}__validation"] = partition.validation_indices
            arrays[f"{safe}__test"] = partition.test_indices

        np.savez_compressed(self.root / "split_indices.npz", **arrays)

        self._write_json(
            "split.json",
            {
                "name": split.name,
                "seed": split.seed,
                "fingerprint": split.fingerprint,
                "fit_timepoints": split.fit_timepoints,
                "cell_counts": split.cell_counts(),
            },
        )

    def save_task_indices(
        self,
        phase: str,
        task_name: str,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
    ) -> None:
        folder = self.root / "task_indices"
        folder.mkdir(exist_ok=True)
        name = f"{phase}__{task_name}".replace("/", "_")

        np.savez_compressed(
            folder / f"{name}.npz",
            source_indices=np.asarray(source_indices, dtype=np.int64),
            target_indices=np.asarray(target_indices, dtype=np.int64),
        )

    def velocity_folder(self, phase: str, task_name: str) -> Path:
        safe = f"{phase}__{task_name}".replace("/", "_").replace(" ", "_")
        folder = self.root / "velocity" / safe
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_velocity_evaluation(
        self,
        phase: str,
        task_name: str,
        metrics,
        per_cell,
    ) -> Path:
        folder = self.velocity_folder(phase, task_name)
        metrics.to_csv(folder / "metrics.csv", index=False)
        per_cell.to_csv(folder / "cells.csv", index=False)
        return folder

    def _write_json(self, name: str, value) -> None:
        with (self.root / name).open("w", encoding="utf-8") as file:
            json.dump(_jsonable(value), file, indent=2, ensure_ascii=False)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(value)


def indices_fingerprint(indices: np.ndarray) -> str:
    digest = hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()

    return digest[:16]


def strings_fingerprint(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _reference_path(reference_dir: Path, reference: str) -> Path:
    path = Path(reference)

    if path.suffix:
        return path if path.is_absolute() else reference_dir / path

    return reference_dir / f"{reference}.npz"


def _dataset_gene_names(data: ExperimentData) -> np.ndarray | None:
    genes = data.genes

    if hasattr(genes, "columns"):
        for column in ("gene", "gene_name", "gene_id", "feature_name"):
            if column in genes.columns:
                values = genes[column].astype(str).to_numpy()
                if len(values) == data.n_genes:
                    return values

    if hasattr(genes, "index") and len(genes.index) == data.n_genes:
        index = np.asarray(genes.index).astype(str)
        if len(set(index.tolist())) == len(index):
            return index

    return None


def _validate_reference_genes(reference, data: ExperimentData) -> None:
    if reference.n_genes != data.n_genes:
        raise ValueError(
            f"Velocity reference expects {reference.n_genes} genes, "
            f"but the experiment data has {data.n_genes}."
        )

    if reference.genes is None:
        return

    dataset_genes = _dataset_gene_names(data)

    if dataset_genes is None:
        return

    if not np.array_equal(reference.genes.astype(str), dataset_genes.astype(str)):
        raise ValueError(
            "Velocity reference gene order does not match the experiment dataset."
        )


def dense_expression(selection, device: torch.device) -> torch.Tensor:
    values = selection.expression.toarray()
    return torch.from_numpy(values).float().to(device)


def _replace_prediction_states(prediction, states: torch.Tensor):
    try:
        return replace(prediction, states=states)
    except TypeError:
        prediction.states = states
        return prediction


class ExperimentRunner:

    def __init__(
        self,
        config: BenchmarkConfig,
        registry: ExperimentRegistry,
        store,
    ):
        self.config = config
        self.registry = registry
        self.store = store
        self.device = resolve_device(config.runtime.device)
        self.sampler = PopulationSampler(
            n_source_cells=config.evaluation.n_source_cells,
            n_target_cells=config.evaluation.n_target_cells,
        )

    def run(
        self,
        model_config: ModelConfig,
        split: ResolvedSplit,
        seed: int,
    ) -> ExperimentResult:
        seed_everything(seed, deterministic=self.config.runtime.deterministic)

        torch.backends.cuda.matmul.allow_tf32 = self.config.runtime.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.config.runtime.allow_tf32
        torch.backends.cudnn.benchmark = self.config.runtime.cudnn_benchmark

        data = ExperimentData(self.store, split)
        adapter = self.registry.model(model_config.name)
        artifacts = RunArtifacts(
            root=self.config.output_dir,
            model=model_config.name,
            split=split.name,
            seed=seed,
        )

        artifacts.save_config(self.config)
        artifacts.save_split(split)

        experiment_id = (
            f"{model_config.name}__{split.name}__seed_{seed}__{split.fingerprint}"
        )

        result = None

        with PhaseProfiler(self.device) as total_profiler:
            try:
                with PhaseProfiler(self.device) as prep_profiler:
                    built = adapter.build(
                        config=model_config,
                        data=data,
                        device=self.device,
                        seed=seed,
                    )

                with PhaseProfiler(self.device) as train_profiler:
                    training_summary = adapter.fit(
                        built=built,
                        config=self.config.training,
                    )

                result = ExperimentResult(
                    experiment_id=experiment_id,
                    model=model_config.name,
                    split=split.name,
                    split_fingerprint=split.fingerprint,
                    seed=seed,
                    status="running",
                    fit_timepoints=split.fit_timepoints,
                    training_cell_counts=data.training_cell_counts,
                    parameters=parameter_stats(built.model),
                    system=system_info(),
                    training_data_prep=prep_profiler.stats,
                    training=train_profiler.stats,
                    training_summary=training_summary,
                )

                for phase, tasks in (
                    ("validation", split.validation_tasks),
                    ("test", split.test_tasks),
                ):
                    for task in tasks:
                        result.tasks.append(
                            self._evaluate_task(
                                adapter=adapter,
                                built=built,
                                data=data,
                                split=split,
                                task=task,
                                phase=phase,
                                seed=seed,
                                artifacts=artifacts,
                            )
                        )

                if self.config.evaluation.velocity.enabled:
                    for phase, tasks in (
                        ("validation", split.validation_velocity_tasks),
                        ("test", split.test_velocity_tasks),
                    ):
                        for task in tasks:
                            result.velocity_tasks.append(
                                self._evaluate_velocity_task(
                                    adapter=adapter,
                                    built=built,
                                    data=data,
                                    split=split,
                                    task=task,
                                    phase=phase,
                                    seed=seed,
                                    artifacts=artifacts,
                                )
                            )

                result.status = "completed"

            except Exception as error:
                if result is None:
                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        model=model_config.name,
                        split=split.name,
                        split_fingerprint=split.fingerprint,
                        seed=seed,
                        status="failed",
                        fit_timepoints=split.fit_timepoints,
                        training_cell_counts=data.training_cell_counts,
                        parameters=parameter_stats(
                            built.model if "built" in locals() else None
                        ),
                        system=system_info(),
                    )
                else:
                    result.status = "failed"

                result.error = f"{type(error).__name__}: {error}"

                if self.config.fail_fast:
                    raise

        result.total = total_profiler.stats
        artifacts.save_result(result)

        return result

    def _evaluate_task(
        self,
        adapter,
        built,
        data: ExperimentData,
        split: ResolvedSplit,
        task,
        phase: str,
        seed: int,
        artifacts: RunArtifacts,
    ) -> TaskResult:
        with PhaseProfiler(self.device) as prep_profiler:
            task_data = data.task_data(task, phase=phase)
            task_data = self.sampler.sample_task(
                data=task_data,
                split_name=split.name,
                seed=seed,
            )
            source = dense_expression(task_data.source, self.device)
            target = dense_expression(task_data.target, self.device)

        artifacts.save_task_indices(
            phase=phase,
            task_name=task.task_name,
            source_indices=task_data.source.indices,
            target_indices=task_data.target.indices,
        )

        for _ in range(self.config.evaluation.prediction_warmup_runs):
            adapter.predict(
                built=built,
                source=source,
                source_time=task_data.source.time_hours,
                target_time=task_data.target.time_hours,
                n_samples=self.config.evaluation.n_samples,
            )

        prediction_runs = []
        prediction = None

        for _ in range(self.config.evaluation.prediction_timing_runs):
            with PhaseProfiler(self.device) as prediction_profiler:
                prediction = adapter.predict(
                    built=built,
                    source=source,
                    source_time=task_data.source.time_hours,
                    target_time=task_data.target.time_hours,
                    n_samples=self.config.evaluation.n_samples,
                )

            prediction_runs.append(prediction_profiler.stats)

        if prediction is None:
            raise RuntimeError("Prediction did not run.")

        with PhaseProfiler(self.device) as evaluation_profiler:
            states = transform_evaluation_space(
                expression=prediction.states,
                transform=self.config.evaluation.space.transform,
                library_size=self.config.evaluation.space.library_size,
            )
            transformed_prediction = _replace_prediction_states(prediction, states)
            transformed_target = transform_evaluation_space(
                expression=target,
                transform=self.config.evaluation.space.transform,
                library_size=self.config.evaluation.space.library_size,
            )
            metric_frame = self.registry.evaluator.evaluate(
                transformed_prediction,
                transformed_target,
            )

        return TaskResult(
            phase=phase,
            task_name=task.task_name,
            task_kind=task.resolved_kind(split.fit_timepoints),
            source=task.source,
            target=task.target,
            delta_hours=task.delta_hours,
            n_source_cells=len(task_data.source),
            n_target_cells=len(task_data.target),
            n_samples=self.config.evaluation.n_samples,
            metrics=metric_values_from_frame(metric_frame),
            data_prep=prep_profiler.stats,
            prediction_runs=prediction_runs,
            evaluation=evaluation_profiler.stats,
            source_indices_fingerprint=indices_fingerprint(task_data.source.indices),
            target_indices_fingerprint=indices_fingerprint(task_data.target.indices),
        )


    def _evaluate_velocity_task(
        self,
        adapter,
        built,
        data: ExperimentData,
        split: ResolvedSplit,
        task: VelocityTask,
        phase: str,
        seed: int,
        artifacts: RunArtifacts,
    ) -> VelocityTaskResult:
        velocity_config = self.config.evaluation.velocity

        if not adapter.supports_velocity(built):
            return VelocityTaskResult(
                phase=phase,
                task_name=task.task_name,
                reference=task.reference,
                timepoints=task.timepoints,
                n_cells=0,
                n_reference_components=0,
                status="unsupported",
                error=(
                    f"{type(built.model).__name__} does not expose "
                    "predict_velocity()."
                ),
            )

        reference_path = _reference_path(
            velocity_config.reference_dir,
            task.reference,
        )
        reference = load_velocity_reference(reference_path)
        _validate_reference_genes(reference, data)

        task_data = data.velocity_task_data(task, phase=phase)
        expression_parts = []
        cell_id_parts = []
        label_parts = []
        timepoint_parts = []

        for selection in task_data.snapshots:
            obs = selection.obs

            if velocity_config.cell_id_column not in obs.columns:
                raise KeyError(
                    f"Velocity evaluation requires obs column "
                    f"{velocity_config.cell_id_column!r}."
                )

            expression_parts.append(selection.expression.toarray())
            cell_id_parts.append(
                obs[velocity_config.cell_id_column].astype(str).to_numpy()
            )
            timepoint_parts.append(
                np.full(len(selection), selection.timepoint, dtype=str)
            )

            if (
                velocity_config.color_by is not None
                and velocity_config.color_by in obs.columns
            ):
                label_parts.append(
                    obs[velocity_config.color_by].astype(str).to_numpy()
                )
            else:
                label_parts.append(
                    np.full(len(selection), selection.timepoint, dtype=str)
                )

        expression = np.vstack(expression_parts).astype(np.float32, copy=False)
        cell_ids = np.concatenate(cell_id_parts)
        labels = np.concatenate(label_parts)
        timepoints = np.concatenate(timepoint_parts)

        query_indices, reference_indices = reference.indices_for(cell_ids)

        if not len(query_indices):
            raise ValueError(
                f"Velocity task {task.task_name!r} has no cells matching "
                f"reference {reference_path}."
            )

        expression = expression[query_indices]
        cell_ids = cell_ids[query_indices]
        labels = labels[query_indices]
        timepoints = timepoints[query_indices]
        reference_indices = reference_indices

        if (
            velocity_config.n_cells is not None
            and len(expression) > velocity_config.n_cells
        ):
            rng = np.random.default_rng(
                stable_seed(
                    seed,
                    split.name,
                    phase,
                    task.task_name,
                    "velocity_cells",
                )
            )
            keep = np.sort(
                rng.choice(
                    len(expression),
                    size=velocity_config.n_cells,
                    replace=False,
                )
            )
            expression = expression[keep]
            cell_ids = cell_ids[keep]
            labels = labels[keep]
            timepoints = timepoints[keep]
            reference_indices = reference_indices[keep]

        source = torch.from_numpy(expression).float().to(self.device)

        with PhaseProfiler(self.device) as prediction_profiler:
            predicted_gene_velocity = adapter.predict_velocity(
                built=built,
                source=source,
            )

        predicted_gene_velocity = (
            predicted_gene_velocity.detach().cpu().numpy()
        )
        reference_vectors = reference.vectors[reference_indices]
        groups = (
            reference.groups[reference_indices]
            if reference.groups is not None
            else None
        )

        with PhaseProfiler(self.device) as evaluation_profiler:
            computed_positions, projected_velocity = project_velocity_to_reference(
                expression=expression,
                velocity=predicted_gene_velocity,
                reference=reference,
            )
            positions = (
                reference.positions[reference_indices]
                if reference.positions is not None
                else computed_positions
            )
            metric_frame, per_cell = velocity_alignment_metrics(
                predicted=projected_velocity,
                reference=reference_vectors,
                cell_ids=cell_ids,
                groups=groups,
                alignment_mode=reference.alignment_mode,
            )

        per_cell.insert(1, "timepoint", timepoints)
        per_cell.insert(2, "label", labels)

        if positions.shape[1] >= 2:
            per_cell["pc1"] = positions[:, 0]
            per_cell["pc2"] = positions[:, 1]
            per_cell["predicted_pc1"] = projected_velocity[:, 0]
            per_cell["predicted_pc2"] = projected_velocity[:, 1]
            per_cell["reference_pc1"] = reference_vectors[:, 0]
            per_cell["reference_pc2"] = reference_vectors[:, 1]

        folder = artifacts.save_velocity_evaluation(
            phase=phase,
            task_name=task.task_name,
            metrics=metric_frame,
            per_cell=per_cell,
        )
        plot_paths = []

        if velocity_config.plots and reference.n_components >= 2:
            comparison_path = folder / "pca_velocity.png"
            histogram_path = folder / "cosine_histogram.png"

            plot_velocity_comparison(
                positions=positions,
                predicted=projected_velocity,
                reference=reference_vectors,
                labels=labels,
                path=comparison_path,
                max_cells=velocity_config.plot_max_cells,
                max_arrows=velocity_config.plot_max_arrows,
                seed=stable_seed(seed, task.task_name, "plot"),
                title=f"{task.task_name} | {type(built.model).__name__}",
            )
            plot_velocity_alignment_histogram(
                cosine_similarity=per_cell["cosine_similarity"].to_numpy(),
                path=histogram_path,
                title=f"{task.task_name} velocity alignment",
            )
            plot_paths = [
                str(comparison_path.relative_to(artifacts.root)),
                str(histogram_path.relative_to(artifacts.root)),
            ]

        return VelocityTaskResult(
            phase=phase,
            task_name=task.task_name,
            reference=reference.name,
            timepoints=task.timepoints,
            n_cells=len(cell_ids),
            n_reference_components=reference.n_components,
            status="completed",
            metrics=metric_values_from_frame(metric_frame),
            prediction=prediction_profiler.stats,
            evaluation=evaluation_profiler.stats,
            cell_indices_fingerprint=strings_fingerprint(cell_ids),
            plot_paths=plot_paths,
        )


class BenchmarkRunner:

    def __init__(
        self,
        config: BenchmarkConfig,
        registry: ExperimentRegistry,
        store,
    ):
        self.config = config
        self.registry = registry
        self.store = store

    def run(self):
        results = []
        runner = ExperimentRunner(
            config=self.config,
            registry=self.registry,
            store=self.store,
        )

        for split_spec in self.config.splits:
            for seed in self.config.seeds:
                split = resolve_split(
                    store=self.store,
                    spec=split_spec,
                    seed=seed,
                )

                for model in self.config.models:
                    if not model.enabled:
                        continue

                    results.append(
                        runner.run(
                            model_config=model,
                            split=split,
                            seed=seed,
                        )
                    )

        return results
