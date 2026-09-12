# std-lib imports
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# 3 party imports

# package imports
from trajectoryflow.experiment.split import CellSplitSpec, ForecastTask, SplitSpec, VelocityTask


Precision = Literal["float32"]
_TRANSFORMS = {"none", "library_log1p"}
_PRECISIONS = {"float32"}


@dataclass(frozen=True)
class TrainingBudget:
    max_seconds: float | None = None
    max_epochs: int | None = None
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.max_seconds is not None and (isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be a positive number or None.")
        for name, value in (("max_epochs", self.max_epochs), ("max_steps", self.max_steps)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None.")

    @property
    def limited(self) -> bool:
        return any(value is not None for value in (self.max_seconds, self.max_epochs, self.max_steps))


@dataclass(frozen=True)
class ModelConfig:
    name: str
    enabled: bool = True
    model_params: dict[str, Any] = field(default_factory=dict)
    trainer_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Model name must not be empty.")
        if not isinstance(self.enabled, bool):
            raise ValueError("Model enabled must be boolean.")
        if not isinstance(self.model_params, dict) or not isinstance(self.trainer_params, dict):
            raise ValueError("model_params and trainer_params must be tables/dicts.")


@dataclass(frozen=True)
class EvaluationSpaceConfig:
    transform: str = "library_log1p"
    library_size: float = 10_000.0

    def __post_init__(self) -> None:
        if self.transform not in _TRANSFORMS:
            raise ValueError(f"Unknown evaluation transform {self.transform!r}.")
        if isinstance(self.library_size, bool) or not isinstance(self.library_size, (int, float)) or self.library_size <= 0:
            raise ValueError("library_size must be a positive number.")


@dataclass(frozen=True)
class VelocityEvaluationConfig:
    enabled: bool = False
    reference_dir: Path = Path("data/processed/scifate2/velocity_references")
    n_cells: int | None = None
    batch_size: int = 256
    plots: bool = True
    plot_max_cells: int = 4000
    plot_max_arrows: int = 300
    cell_id_column: str = "cell_id"
    color_by: str | None = "cell_type"

    def __post_init__(self) -> None:
        if self.n_cells is not None and (isinstance(self.n_cells, bool) or not isinstance(self.n_cells, int) or self.n_cells < 1):
            raise ValueError("velocity n_cells must be a positive integer or None.")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("velocity batch_size must be a positive integer.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (self.plot_max_cells, self.plot_max_arrows)):
            raise ValueError("Velocity plot limits must be positive integers.")
        if not isinstance(self.enabled, bool) or not isinstance(self.plots, bool):
            raise ValueError("velocity enabled/plots must be boolean.")
        if not isinstance(self.cell_id_column, str) or not self.cell_id_column.strip():
            raise ValueError("cell_id_column must not be empty.")


@dataclass(frozen=True)
class EvaluationConfig:
    n_source_cells: int | None = 1000
    n_target_cells: int | None = 1000
    n_samples: int = 10
    sample_batch_size: int = 1
    prediction_warmup_runs: int = 0
    prediction_timing_runs: int = 1
    space: EvaluationSpaceConfig = field(default_factory=EvaluationSpaceConfig)
    velocity: VelocityEvaluationConfig = field(default_factory=VelocityEvaluationConfig)

    def __post_init__(self) -> None:
        for name, value in (("n_source_cells", self.n_source_cells), ("n_target_cells", self.n_target_cells)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None.")
        for name, value, minimum in (("n_samples", self.n_samples, 1), ("sample_batch_size", self.sample_batch_size, 1), ("prediction_warmup_runs", self.prediction_warmup_runs, 0), ("prediction_timing_runs", self.prediction_timing_runs, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    deterministic: bool = True
    precision: Precision = "float32"
    store_cache_size: int = 0
    allow_tf32: bool = False
    cudnn_benchmark: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must not be empty.")
        if self.precision not in _PRECISIONS:
            raise ValueError(f"Unsupported precision {self.precision!r}; expected one of {sorted(_PRECISIONS)}.")
        if isinstance(self.store_cache_size, bool) or not isinstance(self.store_cache_size, int) or self.store_cache_size < 0:
            raise ValueError("store_cache_size must be an integer >= 0.")
        if any(not isinstance(value, bool) for value in (self.deterministic, self.allow_tf32, self.cudnn_benchmark)):
            raise ValueError("deterministic, allow_tf32 and cudnn_benchmark must be boolean.")
        if self.deterministic and self.cudnn_benchmark:
            raise ValueError("cudnn_benchmark=True conflicts with deterministic=True.")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    metric: str
    mode: Literal["min", "max"] = "min"
    patience: int = 20
    min_delta: float = 0.0
    warmup_epochs: int = 0

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("Early-stopping metric must not be empty.")
        if self.mode not in ("min", "max"):
            raise ValueError("Early-stopping mode must be 'min' or 'max'.")
        if self.patience < 1 or self.min_delta < 0 or self.warmup_epochs < 0:
            raise ValueError("patience must be >= 1 and min_delta/warmup_epochs must be >= 0.")


@dataclass(frozen=True)
class TrainingConfig:
    budget: TrainingBudget = field(default_factory=TrainingBudget)
    early_stopping: EarlyStoppingConfig | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    data_root: Path
    output_dir: Path
    seeds: tuple[int, ...]
    models: tuple[ModelConfig, ...]
    splits: tuple[SplitSpec, ...]
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    fail_fast: bool = True
    save_checkpoints: bool = True
    save_predictions: bool = False

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("At least one benchmark seed is required.")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("Benchmark seeds must be integers.")
        if any(not isinstance(value, bool) for value in (self.fail_fast, self.save_checkpoints, self.save_predictions)):
            raise ValueError("fail_fast/save_checkpoints/save_predictions must be boolean.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Benchmark seeds must be unique.")
        if not any(model.enabled for model in self.models):
            raise ValueError("At least one model must be enabled.")
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("Model names must be unique.")
        if not self.splits:
            raise ValueError("At least one split is required.")
        if len({split.name for split in self.splits}) != len(self.splits):
            raise ValueError("Split names must be unique.")


def _reject_unknown(value: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {section} option(s): {sorted(unknown)}.")


def _task(value: dict[str, Any]) -> ForecastTask:
    _reject_unknown(value, {"source", "target", "name", "kind", "source_partition", "target_partition"}, "forecast task")
    return ForecastTask(source=value["source"], target=value["target"], name=value.get("name"), kind=value.get("kind", "auto"), source_partition=value.get("source_partition", "auto"), target_partition=value.get("target_partition", "auto"))


def _velocity_task(value: dict[str, Any]) -> VelocityTask:
    _reject_unknown(value, {"reference", "timepoints", "name", "partition"}, "velocity task")
    return VelocityTask(reference=value["reference"], timepoints=tuple(value["timepoints"]), name=value.get("name"), partition=value.get("partition", "auto"))


def _cell_split(value: dict[str, Any] | None) -> CellSplitSpec:
    value = value or {}
    _reject_unknown(value, {"train_fraction", "validation_fraction", "test_fraction", "apply_to", "stratify_by", "group_by"}, "cell split")
    return CellSplitSpec(train_fraction=float(value.get("train_fraction", 1.0)), validation_fraction=float(value.get("validation_fraction", 0.0)), test_fraction=float(value.get("test_fraction", 0.0)), apply_to=value.get("apply_to", "fit_timepoints"), stratify_by=tuple(value.get("stratify_by", ())), group_by=value.get("group_by"))


def _split(value: dict[str, Any]) -> SplitSpec:
    _reject_unknown(value, {"name", "fit_timepoints", "exclude_timepoints", "validation_tasks", "test_tasks", "validation_velocity_tasks", "test_velocity_tasks", "cell_split", "allow_validation_test_overlap"}, "split")
    fit_timepoints = value.get("fit_timepoints")
    return SplitSpec(name=value["name"], fit_timepoints=tuple(fit_timepoints) if fit_timepoints is not None else None, exclude_timepoints=tuple(value.get("exclude_timepoints", ())), validation_tasks=tuple(_task(task) for task in value.get("validation_tasks", ())), test_tasks=tuple(_task(task) for task in value.get("test_tasks", ())), validation_velocity_tasks=tuple(_velocity_task(task) for task in value.get("validation_velocity_tasks", ())), test_velocity_tasks=tuple(_velocity_task(task) for task in value.get("test_velocity_tasks", ())), cell_split=_cell_split(value.get("cell_split")), allow_validation_test_overlap=value.get("allow_validation_test_overlap", False))


def _model(value: dict[str, Any]) -> ModelConfig:
    _reject_unknown(value, {"name", "enabled", "model_params", "trainer_params"}, "model")
    return ModelConfig(name=value["name"], enabled=value.get("enabled", True), model_params=dict(value.get("model_params", {})), trainer_params=dict(value.get("trainer_params", {})))


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("rb") as file:
        raw = tomllib.load(file)

    _reject_unknown(raw, {"data_root", "output_dir", "seeds", "models", "splits", "training", "evaluation", "runtime", "fail_fast", "save_checkpoints", "save_predictions"}, "benchmark")
    training_raw = raw.get("training", {})
    _reject_unknown(training_raw, {"budget", "early_stopping"}, "training")
    budget_raw = training_raw.get("budget", {})
    _reject_unknown(budget_raw, {"max_seconds", "max_epochs", "max_steps"}, "training budget")
    early_raw = training_raw.get("early_stopping")
    if early_raw is not None:
        _reject_unknown(early_raw, {"metric", "mode", "patience", "min_delta", "warmup_epochs"}, "early stopping")
    training = TrainingConfig(budget=TrainingBudget(max_seconds=budget_raw.get("max_seconds"), max_epochs=budget_raw.get("max_epochs"), max_steps=budget_raw.get("max_steps")), early_stopping=EarlyStoppingConfig(**early_raw) if early_raw is not None else None)

    evaluation_raw = raw.get("evaluation", {})
    _reject_unknown(evaluation_raw, {"n_source_cells", "n_target_cells", "n_samples", "sample_batch_size", "prediction_warmup_runs", "prediction_timing_runs", "space", "velocity"}, "evaluation")
    space_raw = evaluation_raw.get("space", {})
    _reject_unknown(space_raw, {"transform", "library_size"}, "evaluation space")
    velocity_raw = evaluation_raw.get("velocity", {})
    _reject_unknown(velocity_raw, {"enabled", "reference_dir", "n_cells", "batch_size", "plots", "plot_max_cells", "plot_max_arrows", "cell_id_column", "color_by"}, "velocity evaluation")
    evaluation = EvaluationConfig(
        n_source_cells=evaluation_raw.get("n_source_cells", 1000), n_target_cells=evaluation_raw.get("n_target_cells", 1000), n_samples=evaluation_raw.get("n_samples", 10), sample_batch_size=evaluation_raw.get("sample_batch_size", 1),
        prediction_warmup_runs=evaluation_raw.get("prediction_warmup_runs", 0), prediction_timing_runs=evaluation_raw.get("prediction_timing_runs", 1),
        space=EvaluationSpaceConfig(transform=space_raw.get("transform", "library_log1p"), library_size=space_raw.get("library_size", 10_000.0)),
        velocity=VelocityEvaluationConfig(enabled=velocity_raw.get("enabled", False), reference_dir=Path(velocity_raw.get("reference_dir", "data/processed/scifate2/velocity_references")), n_cells=velocity_raw.get("n_cells"), batch_size=velocity_raw.get("batch_size", 256), plots=velocity_raw.get("plots", True), plot_max_cells=velocity_raw.get("plot_max_cells", 4000), plot_max_arrows=velocity_raw.get("plot_max_arrows", 300), cell_id_column=velocity_raw.get("cell_id_column", "cell_id"), color_by=velocity_raw.get("color_by", "cell_type")),
    )

    runtime_raw = raw.get("runtime", {})
    _reject_unknown(runtime_raw, {"device", "deterministic", "precision", "store_cache_size", "allow_tf32", "cudnn_benchmark"}, "runtime")
    return BenchmarkConfig(data_root=Path(raw["data_root"]), output_dir=Path(raw.get("output_dir", "runs/benchmark")), seeds=tuple(raw.get("seeds", (0,))), models=tuple(_model(model) for model in raw.get("models", ())), splits=tuple(_split(split) for split in raw.get("splits", ())), training=training, evaluation=evaluation, runtime=RuntimeConfig(**runtime_raw), fail_fast=raw.get("fail_fast", True), save_checkpoints=raw.get("save_checkpoints", True), save_predictions=raw.get("save_predictions", False))
