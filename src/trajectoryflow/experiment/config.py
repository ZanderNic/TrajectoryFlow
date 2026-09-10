# std-lib imports
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

# 3 party imports

# package imports
from trajectoryflow.experiment.split import CellSplitSpec, ForecastTask, SplitSpec, VelocityTask


@dataclass(frozen=True)
class TrainingBudget:
    max_seconds: float | None = None
    max_epochs: int | None = None
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be > 0.")
        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ValueError("max_epochs must be > 0.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be > 0.")

    @property
    def limited(self) -> bool:
        return any(
            value is not None
            for value in (self.max_seconds, self.max_epochs, self.max_steps)
        )


class BudgetTracker:

    def __init__(self, budget: TrainingBudget):
        self.budget = budget
        self.started_at = perf_counter()
        self.epochs = 0
        self.steps = 0
        self.stop_reason: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.started_at

    @property
    def remaining_seconds(self) -> float | None:
        if self.budget.max_seconds is None:
            return None

        return max(self.budget.max_seconds - self.elapsed_seconds, 0.0)

    def mark_step(self, count: int = 1) -> None:
        self.steps += count

    def mark_epoch(self, count: int = 1) -> None:
        self.epochs += count

    def should_stop(self) -> bool:
        if self.budget.max_seconds is not None:
            if self.elapsed_seconds >= self.budget.max_seconds:
                self.stop_reason = "max_seconds"
                return True

        if self.budget.max_steps is not None:
            if self.steps >= self.budget.max_steps:
                self.stop_reason = "max_steps"
                return True

        if self.budget.max_epochs is not None:
            if self.epochs >= self.budget.max_epochs:
                self.stop_reason = "max_epochs"
                return True

        return False


Precision = Literal["float32", "amp_float16", "amp_bfloat16"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    enabled: bool = True
    model_params: dict[str, Any] = field(default_factory=dict)
    trainer_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSpaceConfig:
    transform: str = "library_log1p"
    library_size: float = 10_000.0


@dataclass(frozen=True)
class VelocityEvaluationConfig:
    enabled: bool = False
    reference_dir: Path = Path("data/processed/scifate2/velocity_references")
    n_cells: int | None = None
    plots: bool = True
    plot_max_cells: int = 4000
    plot_max_arrows: int = 300
    cell_id_column: str = "cell_id"
    color_by: str | None = "cell_type"

    def __post_init__(self) -> None:
        if self.n_cells is not None and self.n_cells < 1:
            raise ValueError("velocity n_cells must be >= 1 or None.")
        if self.plot_max_cells < 1:
            raise ValueError("plot_max_cells must be >= 1.")
        if self.plot_max_arrows < 1:
            raise ValueError("plot_max_arrows must be >= 1.")
        if not self.cell_id_column:
            raise ValueError("cell_id_column must not be empty.")


@dataclass(frozen=True)
class EvaluationConfig:
    n_source_cells: int | None = 1000
    n_target_cells: int | None = 1000
    n_samples: int = 10
    prediction_warmup_runs: int = 0
    prediction_timing_runs: int = 1
    space: EvaluationSpaceConfig = field(default_factory=EvaluationSpaceConfig)
    velocity: VelocityEvaluationConfig = field(
        default_factory=VelocityEvaluationConfig
    )

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        if self.prediction_warmup_runs < 0:
            raise ValueError("prediction_warmup_runs must be >= 0.")
        if self.prediction_timing_runs < 1:
            raise ValueError("prediction_timing_runs must be >= 1.")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    deterministic: bool = True
    precision: Precision = "float32"
    allow_tf32: bool = False
    cudnn_benchmark: bool = False


@dataclass(frozen=True)
class EarlyStoppingConfig:
    metric: str
    mode: Literal["min", "max"] = "min"
    patience: int = 20
    min_delta: float = 0.0
    warmup_epochs: int = 0


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
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Benchmark seeds must be unique.")
        if not any(model.enabled for model in self.models):
            raise ValueError("At least one model must be enabled.")
        if not self.splits:
            raise ValueError("At least one split is required.")


def _task(value: dict[str, Any]) -> ForecastTask:
    return ForecastTask(
        source=value["source"],
        target=value["target"],
        name=value.get("name"),
        kind=value.get("kind", "auto"),
        source_partition=value.get("source_partition", "auto"),
        target_partition=value.get("target_partition", "auto"),
    )


def _velocity_task(value: dict[str, Any]) -> VelocityTask:
    return VelocityTask(
        reference=value["reference"],
        timepoints=tuple(value["timepoints"]),
        name=value.get("name"),
        partition=value.get("partition", "auto"),
    )


def _cell_split(value: dict[str, Any] | None) -> CellSplitSpec:
    value = value or {}

    return CellSplitSpec(
        train_fraction=float(value.get("train_fraction", 1.0)),
        validation_fraction=float(value.get("validation_fraction", 0.0)),
        test_fraction=float(value.get("test_fraction", 0.0)),
        apply_to=value.get("apply_to", "fit_timepoints"),
        stratify_by=tuple(value.get("stratify_by", ())),
        group_by=value.get("group_by"),
    )


def _split(value: dict[str, Any]) -> SplitSpec:
    fit_timepoints = value.get("fit_timepoints")

    return SplitSpec(
        name=value["name"],
        fit_timepoints=(
            tuple(fit_timepoints)
            if fit_timepoints is not None
            else None
        ),
        exclude_timepoints=tuple(value.get("exclude_timepoints", ())),
        validation_tasks=tuple(
            _task(task) for task in value.get("validation_tasks", ())
        ),
        test_tasks=tuple(_task(task) for task in value.get("test_tasks", ())),
        validation_velocity_tasks=tuple(
            _velocity_task(task)
            for task in value.get("validation_velocity_tasks", ())
        ),
        test_velocity_tasks=tuple(
            _velocity_task(task)
            for task in value.get("test_velocity_tasks", ())
        ),
        cell_split=_cell_split(value.get("cell_split")),
        allow_validation_test_overlap=bool(
            value.get("allow_validation_test_overlap", False)
        ),
    )


def _model(value: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name=value["name"],
        enabled=bool(value.get("enabled", True)),
        model_params=dict(value.get("model_params", {})),
        trainer_params=dict(value.get("trainer_params", {})),
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)

    with path.open("rb") as file:
        raw = tomllib.load(file)

    training_raw = raw.get("training", {})
    budget_raw = training_raw.get("budget", {})
    early_raw = training_raw.get("early_stopping")

    training = TrainingConfig(
        budget=TrainingBudget(
            max_seconds=budget_raw.get("max_seconds"),
            max_epochs=budget_raw.get("max_epochs"),
            max_steps=budget_raw.get("max_steps"),
        ),
        early_stopping=(
            EarlyStoppingConfig(**early_raw)
            if early_raw is not None
            else None
        ),
    )

    evaluation_raw = raw.get("evaluation", {})
    space_raw = evaluation_raw.get("space", {})
    velocity_raw = evaluation_raw.get("velocity", {})

    evaluation = EvaluationConfig(
        n_source_cells=evaluation_raw.get("n_source_cells", 1000),
        n_target_cells=evaluation_raw.get("n_target_cells", 1000),
        n_samples=int(evaluation_raw.get("n_samples", 10)),
        prediction_warmup_runs=int(
            evaluation_raw.get("prediction_warmup_runs", 0)
        ),
        prediction_timing_runs=int(
            evaluation_raw.get("prediction_timing_runs", 1)
        ),
        space=EvaluationSpaceConfig(
            transform=space_raw.get("transform", "library_log1p"),
            library_size=float(space_raw.get("library_size", 10_000.0)),
        ),
        velocity=VelocityEvaluationConfig(
            enabled=bool(velocity_raw.get("enabled", False)),
            reference_dir=Path(
                velocity_raw.get(
                    "reference_dir",
                    "data/processed/scifate2/velocity_references",
                )
            ),
            n_cells=velocity_raw.get("n_cells"),
            plots=bool(velocity_raw.get("plots", True)),
            plot_max_cells=int(velocity_raw.get("plot_max_cells", 4000)),
            plot_max_arrows=int(velocity_raw.get("plot_max_arrows", 300)),
            cell_id_column=str(
                velocity_raw.get("cell_id_column", "cell_id")
            ),
            color_by=velocity_raw.get("color_by", "cell_type"),
        ),
    )

    return BenchmarkConfig(
        data_root=Path(raw["data_root"]),
        output_dir=Path(raw.get("output_dir", "runs/benchmark")),
        seeds=tuple(int(seed) for seed in raw.get("seeds", (0,))),
        models=tuple(_model(model) for model in raw.get("models", ())),
        splits=tuple(_split(split) for split in raw.get("splits", ())),
        training=training,
        evaluation=evaluation,
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        fail_fast=bool(raw.get("fail_fast", True)),
        save_checkpoints=bool(raw.get("save_checkpoints", True)),
        save_predictions=bool(raw.get("save_predictions", False)),
    )
