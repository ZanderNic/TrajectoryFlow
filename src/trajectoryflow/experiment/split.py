# std-lib imports
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

# 3 party imports
import numpy as np
import pandas as pd

# package imports
from trajectoryflow.experiment.runtime import stable_seed


TaskKind = Literal[
    "auto",
    "observed_transition",
    "interpolation",
    "extrapolation",
    "backcast",
]

CellPartition = Literal[
    "auto",
    "all",
    "train",
    "validation",
    "test",
]


def timepoint_hours(timepoint: str) -> float:
    value = str(timepoint).strip().lower()

    if not value.endswith("h"):
        raise ValueError(
            f"Timepoint {timepoint!r} must use the canonical '<hours>h' format."
        )

    try:
        hours = float(value[:-1])
    except ValueError as error:
        raise ValueError(f"Invalid timepoint {timepoint!r}.") from error

    if hours < 0:
        raise ValueError("Timepoints must be >= 0 hours.")

    return hours


@dataclass(frozen=True)
class ForecastTask:
    source: str
    target: str
    name: str | None = None
    kind: TaskKind = "auto"
    source_partition: CellPartition = "auto"
    target_partition: CellPartition = "auto"

    def __post_init__(self) -> None:
        if self.source_hours == self.target_hours:
            raise ValueError("Source and target timepoints must differ.")
        if self.kind not in ("auto", "observed_transition", "interpolation", "extrapolation", "backcast"):
            raise ValueError(f"Unknown task kind {self.kind!r}.")
        valid_partitions = ("auto", "all", "train", "validation", "test")
        if self.source_partition not in valid_partitions or self.target_partition not in valid_partitions:
            raise ValueError("Unknown source/target cell partition.")
        if self.name is not None and not self.name.strip():
            raise ValueError("Task name must not be empty.")

    @property
    def task_name(self) -> str:
        return self.name or f"{self.source}_to_{self.target}"

    @property
    def source_hours(self) -> float:
        return timepoint_hours(self.source)

    @property
    def target_hours(self) -> float:
        return timepoint_hours(self.target)

    @property
    def delta_hours(self) -> float:
        return self.target_hours - self.source_hours

    def resolved_kind(self, fit_timepoints: tuple[str, ...]) -> str:
        if self.kind != "auto":
            return self.kind

        fit_hours = sorted(timepoint_hours(tp) for tp in fit_timepoints)

        if not fit_hours:
            raise ValueError("At least one fit timepoint is required.")

        if self.target_hours < self.source_hours:
            return "backcast"

        if self.target in fit_timepoints:
            return "observed_transition"

        if fit_hours[0] < self.target_hours < fit_hours[-1]:
            return "interpolation"

        if self.target_hours > fit_hours[-1]:
            return "extrapolation"

        if self.target_hours < fit_hours[0]:
            return "backcast"

        return "observed_transition"


@dataclass(frozen=True)
class VelocityTask:
    reference: str
    timepoints: tuple[str, ...]
    name: str | None = None
    partition: CellPartition = "auto"

    def __post_init__(self) -> None:
        if not self.reference.strip():
            raise ValueError("VelocityTask reference must not be empty.")
        if not self.timepoints:
            raise ValueError("VelocityTask requires at least one timepoint.")
        if len(set(self.timepoints)) != len(self.timepoints):
            raise ValueError("VelocityTask timepoints must be unique.")
        if self.partition not in ("auto", "all", "train", "validation", "test"):
            raise ValueError(f"Unknown velocity task partition {self.partition!r}.")
        for timepoint in self.timepoints:
            timepoint_hours(timepoint)

    @property
    def task_name(self) -> str:
        if self.name:
            return self.name
        timepoints = "_".join(self.timepoints)
        return f"{self.reference}__{timepoints}"


SplitApplication = Literal["fit_timepoints", "all"]


@dataclass(frozen=True)
class CellSplitSpec:
    train_fraction: float = 1.0
    validation_fraction: float = 0.0
    test_fraction: float = 0.0
    apply_to: SplitApplication = "fit_timepoints"
    stratify_by: tuple[str, ...] = ()
    group_by: str | None = None

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in fractions):
            raise ValueError("Cell split fractions must be finite numbers.")
        if any(value < 0 or value > 1 for value in fractions):
            raise ValueError("Cell split fractions must lie in [0, 1].")

        if not np.isclose(sum(fractions), 1.0):
            raise ValueError(
                "train_fraction + validation_fraction + test_fraction must equal 1."
            )

        if self.apply_to not in ("fit_timepoints", "all"):
            raise ValueError(f"Unsupported apply_to={self.apply_to!r}.")

        if len(set(self.stratify_by)) != len(self.stratify_by):
            raise ValueError("stratify_by columns must be unique.")
        if any(not column.strip() for column in self.stratify_by):
            raise ValueError("stratify_by columns must not be empty.")
        if self.group_by is not None and not self.group_by.strip():
            raise ValueError("group_by must not be empty.")
        if self.group_by is not None and self.stratify_by:
            raise ValueError("group_by and stratify_by cannot be combined; group holdout does not use cell-level stratification.")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    fit_timepoints: tuple[str, ...] | None = None
    exclude_timepoints: tuple[str, ...] = ()
    validation_tasks: tuple[ForecastTask, ...] = ()
    test_tasks: tuple[ForecastTask, ...] = ()
    validation_velocity_tasks: tuple[VelocityTask, ...] = ()
    test_velocity_tasks: tuple[VelocityTask, ...] = ()
    cell_split: CellSplitSpec = field(default_factory=CellSplitSpec)
    allow_validation_test_overlap: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Split name must not be empty.")
        if not isinstance(self.allow_validation_test_overlap, bool):
            raise ValueError("allow_validation_test_overlap must be boolean.")
        if self.fit_timepoints is not None and self.exclude_timepoints:
            raise ValueError(
                "Specify either fit_timepoints or exclude_timepoints, not both."
            )

        if not any(
            (
                self.validation_tasks,
                self.test_tasks,
                self.validation_velocity_tasks,
                self.test_velocity_tasks,
            )
        ):
            raise ValueError(
                "A split needs at least one forecast or velocity validation/test task."
            )
        if self.fit_timepoints is not None and len(set(self.fit_timepoints)) != len(self.fit_timepoints):
            raise ValueError("fit_timepoints must be unique.")
        if len(set(self.exclude_timepoints)) != len(self.exclude_timepoints):
            raise ValueError("exclude_timepoints must be unique.")
        for timepoint in (*(() if self.fit_timepoints is None else self.fit_timepoints), *self.exclude_timepoints):
            timepoint_hours(timepoint)
        tasks = (*self.validation_tasks, *self.test_tasks, *self.validation_velocity_tasks, *self.test_velocity_tasks)
        names = [task.task_name for task in tasks]
        if len(set(names)) != len(names):
            raise ValueError("Task names within a split must be unique.")


@dataclass(frozen=True)
class CellPartitions:
    all_indices: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    split_applied: bool

    def indices(self, partition: str) -> np.ndarray:
        if partition == "all":
            return self.all_indices
        if partition == "train":
            return self.train_indices
        if partition == "validation":
            return self.validation_indices
        if partition == "test":
            return self.test_indices

        raise ValueError(f"Unknown cell partition {partition!r}.")


@dataclass(frozen=True)
class ResolvedSplit:
    spec: SplitSpec
    seed: int
    fit_timepoints: tuple[str, ...]
    partitions: dict[str, CellPartitions]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def validation_tasks(self) -> tuple[ForecastTask, ...]:
        return self.spec.validation_tasks

    @property
    def test_tasks(self) -> tuple[ForecastTask, ...]:
        return self.spec.test_tasks

    @property
    def validation_velocity_tasks(self) -> tuple[VelocityTask, ...]:
        return self.spec.validation_velocity_tasks

    @property
    def test_velocity_tasks(self) -> tuple[VelocityTask, ...]:
        return self.spec.test_velocity_tasks

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(asdict(self.spec), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(str(self.seed).encode("utf-8"))

        for timepoint in sorted(self.partitions):
            partition = self.partitions[timepoint]
            digest.update(timepoint.encode("utf-8"))

            for indices in (
                partition.train_indices,
                partition.validation_indices,
                partition.test_indices,
            ):
                digest.update(np.asarray(indices, dtype=np.int64).tobytes())

        return digest.hexdigest()[:16]

    def cell_counts(self) -> dict[str, dict[str, int]]:
        return {
            timepoint: {
                "all": len(partition.all_indices),
                "train": len(partition.train_indices),
                "validation": len(partition.validation_indices),
                "test": len(partition.test_indices),
            }
            for timepoint, partition in self.partitions.items()
        }


def _allocate_counts(n: int, fractions: tuple[float, float, float]) -> np.ndarray:
    expected = np.asarray(fractions, dtype=np.float64) * n
    counts = np.floor(expected).astype(np.int64)
    remainder = n - int(counts.sum())

    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1

    return counts


def _split_plain(
    indices: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    counts = _allocate_counts(len(shuffled), fractions)

    first = int(counts[0])
    second = first + int(counts[1])

    return (
        np.sort(shuffled[:first]),
        np.sort(shuffled[first:second]),
        np.sort(shuffled[second:]),
    )


def _strata(obs: pd.DataFrame, columns: tuple[str, ...]) -> list[np.ndarray]:
    if not columns:
        return [np.arange(len(obs), dtype=np.int64)]

    missing = [column for column in columns if column not in obs.columns]

    if missing:
        raise KeyError(f"Cannot stratify by missing obs columns: {missing}.")

    grouped = obs.groupby(
        list(columns),
        sort=False,
        dropna=False,
        observed=False,
    ).indices

    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def _split_cells(
    obs: pd.DataFrame,
    spec: CellSplitSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = (
        spec.train_fraction,
        spec.validation_fraction,
        spec.test_fraction,
    )

    if spec.group_by is None:
        output = [[], [], []]

        for group_id, indices in enumerate(_strata(obs, spec.stratify_by)):
            parts = _split_plain(
                indices=indices,
                fractions=fractions,
                seed=stable_seed(seed, "stratum", group_id),
            )

            for partition_id, part in enumerate(parts):
                output[partition_id].append(part)

        return tuple(
            np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
            if parts
            else np.empty(0, dtype=np.int64)
            for parts in output
        )

    if spec.group_by not in obs.columns:
        raise KeyError(f"Cannot group by missing obs column {spec.group_by!r}.")

    group_values = obs[spec.group_by].astype(str).to_numpy()
    unique_groups = np.unique(group_values)
    group_ids = np.arange(len(unique_groups), dtype=np.int64)
    train_groups, val_groups, test_groups = _split_plain(
        indices=group_ids,
        fractions=fractions,
        seed=seed,
    )

    partitions = []

    for selected in (train_groups, val_groups, test_groups):
        selected_groups = set(unique_groups[selected].tolist())
        mask = np.fromiter(
            (value in selected_groups for value in group_values),
            dtype=bool,
            count=len(group_values),
        )
        partitions.append(np.flatnonzero(mask).astype(np.int64))

    return tuple(partitions)


def _task_timepoints(
    tasks: tuple[ForecastTask | VelocityTask, ...],
) -> set[str]:
    output = set()

    for task in tasks:
        if isinstance(task, ForecastTask):
            output.update((task.source, task.target))
        else:
            output.update(task.timepoints)

    return output


def _validate_task_timepoints(
    available: set[str],
    tasks: tuple[ForecastTask | VelocityTask, ...],
    phase: str,
) -> None:
    for task in tasks:
        timepoints = (
            (task.source, task.target)
            if isinstance(task, ForecastTask)
            else task.timepoints
        )

        for timepoint in timepoints:
            if timepoint not in available:
                raise ValueError(
                    f"{phase} task {task.task_name!r} uses unavailable "
                    f"timepoint {timepoint!r}."
                )


def _validate_phase_partitions(split: ResolvedSplit) -> None:
    for phase, forecast_tasks, velocity_tasks in (
        ("validation", split.validation_tasks, split.validation_velocity_tasks),
        ("test", split.test_tasks, split.test_velocity_tasks),
    ):
        for task in forecast_tasks:
            for role, timepoint, explicit in (
                ("source", task.source, task.source_partition),
                ("target", task.target, task.target_partition),
            ):
                partition = split.partitions[timepoint]

                if explicit != "auto" or not partition.split_applied:
                    continue

                if len(partition.indices(phase)) == 0:
                    raise ValueError(
                        f"{phase} task {task.task_name!r} requires held-out "
                        f"{role} cells at {timepoint}, but the {phase}_fraction is 0. "
                        f"Set {role}_partition='all' explicitly if this is intended."
                    )

        for task in velocity_tasks:
            for timepoint in task.timepoints:
                partition = split.partitions[timepoint]

                if task.partition != "auto" or not partition.split_applied:
                    continue

                if len(partition.indices(phase)) == 0:
                    raise ValueError(
                        f"{phase} velocity task {task.task_name!r} requires held-out "
                        f"cells at {timepoint}, but the {phase}_fraction is 0. "
                        "Set partition='all' explicitly if this is intended."
                    )


def _validate_validation_test_overlap(split: ResolvedSplit) -> None:
    if split.spec.allow_validation_test_overlap:
        return

    validation = _task_timepoints(
        split.validation_tasks + split.validation_velocity_tasks
    )
    test = _task_timepoints(
        split.test_tasks + split.test_velocity_tasks
    )

    for timepoint in validation & test:
        if not split.partitions[timepoint].split_applied:
            raise ValueError(
                f"Timepoint {timepoint} is used by validation and test without "
                "a cell split. This evaluates both phases on the same cells."
            )


def resolve_split(store, spec: SplitSpec, seed: int) -> ResolvedSplit:
    available = tuple(store.timepoints)
    available_set = set(available)

    _validate_task_timepoints(available_set, spec.validation_tasks, "Validation")
    _validate_task_timepoints(available_set, spec.test_tasks, "Test")
    _validate_task_timepoints(
        available_set,
        spec.validation_velocity_tasks,
        "Validation velocity",
    )
    _validate_task_timepoints(
        available_set,
        spec.test_velocity_tasks,
        "Test velocity",
    )

    if spec.fit_timepoints is None:
        excluded = set(spec.exclude_timepoints)
        unknown = excluded - available_set

        if unknown:
            raise ValueError(f"Excluded timepoints are unavailable: {sorted(unknown)}.")

        fit_timepoints = tuple(tp for tp in available if tp not in excluded)
    else:
        unknown = set(spec.fit_timepoints) - available_set

        if unknown:
            raise ValueError(f"Fit timepoints are unavailable: {sorted(unknown)}.")

        fit_timepoints = tuple(spec.fit_timepoints)

    if not fit_timepoints:
        raise ValueError("At least one fit timepoint is required.")

    split_timepoints = (
        set(available)
        if spec.cell_split.apply_to == "all"
        else set(fit_timepoints)
    )
    has_holdout = (
        spec.cell_split.validation_fraction > 0
        or spec.cell_split.test_fraction > 0
    )
    fit_set = set(fit_timepoints)
    partitions = {}

    for timepoint in available:
        snapshot = store.load(timepoint)
        all_indices = np.arange(len(snapshot), dtype=np.int64)

        if timepoint in split_timepoints:
            train, validation, test = _split_cells(
                obs=snapshot.obs,
                spec=spec.cell_split,
                seed=stable_seed(seed, spec.name, timepoint),
            )

            partitions[timepoint] = CellPartitions(
                all_indices=all_indices,
                train_indices=(
                    train if timepoint in fit_set else np.empty(0, dtype=np.int64)
                ),
                validation_indices=validation,
                test_indices=test,
                split_applied=has_holdout,
            )
        else:
            partitions[timepoint] = CellPartitions(
                all_indices=all_indices,
                train_indices=(
                    all_indices if timepoint in fit_set else np.empty(0, dtype=np.int64)
                ),
                validation_indices=np.empty(0, dtype=np.int64),
                test_indices=np.empty(0, dtype=np.int64),
                split_applied=False,
            )

    empty_training = [timepoint for timepoint in fit_timepoints if len(partitions[timepoint].train_indices) == 0]
    if empty_training:
        raise ValueError(f"Training partition is empty for fit timepoints: {empty_training}.")

    resolved = ResolvedSplit(
        spec=spec,
        seed=seed,
        fit_timepoints=fit_timepoints,
        partitions=partitions,
    )

    _validate_phase_partitions(resolved)
    _validate_validation_test_overlap(resolved)

    return resolved
