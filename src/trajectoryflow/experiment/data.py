# std-lib imports
from dataclasses import dataclass

# 3 party imports
import numpy as np
import pandas as pd
from scipy import sparse

# package imports
from trajectoryflow.experiment.runtime import stable_seed
from trajectoryflow.experiment.split import ForecastTask, ResolvedSplit, VelocityTask, timepoint_hours


@dataclass(frozen=True)
class SnapshotSelection:
    snapshot: object
    indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)

        if indices.ndim != 1:
            raise ValueError("SnapshotSelection indices must be one-dimensional.")

        if len(indices) and (indices.min() < 0 or indices.max() >= len(self.snapshot)):
            raise IndexError("SnapshotSelection contains out-of-range cell indices.")

        object.__setattr__(self, "indices", indices)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def timepoint(self) -> str:
        return self.snapshot.timepoint

    @property
    def time_hours(self) -> float:
        if hasattr(self.snapshot, "time_hours"):
            return float(self.snapshot.time_hours)

        return timepoint_hours(self.timepoint)

    @property
    def n_cells(self) -> int:
        return len(self)

    @property
    def n_genes(self) -> int:
        return self.snapshot.expression.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_cells, self.n_genes

    @property
    def expression(self) -> sparse.csr_matrix:
        return self.snapshot.expression[self.indices].tocsr()

    @property
    def ntr(self) -> sparse.csr_matrix:
        return self.snapshot.ntr[self.indices].tocsr()

    @property
    def new(self) -> sparse.csr_matrix:
        if not hasattr(self.snapshot, "new"):
            raise AttributeError(
                f"Snapshot {self.timepoint} does not expose a `new` RNA matrix."
            )

        return self.snapshot.new[self.indices].tocsr()

    @property
    def obs(self) -> pd.DataFrame:
        return self.snapshot.obs.iloc[self.indices].reset_index(drop=True)

    def subset(self, local_indices: np.ndarray) -> "SnapshotSelection":
        local_indices = np.asarray(local_indices, dtype=np.int64)

        if local_indices.ndim != 1:
            raise ValueError("local_indices must be one-dimensional.")

        if len(local_indices) and (
            local_indices.min() < 0 or local_indices.max() >= len(self)
        ):
            raise IndexError("Local cell indices are out of range.")

        return SnapshotSelection(
            snapshot=self.snapshot,
            indices=self.indices[local_indices],
        )


@dataclass(frozen=True)
class TaskData:
    task: ForecastTask
    phase: str
    source: SnapshotSelection
    target: SnapshotSelection


@dataclass(frozen=True)
class VelocityTaskData:
    task: VelocityTask
    phase: str
    snapshots: tuple[SnapshotSelection, ...]


class ExperimentData:
    """Restricted data view for one resolved experiment split."""

    def __init__(self, store, split: ResolvedSplit):
        self._store = store
        self.split = split

    @property
    def n_genes(self) -> int:
        return int(self._store.n_genes)

    @property
    def genes(self):
        return self._store.genes

    @property
    def preprocessing(self):
        return self._store.preprocessing

    @property
    def training_timepoints(self) -> tuple[str, ...]:
        return self.split.fit_timepoints

    @property
    def training_cell_counts(self) -> dict[str, int]:
        return {
            timepoint: len(self.split.partitions[timepoint].train_indices)
            for timepoint in self.training_timepoints
        }

    def training_snapshot(self, timepoint: str) -> SnapshotSelection:
        if timepoint not in self.training_timepoints:
            raise PermissionError(
                f"Timepoint {timepoint!r} is not available to the trainer. "
                f"Allowed fit timepoints: {self.training_timepoints}."
            )

        partition = self.split.partitions[timepoint]

        return SnapshotSelection(
            snapshot=self._store.load(timepoint),
            indices=partition.train_indices,
        )

    def training_snapshots(self) -> tuple[SnapshotSelection, ...]:
        return tuple(
            self.training_snapshot(timepoint)
            for timepoint in self.training_timepoints
        )

    def evaluation_snapshot(
        self,
        timepoint: str,
        phase: str,
        requested_partition: str = "auto",
    ) -> SnapshotSelection:
        if phase not in ("validation", "test"):
            raise ValueError("phase must be 'validation' or 'test'.")

        partition = self.split.partitions[timepoint]

        if requested_partition == "auto":
            resolved_partition = phase if partition.split_applied else "all"
        else:
            resolved_partition = requested_partition

        indices = partition.indices(resolved_partition)

        if len(indices) == 0:
            raise ValueError(
                f"Resolved {resolved_partition!r} partition for {timepoint} is empty."
            )

        return SnapshotSelection(
            snapshot=self._store.load(timepoint),
            indices=indices,
        )

    def task_data(self, task: ForecastTask, phase: str) -> TaskData:
        return TaskData(
            task=task,
            phase=phase,
            source=self.evaluation_snapshot(
                timepoint=task.source,
                phase=phase,
                requested_partition=task.source_partition,
            ),
            target=self.evaluation_snapshot(
                timepoint=task.target,
                phase=phase,
                requested_partition=task.target_partition,
            ),
        )


    def velocity_task_data(
        self,
        task: VelocityTask,
        phase: str,
    ) -> VelocityTaskData:
        snapshots = tuple(
            self.evaluation_snapshot(
                timepoint=timepoint,
                phase=phase,
                requested_partition=task.partition,
            )
            for timepoint in task.timepoints
        )

        return VelocityTaskData(
            task=task,
            phase=phase,
            snapshots=snapshots,
        )


@dataclass(frozen=True)
class PopulationSampler:
    n_source_cells: int | None = None
    n_target_cells: int | None = None

    def _sample(
        self,
        selection: SnapshotSelection,
        n_cells: int | None,
        seed: int,
    ) -> SnapshotSelection:
        if n_cells is None or n_cells <= 0 or n_cells >= len(selection):
            return selection

        rng = np.random.default_rng(seed)
        local_indices = np.sort(
            rng.choice(len(selection), size=n_cells, replace=False)
        )

        return selection.subset(local_indices)

    def sample_task(
        self,
        data: TaskData,
        split_name: str,
        seed: int,
    ) -> TaskData:
        source_seed = stable_seed(
            seed,
            split_name,
            data.phase,
            data.task.task_name,
            "source",
        )

        target_seed = stable_seed(
            seed,
            split_name,
            data.phase,
            data.task.task_name,
            "target",
        )

        return TaskData(
            task=data.task,
            phase=data.phase,
            source=self._sample(
                data.source,
                n_cells=self.n_source_cells,
                seed=source_seed,
            ),
            target=self._sample(
                data.target,
                n_cells=self.n_target_cells,
                seed=target_seed,
            ),
        )
