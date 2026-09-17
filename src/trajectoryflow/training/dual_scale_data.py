# std-lib imports
from dataclasses import dataclass
from typing import Protocol

# 3 party imports
import numpy as np
import torch
from scipy import sparse

# package imports
from trajectoryflow.training.dual_scale import GlobalBatch, LocalBatch


class KineticPopulationLike(Protocol):
    labeling_time: float

    def __len__(self) -> int: ...

    @property
    def n_genes(self) -> int: ...

    def take(self, indices: torch.Tensor) -> LocalBatch: ...


@dataclass(frozen=True)
class KineticPopulation:
    total: torch.Tensor
    new: torch.Tensor
    old: torch.Tensor | None = None
    ntr: torch.Tensor | None = None
    labeling_time: float = 2.0

    def __post_init__(self) -> None:
        if self.total.ndim != 2 or self.new.shape != self.total.shape:
            raise ValueError("total and new must have identical [n_cells, n_genes] shape.")
        for name, value in (("old", self.old), ("ntr", self.ntr)):
            if value is not None and value.shape != self.total.shape:
                raise ValueError(f"{name} must match total shape.")

    def __len__(self) -> int:
        return self.total.shape[0]

    @property
    def n_genes(self) -> int:
        return self.total.shape[1]

    def take(self, indices: torch.Tensor) -> LocalBatch:
        return LocalBatch(
            total=self.total[indices],
            new=self.new[indices],
            old=None if self.old is None else self.old[indices],
            ntr=None if self.ntr is None else self.ntr[indices],
            labeling_time=self.labeling_time,
        )


@dataclass(frozen=True)
class SparseKineticPopulation:
    """CSR-backed population; only selected rows are densified."""

    total: sparse.csr_matrix
    new: sparse.csr_matrix
    old: sparse.csr_matrix | None = None
    ntr: sparse.csr_matrix | None = None
    labeling_time: float = 2.0

    def __post_init__(self) -> None:
        if self.total.shape != self.new.shape:
            raise ValueError("total and new must have identical [n_cells, n_genes] shape.")
        for name, value in (("old", self.old), ("ntr", self.ntr)):
            if value is not None and value.shape != self.total.shape:
                raise ValueError(f"{name} must match total shape.")

    def __len__(self) -> int:
        return self.total.shape[0]

    @property
    def n_genes(self) -> int:
        return self.total.shape[1]

    @staticmethod
    def _dense(matrix: sparse.csr_matrix, indices: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(matrix[indices].toarray().astype(np.float32, copy=False))

    def take(self, indices: torch.Tensor) -> LocalBatch:
        rows = indices.detach().cpu().numpy()
        return LocalBatch(
            total=self._dense(self.total, rows),
            new=self._dense(self.new, rows),
            old=None if self.old is None else self._dense(self.old, rows),
            ntr=None if self.ntr is None else self._dense(self.ntr, rows),
            labeling_time=self.labeling_time,
        )


class TotalPopulation:
    """Dense or sparse total-RNA population with row-wise materialization."""

    def __init__(self, total: torch.Tensor | sparse.csr_matrix):
        if total.ndim != 2:
            raise ValueError("total must be a two-dimensional matrix.")
        self.total = total

    def __len__(self) -> int:
        return self.total.shape[0]

    @property
    def n_genes(self) -> int:
        return self.total.shape[1]

    def take(self, indices: torch.Tensor) -> torch.Tensor:
        if isinstance(self.total, torch.Tensor):
            return self.total[indices]
        rows = indices.detach().cpu().numpy()
        return torch.from_numpy(self.total[rows].toarray().astype(np.float32, copy=False))


class LocalPopulationLoader:
    """Simple shuffled local batches with exact within-cell old/new pairing."""

    def __init__(
        self,
        population: KineticPopulationLike,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.population, self.batch_size, self.shuffle, self.seed = population, batch_size, shuffle, seed
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.population) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        indices = torch.randperm(len(self.population), generator=generator) if self.shuffle else torch.arange(len(self.population))
        for start in range(0, len(indices), self.batch_size):
            yield self.population.take(indices[start : start + self.batch_size])


class UnpairedGlobalPopulationLoader:
    """
    Independently sample source, future-target and optional past-target cells.

    No source-target lineage correspondence is introduced by this loader. Dense
    torch tensors and sparse CSR target populations are both supported.
    """

    def __init__(
        self,
        source: KineticPopulationLike,
        future_target_total: torch.Tensor | sparse.csr_matrix | TotalPopulation,
        delta_time: float,
        batch_size: int,
        steps_per_epoch: int,
        past_target_total: torch.Tensor | sparse.csr_matrix | TotalPopulation | None = None,
        past_source: KineticPopulationLike | None = None,
        past_delta_time: float | None = None,
        seed: int = 0,
    ):
        self.source = source
        self.future_target = future_target_total if isinstance(future_target_total, TotalPopulation) else TotalPopulation(future_target_total)
        self.past_target = None if past_target_total is None else (
            past_target_total if isinstance(past_target_total, TotalPopulation) else TotalPopulation(past_target_total)
        )
        self.past_source = past_source
        self.past_delta_time = past_delta_time
        if self.future_target.n_genes != source.n_genes:
            raise ValueError("future target gene count must match the source population.")
        if self.past_target is not None and self.past_target.n_genes != source.n_genes:
            raise ValueError("past target gene count must match the source population.")
        if self.past_source is not None and self.past_source.n_genes != source.n_genes:
            raise ValueError("past source gene count must match the source population.")
        if self.past_source is not None and (self.past_delta_time is None or self.past_delta_time <= 0):
            raise ValueError("past_delta_time must be positive when past_source is provided.")
        self.delta_time = delta_time
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    @staticmethod
    def _sample_indices(n: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        if n >= batch_size:
            return torch.randperm(n, generator=generator)[:batch_size]
        return torch.randint(n, (batch_size,), generator=generator)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.steps_per_epoch):
            source_indices = self._sample_indices(len(self.source), self.batch_size, generator)
            future_indices = self._sample_indices(len(self.future_target), self.batch_size, generator)
            past = None
            if self.past_target is not None:
                past_indices = self._sample_indices(len(self.past_target), self.batch_size, generator)
                past = self.past_target.take(past_indices)
            past_source_batch = None
            if self.past_source is not None:
                past_source_indices = self._sample_indices(len(self.past_source), self.batch_size, generator)
                past_source_batch = self.past_source.take(past_source_indices)
            local = self.source.take(source_indices)
            yield GlobalBatch(
                total=local.total,
                new=local.new,
                old=local.old,
                ntr=local.ntr,
                target_total=self.future_target.take(future_indices),
                past_target_total=past,
                past_source=past_source_batch,
                past_delta_time=self.past_delta_time,
                delta_time=self.delta_time,
                labeling_time=local.labeling_time,
            )


class MixedBatchLoader:
    """Mix multiple local or global loaders without concatenating their populations."""

    def __init__(
        self,
        loaders: tuple[object, ...],
        steps_per_epoch: int,
        weights: tuple[float, ...] | None = None,
        seed: int = 0,
    ):
        if not loaders:
            raise ValueError("At least one loader is required.")
        if weights is None:
            weights = tuple(1.0 for _ in loaders)
        if len(weights) != len(loaders) or any(weight <= 0 for weight in weights):
            raise ValueError("weights must be positive and match the number of loaders.")
        self.loaders, self.steps_per_epoch, self.weights, self.seed = loaders, steps_per_epoch, weights, seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        probabilities = torch.tensor(self.weights, dtype=torch.float32)
        probabilities = probabilities / probabilities.sum()
        iterators = [iter(loader) for loader in self.loaders]
        for _ in range(self.steps_per_epoch):
            index = int(torch.multinomial(probabilities, 1, generator=generator))
            try:
                yield next(iterators[index])
            except StopIteration:
                iterators[index] = iter(self.loaders[index])
                yield next(iterators[index])
