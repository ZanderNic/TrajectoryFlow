# std-lib imports
from collections.abc import Sequence
from dataclasses import dataclass

# 3 party imports
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

# package imports
from trajectoryflow.data.store import ScifateStore
from trajectoryflow.models.baselines.velvet.neighborhood import build_knn_indices


@dataclass
class VelvetData:
    total: sparse.csr_matrix
    new: sparse.csr_matrix
    obs: pd.DataFrame | None
    timepoints: Sequence[str]

    def __post_init__(self) -> None:
        self.total, self.new = self.total.tocsr(), self.new.tocsr()
        if self.total.shape != self.new.shape:
            raise ValueError(f"total/new shape mismatch: {self.total.shape} vs {self.new.shape}.")
        if self.obs is not None and self.total.shape[0] != len(self.obs):
            raise ValueError("obs row count must match total/new cells.")
        if 0 in self.total.shape:
            raise ValueError("VelvetData must contain at least one cell and one gene.")
        if isinstance(self.timepoints, str) or not self.timepoints:
            raise ValueError("VelvetData requires a non-string sequence of timepoints.")
        if len(set(map(str, self.timepoints))) != len(self.timepoints):
            raise ValueError("VelvetData timepoints must be unique.")
        validate_velvet_matrix(self.total, "total RNA")
        validate_velvet_matrix(self.new, "new RNA")

    def __len__(self) -> int:
        return self.total.shape[0]

    @property
    def n_cells(self) -> int:
        return self.total.shape[0]

    @property
    def n_genes(self) -> int:
        return self.total.shape[1]


def validate_velvet_matrix(matrix: sparse.csr_matrix, name: str) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"{name} contains {int((~np.isfinite(matrix.data)).sum())} non-finite stored values.")
    if (matrix.data < 0).any():
        raise ValueError(f"{name} contains {int((matrix.data < 0).sum())} negative values (minimum={float(matrix.data.min()):g}); Velvet requires non-negative RNA values.")


def load_velvet_data(store: ScifateStore, timepoints: Sequence[str] | None = None) -> VelvetData:
    selected = list(store.timepoints if timepoints is None else timepoints)
    if not selected:
        raise ValueError("At least one timepoint is required.")
    snapshots = [store.load(timepoint) for timepoint in selected]
    if any(not hasattr(snapshot, "new") for snapshot in snapshots):
        raise AttributeError("TimepointData must expose `.new`; re-run preprocessing with new.npz enabled.")
    obs_parts = []
    for snapshot in snapshots:
        obs = snapshot.obs.copy()
        obs["timepoint"], obs["time_hours"] = snapshot.timepoint, snapshot.time_hours
        obs_parts.append(obs)
    return VelvetData(total=sparse.vstack([snapshot.expression for snapshot in snapshots], format="csr").astype(np.float32), new=sparse.vstack([snapshot.new for snapshot in snapshots], format="csr").astype(np.float32), obs=pd.concat(obs_parts, ignore_index=True), timepoints=selected)


def normalized_svd_embedding(total: sparse.csr_matrix, n_components: int = 50, target_sum: float = 10_000.0, seed: int = 0) -> np.ndarray:
    """Sparse library-log normalized SVD embedding used only for fixed KNN construction."""
    if n_components < 1 or target_sum <= 0:
        raise ValueError("n_components and target_sum must be > 0.")
    validate_velvet_matrix(total.tocsr(), "total RNA")
    x = total.astype(np.float32, copy=True)
    library = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(target_sum, library, out=np.zeros_like(library, dtype=np.float32), where=library > 0)
    x = (sparse.diags(scale) @ x).tocsr()
    x.data = np.log1p(x.data)
    n_components = min(n_components, x.shape[0] - 1, x.shape[1] - 1)
    if n_components < 1:
        raise ValueError("Not enough cells/features for an SVD embedding.")
    return TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(x).astype(np.float32)


def build_velvet_neighbors(data: VelvetData, n_neighbors: int = 100, n_components: int = 50, seed: int = 0) -> np.ndarray:
    return build_knn_indices(normalized_svd_embedding(data.total, n_components=n_components, seed=seed), n_neighbors=n_neighbors)
