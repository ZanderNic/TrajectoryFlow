# std-lib imports
from dataclasses import dataclass
from collections.abc import Sequence

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
    obs: pd.DataFrame
    timepoints: list[str]

    def __len__(self) -> int:
        return self.total.shape[0]

    @property
    def n_cells(self) -> int:
        return self.total.shape[0]

    @property
    def n_genes(self) -> int:
        return self.total.shape[1]


def validate_velvet_matrix(matrix: sparse.csr_matrix, name: str) -> None:
    """Validate sparse Velvet inputs before any neural-network training."""
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")

    values = matrix.data

    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"{name} contains {n_bad} non-finite stored values.")

    if (values < 0).any():
        n_negative = int((values < 0).sum())
        minimum = float(values.min())
        raise ValueError(
            f"{name} contains {n_negative} negative values "
            f"(minimum={minimum:g}). Velvet's count likelihood requires "
            "non-negative RNA values."
        )




def load_velvet_data(
    store: ScifateStore,
    timepoints: Sequence[str] | None = None,
) -> VelvetData:
    """Stack selected TrajectoryFlow snapshots for Velvet training."""
    selected = store.timepoints if timepoints is None else list(timepoints)

    if not selected:
        raise ValueError("At least one timepoint is required.")

    snapshots = [store.load(timepoint) for timepoint in selected]

    for snapshot in snapshots:
        if not hasattr(snapshot, "new"):
            raise AttributeError(
                "TimepointData must expose `.new`. Re-run preprocessing with "
                "new.npz enabled and load it in ScifateStore."
            )

    total = sparse.vstack(
        [snapshot.expression for snapshot in snapshots],
        format="csr",
    ).astype(np.float32)

    new = sparse.vstack(
        [snapshot.new for snapshot in snapshots],
        format="csr",
    ).astype(np.float32)

    if total.shape != new.shape:
        raise ValueError(f"total/new shape mismatch: {total.shape} vs {new.shape}.")

    validate_velvet_matrix(total, "total RNA")
    validate_velvet_matrix(new, "new RNA")

    obs_parts = []

    for snapshot in snapshots:
        obs = snapshot.obs.copy()
        obs["timepoint"] = snapshot.timepoint
        obs["time_hours"] = snapshot.time_hours
        obs_parts.append(obs)

    obs = pd.concat(obs_parts, ignore_index=True)

    return VelvetData(
        total=total,
        new=new,
        obs=obs,
        timepoints=selected,
    )


def normalized_svd_embedding(
    total: sparse.csr_matrix,
    n_components: int = 50,
    target_sum: float = 10_000.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Produce a sparse log-normalized SVD embedding for fixed KNN construction.

    This is a TrajectoryFlow preprocessing helper, not a learned model input.
    """
    x = total.astype(np.float32, copy=True)

    library = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(
        target_sum,
        library,
        out=np.zeros_like(library, dtype=np.float32),
        where=library > 0,
    )

    x = sparse.diags(scale) @ x
    x = x.tocsr()
    x.data = np.log1p(x.data)

    n_components = min(n_components, x.shape[0] - 1, x.shape[1] - 1)
    if n_components < 1:
        raise ValueError("Not enough cells/features for an SVD embedding.")

    model = TruncatedSVD(n_components=n_components, random_state=seed)
    return model.fit_transform(x).astype(np.float32)


def build_velvet_neighbors(
    data: VelvetData,
    n_neighbors: int = 100,
    n_components: int = 50,
    seed: int = 0,
) -> np.ndarray:
    embedding = normalized_svd_embedding(
        total=data.total,
        n_components=n_components,
        seed=seed,
    )

    return build_knn_indices(
        embedding=embedding,
        n_neighbors=n_neighbors,
    )
