# std-lib imports
from pathlib import Path

# 3 party imports
import matplotlib.pyplot as plt
import numpy as np

# package imports


def _normalized(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _sample_indices(n: int, maximum: int, seed: int) -> np.ndarray:
    if n <= maximum:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=maximum, replace=False))


def _scatter(
    ax,
    positions: np.ndarray,
    labels: np.ndarray | None,
    maximum: int,
    seed: int,
) -> None:
    indices = _sample_indices(len(positions), maximum, seed)
    points = positions[indices]

    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], s=7, alpha=0.45)
        return

    labels = np.asarray(labels).astype(str)[indices]

    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=7,
            alpha=0.45,
            label=label,
        )


def _quiver(
    ax,
    positions: np.ndarray,
    vectors: np.ndarray,
    maximum: int,
    seed: int,
) -> None:
    valid = (
        np.isfinite(positions[:, :2]).all(axis=1)
        & np.isfinite(vectors[:, :2]).all(axis=1)
        & (np.linalg.norm(vectors[:, :2], axis=1) > 1e-12)
    )
    candidates = np.flatnonzero(valid)

    if len(candidates) > maximum:
        rng = np.random.default_rng(seed)
        candidates = np.sort(
            rng.choice(candidates, size=maximum, replace=False)
        )

    if not len(candidates):
        return

    directions = _normalized(vectors[candidates, :2])
    ax.quiver(
        positions[candidates, 0],
        positions[candidates, 1],
        directions[:, 0],
        directions[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0025,
        alpha=0.8,
    )


def plot_velocity_comparison(
    positions: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    path: str | Path,
    labels: np.ndarray | None = None,
    max_cells: int = 4000,
    max_arrows: int = 300,
    seed: int = 0,
    title: str | None = None,
) -> Path:
    positions = np.asarray(positions, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("positions must have at least two dimensions.")
    if predicted.shape[0] != len(positions):
        raise ValueError("predicted velocity rows must match positions.")
    if reference.shape[0] != len(positions):
        raise ValueError("reference velocity rows must match positions.")
    if predicted.shape[1] < 2 or reference.shape[1] < 2:
        raise ValueError("velocity vectors need at least two PCA components.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for index, (ax, vectors, panel_title) in enumerate(
        (
            (axes[0], predicted, "Predicted velocity"),
            (axes[1], reference, "Reference direction"),
        )
    ):
        _scatter(
            ax,
            positions=positions,
            labels=labels,
            maximum=max_cells,
            seed=seed + index,
        )
        _quiver(
            ax,
            positions=positions,
            vectors=vectors,
            maximum=max_arrows,
            seed=seed + 100 + index,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(panel_title)

    if labels is not None:
        handles, names = axes[1].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                names,
                loc="center right",
                bbox_to_anchor=(1.02, 0.5),
            )
            figure.subplots_adjust(right=0.84)

    if title:
        figure.suptitle(title)

    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_velocity_alignment_histogram(
    cosine_similarity: np.ndarray,
    path: str | Path,
    title: str | None = None,
) -> Path:
    values = np.asarray(cosine_similarity, dtype=np.float64)
    values = values[np.isfinite(values)]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=40)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Cells")
    ax.set_title(title or "Local velocity alignment")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
