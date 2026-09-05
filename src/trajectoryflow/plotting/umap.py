# std-lib imports

# 3 party imports
import matplotlib.pyplot as plt
import numpy as np

# package imports


def _validate_coordinates(coordinates: np.ndarray) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [n_cells, 2].")


def plot_umap(
    coordinates: np.ndarray,
    labels: np.ndarray | None = None,
    title: str | None = None,
    alpha: float = 0.8,
    size: float = 8,
):
    _validate_coordinates(coordinates)

    fig, ax = plt.subplots(figsize=(7, 6))

    if labels is None:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=size, alpha=alpha)
    else:
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=size,
                alpha=alpha,
                label=str(label),
            )

        ax.legend(frameon=False, markerscale=2)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_grouped_umap(
    coordinates: np.ndarray,
    groups: np.ndarray,
    title: str | None = None,
    alpha: float = 0.7,
    size: float = 8,
):
    return plot_umap(
        coordinates=coordinates,
        labels=groups,
        title=title,
        alpha=alpha,
        size=size,
    )


def plot_prediction_comparison(
    source_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    prediction_coordinates: np.ndarray,
    source_title: str = "Source",
    target_title: str = "Target",
    prediction_title: str = "Prediction",
    alpha: float = 0.8,
    size: float = 8,
):
    _validate_coordinates(source_coordinates)
    _validate_coordinates(target_coordinates)
    _validate_coordinates(prediction_coordinates)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    axes[0].scatter(source_coordinates[:, 0], source_coordinates[:, 1], s=size, alpha=alpha)
    axes[0].set_title(source_title)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    axes[1].scatter(target_coordinates[:, 0], target_coordinates[:, 1], s=size, alpha=alpha)
    axes[1].set_title(target_title)
    axes[1].set_xlabel("UMAP 1")

    axes[2].scatter(prediction_coordinates[:, 0], prediction_coordinates[:, 1], s=size, alpha=alpha)
    axes[2].set_title(prediction_title)
    axes[2].set_xlabel("UMAP 1")

    return fig, axes


def plot_joint_prediction_overlay(
    target_coordinates: np.ndarray,
    prediction_coordinates: np.ndarray,
    alpha: float = 0.7,
    size: float = 8,
    title: str = "Prediction vs Target",
):
    _validate_coordinates(target_coordinates)
    _validate_coordinates(prediction_coordinates)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        target_coordinates[:, 0],
        target_coordinates[:, 1],
        s=size,
        alpha=alpha,
        label="target",
    )

    ax.scatter(
        prediction_coordinates[:, 0],
        prediction_coordinates[:, 1],
        s=size,
        alpha=alpha,
        label="prediction",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(frameon=False, markerscale=2)

    return fig, ax