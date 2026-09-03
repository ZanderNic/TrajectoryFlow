# std-lib imports

# 3 party imports
import torch

# package imports


def _validate(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if predicted.ndim != 2:
        raise ValueError("predicted must have shape [n_cells, n_features].")

    if target.ndim != 2:
        raise ValueError("target must have shape [n_cells, n_features].")

    if predicted.shape[0] == 0:
        raise ValueError("predicted must contain at least one cell.")

    if target.shape[0] == 0:
        raise ValueError("target must contain at least one cell.")

    if predicted.shape[1] != target.shape[1]:
        raise ValueError("Predicted and target must have the same number of features.")

    if not torch.isfinite(predicted).all():
        raise ValueError("predicted contains non-finite values.")

    if not torch.isfinite(target).all():
        raise ValueError("target contains non-finite values.")


def _subsample(
    x: torch.Tensor,
    max_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if len(x) <= max_samples:
        return x

    indices = torch.randperm(len(x), generator=generator)[:max_samples]
    return x[indices.to(x.device)]


# ------------------------------------------------------------------
# First moments
# ------------------------------------------------------------------


def centroid_distance(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Euclidean distance between population means."""

    _validate(predicted, target)

    predicted_mean = predicted.mean(dim=0)
    target_mean = target.mean(dim=0)

    return torch.linalg.vector_norm(predicted_mean - target_mean).item()


def mean_absolute_error(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute difference between feature means."""

    _validate(predicted, target)

    predicted_mean = predicted.mean(dim=0)
    target_mean = target.mean(dim=0)

    return torch.mean(torch.abs(predicted_mean - target_mean)).item()


# ------------------------------------------------------------------
# Second moments
# ------------------------------------------------------------------


def variance_absolute_error(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute difference between feature variances."""

    _validate(predicted, target)

    predicted_var = predicted.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)

    return torch.mean(torch.abs(predicted_var - target_var)).item()


def dispersion_error(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """
        Relative difference in mean distance from the population centroid.

        Useful for detecting distribution collapse or excessive spread.
    """

    _validate(predicted, target)

    predicted_center = predicted.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)

    predicted_dispersion = torch.linalg.vector_norm(predicted - predicted_center, dim=1).mean()
    target_dispersion = torch.linalg.vector_norm(target - target_center, dim=1).mean()

    return (
        torch.abs(predicted_dispersion - target_dispersion)
        / target_dispersion.clamp_min(1e-8)
    ).item()


def covariance_error(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """
        Relative Frobenius distance between covariance matrices.

        Best suited for lower-dimensional representations such as PCA
        or learned latent spaces.
    """

    _validate(predicted, target)

    predicted_centered = predicted - predicted.mean(dim=0)
    target_centered = target - target.mean(dim=0)

    predicted_covariance = (
        predicted_centered.T @ predicted_centered
    ) / max(len(predicted) - 1, 1)

    target_covariance = (
        target_centered.T @ target_centered
    ) / max(len(target) - 1, 1)

    difference = torch.linalg.matrix_norm(predicted_covariance - target_covariance)
    target_scale = torch.linalg.matrix_norm(target_covariance).clamp_min(1e-8)

    return (difference / target_scale).item()


# ------------------------------------------------------------------
# Correlation
# ------------------------------------------------------------------


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()

    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)

    if denominator <= 1e-12:
        return float("nan")

    return (torch.dot(x, y) / denominator).item()


def mean_correlation(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson correlation between population feature means."""

    _validate(predicted, target)

    return _pearson(predicted.mean(dim=0), target.mean(dim=0))


def variance_correlation(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson correlation between population feature variances."""

    _validate(predicted, target)

    predicted_var = predicted.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)

    return _pearson(predicted_var, target_var)


# ------------------------------------------------------------------
# Distribution metrics
# ------------------------------------------------------------------


def mmd_rbf(
    predicted: torch.Tensor,
    target: torch.Tensor,
    max_samples: int = 1024,
    seed: int = 0,
) -> float:
    """Squared Maximum Mean Discrepancy using an RBF kernel."""

    _validate(predicted, target)

    generator = torch.Generator()
    generator.manual_seed(seed)

    predicted = _subsample(predicted, max_samples, generator)
    target = _subsample(target, max_samples, generator)

    combined = torch.cat([predicted, target], dim=0)
    distances = torch.cdist(combined, combined).pow(2)

    positive_distances = distances[distances > 0]

    if len(positive_distances) == 0:
        return 0.0

    bandwidth = torch.median(positive_distances).clamp_min(1e-8)
    gamma = 1.0 / (2.0 * bandwidth)

    distance_xx = torch.cdist(predicted, predicted).pow(2)
    distance_yy = torch.cdist(target, target).pow(2)
    distance_xy = torch.cdist(predicted, target).pow(2)

    kernel_xx = torch.exp(-gamma * distance_xx)
    kernel_yy = torch.exp(-gamma * distance_yy)
    kernel_xy = torch.exp(-gamma * distance_xy)

    mmd = kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()

    return max(mmd.item(), 0.0)


def sliced_wasserstein(
    predicted: torch.Tensor,
    target: torch.Tensor,
    n_projections: int = 100,
    max_samples: int = 2048,
    seed: int = 0,
) -> float:
    """
    Approximate Wasserstein distance using random one-dimensional projections.

    Works with different numbers of predicted and target cells.
    """

    _validate(predicted, target)

    generator = torch.Generator()
    generator.manual_seed(seed)

    n_samples = min(len(predicted), len(target), max_samples)

    predicted = _subsample(predicted, n_samples, generator)
    target = _subsample(target, n_samples, generator)

    n_features = predicted.shape[1]

    projection_generator = torch.Generator(device=predicted.device)
    projection_generator.manual_seed(seed)

    projections = torch.randn(
        n_projections,
        n_features,
        device=predicted.device,
        dtype=predicted.dtype,
        generator=projection_generator,
    )

    projection_norms = torch.linalg.vector_norm(
        projections, dim=1, keepdim=True
    ).clamp_min(1e-8)

    projections = projections / projection_norms

    predicted_projection = predicted @ projections.T
    target_projection = target @ projections.T

    predicted_sorted = torch.sort(predicted_projection, dim=0).values
    target_sorted = torch.sort(target_projection, dim=0).values

    distance = torch.sqrt(
        torch.mean((predicted_sorted - target_sorted) ** 2)
    )

    return distance.item()


def chamfer_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    max_samples: int = 2048,
    seed: int = 0,
) -> float:
    """
    Symmetric nearest-neighbor distance.

    Penalizes predicted cells far from the real population as well as
    regions of the real population not covered by predictions.
    """

    _validate(predicted, target)

    generator = torch.Generator()
    generator.manual_seed(seed)

    predicted = _subsample(predicted, max_samples, generator)
    target = _subsample(target, max_samples, generator)

    distances = torch.cdist(predicted, target)

    predicted_to_target = distances.min(dim=1).values.mean()
    target_to_predicted = distances.min(dim=0).values.mean()

    return (0.5 * (predicted_to_target + target_to_predicted)).item()