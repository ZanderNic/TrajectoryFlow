# std-lib imports

# 3 party imports
import pytest
import torch

# package imports
from trajectoryflow.evaluation.metrics.population import (
    centroid_distance,
    chamfer_distance,
    covariance_error,
    dispersion_error,
    mean_absolute_error,
    mean_correlation,
    mmd_rbf,
    sliced_wasserstein,
    variance_absolute_error,
    variance_correlation,
)


# ---------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------


@pytest.fixture
def simple_population() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def shifted_population(
    simple_population,
) -> torch.Tensor:
    return simple_population + 1.0


# ---------------------------------------------------------------------
# Centroid distance
# ---------------------------------------------------------------------


def test_centroid_distance_identical(
    simple_population,
):
    result = centroid_distance(
        simple_population,
        simple_population,
    )

    assert result == pytest.approx(0.0)


def test_centroid_distance_known_shift(
    simple_population,
):
    shifted = simple_population + torch.tensor(
        [3.0, 4.0]
    )

    result = centroid_distance(
        simple_population,
        shifted,
    )

    # Euclidean norm of [3, 4]
    assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------
# Mean absolute error
# ---------------------------------------------------------------------


def test_mean_absolute_error_identical(
    simple_population,
):
    result = mean_absolute_error(
        simple_population,
        simple_population,
    )

    assert result == pytest.approx(0.0)


def test_mean_absolute_error_known_shift(
    simple_population,
):
    shifted = simple_population + 2.0

    result = mean_absolute_error(
        simple_population,
        shifted,
    )

    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------
# Variance absolute error
# ---------------------------------------------------------------------


def test_variance_absolute_error_identical(
    simple_population,
):
    result = variance_absolute_error(
        simple_population,
        simple_population,
    )

    assert result == pytest.approx(0.0)


def test_variance_absolute_error_detects_scale_change(
    simple_population,
):
    scaled = simple_population * 2.0

    result = variance_absolute_error(
        simple_population,
        scaled,
    )

    assert result > 0.0


def test_variance_unchanged_by_translation(
    simple_population,
):
    shifted = simple_population + 100.0

    result = variance_absolute_error(
        simple_population,
        shifted,
    )

    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------


def test_dispersion_error_identical(
    simple_population,
):
    result = dispersion_error(
        simple_population,
        simple_population,
    )

    assert result == pytest.approx(0.0)


def test_dispersion_translation_invariant(
    simple_population,
):
    shifted = simple_population + 10.0

    result = dispersion_error(
        simple_population,
        shifted,
    )

    assert result == pytest.approx(0.0)


def test_dispersion_detects_spreading(
    simple_population,
):
    spread = simple_population * 4.0

    result = dispersion_error(
        spread,
        simple_population,
    )

    assert result > 0.0


# ---------------------------------------------------------------------
# Mean correlation
# ---------------------------------------------------------------------


def test_mean_correlation_identical():
    predicted = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ]
    )

    result = mean_correlation(
        predicted,
        predicted,
    )

    assert result == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_mean_correlation_perfect_negative():
    predicted = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )

    target = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ]
    )

    result = mean_correlation(
        predicted,
        target,
    )

    assert result == pytest.approx(
        -1.0,
        abs=1e-6,
    )


def test_mean_correlation_constant_vector_is_nan():
    predicted = torch.ones(3, 4)
    target = torch.ones(3, 4)

    result = mean_correlation(
        predicted,
        target,
    )

    assert result != result  # NaN


# ---------------------------------------------------------------------
# Variance correlation
# ---------------------------------------------------------------------


def test_variance_correlation_identical():
    population = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ]
    )

    result = variance_correlation(
        population,
        population,
    )

    assert result == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_variance_correlation_translation_invariant():
    population = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ]
    )

    shifted = population + 100.0

    result = variance_correlation(
        population,
        shifted,
    )

    assert result == pytest.approx(
        1.0,
        abs=1e-6,
    )


# ---------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------


def test_chamfer_identical(
    simple_population,
):
    result = chamfer_distance(
        simple_population,
        simple_population,
        max_samples=100,
    )

    assert result == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_chamfer_shifted_is_positive(
    simple_population,
    shifted_population,
):
    result = chamfer_distance(
        simple_population,
        shifted_population,
        max_samples=100,
    )

    assert result > 0.0


def test_chamfer_allows_different_population_sizes():
    predicted = torch.randn(10, 3)
    target = torch.randn(20, 3)

    result = chamfer_distance(
        predicted,
        target,
        max_samples=100,
    )

    assert result >= 0.0


# ---------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------


def test_mmd_identical(
    simple_population,
):
    result = mmd_rbf(
        simple_population,
        simple_population,
        max_samples=100,
    )

    assert result == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_mmd_shifted_is_positive():
    torch.manual_seed(1)

    predicted = torch.randn(100, 3)
    target = predicted + 5.0

    result = mmd_rbf(
        predicted,
        target,
        max_samples=100,
    )

    assert result > 0.0


def test_mmd_allows_different_population_sizes():
    predicted = torch.randn(30, 4)
    target = torch.randn(50, 4)

    result = mmd_rbf(
        predicted,
        target,
        max_samples=100,
    )

    assert result >= 0.0


# ---------------------------------------------------------------------
# Sliced Wasserstein
# ---------------------------------------------------------------------


def test_sliced_wasserstein_identical(
    simple_population,
):
    result = sliced_wasserstein(
        simple_population,
        simple_population,
        n_projections=50,
        max_samples=100,
        seed=1,
    )

    assert result == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_sliced_wasserstein_shifted_is_positive():
    torch.manual_seed(1)

    predicted = torch.randn(100, 4)
    target = predicted + 5.0

    result = sliced_wasserstein(
        predicted,
        target,
        n_projections=100,
        max_samples=100,
        seed=1,
    )

    assert result > 0.0


# ---------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------


def test_covariance_error_identical(
    simple_population,
):
    result = covariance_error(
        simple_population,
        simple_population,
    )

    assert result == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_covariance_translation_invariant(
    simple_population,
):
    shifted = simple_population + 50.0

    result = covariance_error(
        simple_population,
        shifted,
    )

    assert result == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_covariance_detects_scale_change(
    simple_population,
):
    scaled = simple_population * 3.0

    result = covariance_error(
        scaled,
        simple_population,
    )

    assert result > 0.0


# ---------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        centroid_distance,
        mean_absolute_error,
        variance_absolute_error,
        dispersion_error,
        mean_correlation,
        variance_correlation,
        sliced_wasserstein,
        mmd_rbf,
        chamfer_distance,
        covariance_error,
    ],
)
def test_metric_rejects_1d_input(metric):
    predicted = torch.tensor([1.0, 2.0])
    target = torch.randn(10, 2)

    with pytest.raises(ValueError):
        metric(predicted, target)


@pytest.mark.parametrize(
    "metric",
    [
        centroid_distance,
        mean_absolute_error,
        variance_absolute_error,
        dispersion_error,
        mean_correlation,
        variance_correlation,
        sliced_wasserstein,
        mmd_rbf,
        chamfer_distance,
        covariance_error,
    ],
)
def test_metric_rejects_feature_mismatch(metric):
    predicted = torch.randn(10, 3)
    target = torch.randn(10, 4)

    with pytest.raises(ValueError):
        metric(predicted, target)


@pytest.mark.parametrize(
    "metric",
    [
        centroid_distance,
        mean_absolute_error,
        variance_absolute_error,
        dispersion_error,
        mean_correlation,
        variance_correlation,
        sliced_wasserstein,
        mmd_rbf,
        chamfer_distance,
        covariance_error,
    ],
)
def test_metric_rejects_empty_population(metric):
    predicted = torch.empty(0, 3)
    target = torch.randn(10, 3)

    with pytest.raises(ValueError):
        metric(predicted, target)


@pytest.mark.parametrize(
    "metric",
    [
        centroid_distance,
        mean_absolute_error,
        variance_absolute_error,
        dispersion_error,
        mean_correlation,
        variance_correlation,
        sliced_wasserstein,
        mmd_rbf,
        chamfer_distance,
        covariance_error,
    ],
)
def test_metric_rejects_nan(metric):
    predicted = torch.randn(10, 3)
    predicted[0, 0] = float("nan")

    target = torch.randn(10, 3)

    with pytest.raises(ValueError):
        metric(predicted, target)
    
    
        
def test_sliced_wasserstein_is_reproducible():

    torch.manual_seed(10)

    predicted = torch.randn(100, 5)
    target = torch.randn(100, 5)

    first = sliced_wasserstein(
        predicted,
        target,
        seed=42,
    )

    second = sliced_wasserstein(
        predicted,
        target,
        seed=42,
    )

    assert first == pytest.approx(second)