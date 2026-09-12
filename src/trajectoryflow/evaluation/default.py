# std-lib imports

# 3 party imports

# package imports
from trajectoryflow.evaluation.evaluator import Evaluator
from trajectoryflow.evaluation.metrics.population import chamfer_distance, centroid_distance, dispersion_error, mean_absolute_error, mean_correlation, mmd_rbf, sliced_wasserstein, variance_absolute_error, variance_correlation
from trajectoryflow.evaluation.result import MetricRange


_DEFAULT_METRICS = (
    ("sliced_wasserstein", sliced_wasserstein, False, MetricRange(0, None, upper_inclusive=False)),
    ("mmd", mmd_rbf, False, MetricRange(0, 2)),
    ("chamfer", chamfer_distance, False, MetricRange(0, None, upper_inclusive=False)),
    ("centroid_distance", centroid_distance, False, MetricRange(0, None, upper_inclusive=False)),
    ("mean_mae", mean_absolute_error, False, MetricRange(0, None, upper_inclusive=False)),
    ("variance_mae", variance_absolute_error, False, MetricRange(0, None, upper_inclusive=False)),
    ("mean_correlation", mean_correlation, True, MetricRange(-1, 1)),
    ("variance_correlation", variance_correlation, True, MetricRange(-1, 1)),
    ("dispersion_error", dispersion_error, False, MetricRange(0, None, upper_inclusive=False)),
)


def make_default_evaluator() -> Evaluator:
    evaluator = Evaluator()
    for args in _DEFAULT_METRICS:
        evaluator.register(*args)
    return evaluator
