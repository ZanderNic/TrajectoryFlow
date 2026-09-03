# std-lib imports

# 3 party imports

# package imports
from trajectoryflow.evaluation.evaluator import Evaluator
from trajectoryflow.evaluation.result import MetricRange
from trajectoryflow.evaluation.metrics.population import (
    centroid_distance,
    chamfer_distance,
    dispersion_error,
    mean_absolute_error,
    mean_correlation,
    mmd_rbf,
    sliced_wasserstein,
    variance_absolute_error,
    variance_correlation,
)


def make_default_evaluator() -> Evaluator:
    evaluator = Evaluator()

    evaluator.register(
        name="sliced_wasserstein",
        function=sliced_wasserstein,
        higher_is_better=False,
        value_range=MetricRange(0, None, upper_inclusive=False),
    )

    evaluator.register(
        name="mmd",
        function=mmd_rbf,
        higher_is_better=False,
        value_range=MetricRange(0, 2),
    )

    evaluator.register(
        name="chamfer",
        function=chamfer_distance,
        higher_is_better=False,
        value_range=MetricRange(0, None, upper_inclusive=False),
    )

    evaluator.register(
        name="centroid_distance",
        function=centroid_distance,
        higher_is_better=False,
        value_range=MetricRange(0, None, upper_inclusive=False),
    )

    evaluator.register(
        name="mean_mae",
        function=mean_absolute_error,
        higher_is_better=False,
        value_range=MetricRange(0, None, upper_inclusive=False),
    )

    evaluator.register(
        name="variance_mae",
        function=variance_absolute_error,
        higher_is_better=False,
        value_range=MetricRange(0, None, upper_inclusive=False),
    )

    evaluator.register(
        name="mean_correlation",
        function=mean_correlation,
        higher_is_better=True,
        value_range=MetricRange(-1, 1),
    )

    evaluator.register(
        name="variance_correlation",
        function=variance_correlation,
        higher_is_better=True,
        value_range=MetricRange(-1, 1),
    )

    evaluator.register(
        name="dispersion_error",
        function=dispersion_error,
        higher_is_better=False,
        value_range=MetricRange(0, None, upper_inclusive=False),
    )

    return evaluator