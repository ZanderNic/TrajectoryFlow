# std-lib imports
from collections.abc import Callable
from dataclasses import dataclass

# 3 party imports
import numpy as np
import torch

# package imports
from trajectoryflow.evaluation.result import EvaluationReport, MetricRange, MetricResult
from trajectoryflow.models.base import TrajectoryPrediction


MetricFunction = Callable[[torch.Tensor, torch.Tensor], float]


@dataclass(frozen=True)
class Metric:
    name: str
    function: MetricFunction
    higher_is_better: bool
    value_range: MetricRange


class Evaluator:
    def __init__(self):
        self.metrics: dict[str, Metric] = {}

    def register(self, name: str, function: MetricFunction, higher_is_better: bool, value_range: MetricRange) -> None:
        if name in self.metrics:
            raise ValueError(f"Metric {name!r} is already registered.")
        self.metrics[name] = Metric(name, function, higher_is_better, value_range)

    def available_metrics(self) -> list[str]:
        return list(self.metrics)

    def evaluate(self, prediction: TrajectoryPrediction, target: torch.Tensor, metrics: list[str] | None = None) -> EvaluationReport:
        if target.ndim != 2 or 0 in target.shape:
            raise ValueError("target must have non-empty shape [n_cells, n_features].")
        if prediction.states.shape[-1] != target.shape[-1]:
            raise ValueError("Prediction and target must have the same number of features.")
        if not torch.isfinite(target).all():
            raise ValueError("target contains non-finite values.")

        names = self.available_metrics() if metrics is None else metrics
        unknown = [name for name in names if name not in self.metrics]
        if unknown:
            raise KeyError(f"Unknown metrics {unknown}. Available: {self.available_metrics()}.")

        results = {}
        for name in names:
            metric = self.metrics[name]
            values = [float(metric.function(sample, target)) for sample in prediction.states]
            finite = [value for value in values if np.isfinite(value)]
            mean = float(np.mean(finite)) if finite else float("nan")
            std = float(np.std(finite, ddof=1)) if len(finite) > 1 else (0.0 if finite else float("nan"))
            results[name] = MetricResult(name, mean, std, values, metric.higher_is_better, metric.value_range)
        return EvaluationReport(prediction.metadata.get("model", "unknown"), prediction.source_time, prediction.target_time, results)
