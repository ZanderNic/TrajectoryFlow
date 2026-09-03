# std-lib imports
from dataclasses import dataclass
from typing import Callable

# 3 party imports
import numpy as np
import torch

# package imports
from trajectoryflow.evaluation.result import EvaluationReport, MetricResult, MetricRange
from trajectoryflow.models.base import TrajectoryPrediction


MetricFunction = Callable[[torch.Tensor, torch.Tensor], float]


@dataclass
class Metric:
    name: str
    function: MetricFunction
    higher_is_better: bool
    value_range: MetricRange
    
    
    
class Evaluator:

    def __init__(self):
        self.metrics: dict[str, Metric] = {}

    # ------------------------------------------------------------------
    # Metric management
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        function: MetricFunction,
        higher_is_better: bool,
        value_range: MetricRange,
    ) -> None:
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' is already registered.")

        self.metrics[name] = Metric(
            name=name,
            function=function,
            higher_is_better=higher_is_better,
            value_range=value_range,
        )


    def available_metrics(self) -> list[str]:
        return list(self.metrics.keys())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        prediction: TrajectoryPrediction,
        target: torch.Tensor,
        metrics: list[str] | None = None,
    ) -> EvaluationReport:
        if target.ndim != 2:
            raise ValueError("target must have shape [n_cells, n_features].")

        if prediction.states.ndim != 3:
            raise ValueError(
                "prediction.states must have shape "
                "[n_samples, n_cells, n_features]."
            )

        if prediction.states.shape[-1] != target.shape[-1]:
            raise ValueError(
                "Prediction and target must have the same number of features."
            )

        metric_names = self.available_metrics() if metrics is None else metrics

        results = {}

        for metric_name in metric_names:
            if metric_name not in self.metrics:
                raise KeyError(
                    f"Unknown metric '{metric_name}'. "
                    f"Available: {self.available_metrics()}"
                )

            metric = self.metrics[metric_name]

            values = [
                float(metric.function(sample, target))
                for sample in prediction.states
            ]

            finite_values = [value for value in values if np.isfinite(value)]

            if finite_values:
                mean = float(np.mean(finite_values))
                std = float(np.std(finite_values))
            else:
                mean = float("nan")
                std = float("nan")

            results[metric_name] = MetricResult(
                name=metric_name,
                mean=mean,
                std=std,
                values=values,
                higher_is_better=metric.higher_is_better,
                value_range=metric.value_range,
            )

        return EvaluationReport(
            model_name=prediction.metadata.get("model", "unknown"),
            source_time=prediction.source_time,
            target_time=prediction.target_time,
            metrics=results,
        )