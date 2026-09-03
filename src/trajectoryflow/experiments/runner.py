# std-lib imports
from dataclasses import dataclass
from pathlib import Path

# 3 party imports
import torch

# package imports
from trajectoryflow.evaluation.evaluator import Evaluator
from trajectoryflow.evaluation.result import EvaluationReport
from trajectoryflow.models.base import BaseTrajectoryModel, TrajectoryPrediction
from trajectoryflow.training.base import BaseTrainer


@dataclass
class ExperimentResult:
    prediction: TrajectoryPrediction
    report: EvaluationReport


class ExperimentRunner:

    def __init__(
        self,
        model: BaseTrajectoryModel,
        evaluator: Evaluator,
        trainer: BaseTrainer | None = None,
    ):
        self.model = model
        self.evaluator = evaluator
        self.trainer = trainer


    def run(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_time: float,
        target_time: float,
        n_samples: int = 1,
        metrics: list[str] | None = None,
    ) -> ExperimentResult:

        if self.trainer is not None:
            self.trainer.fit(self.model)

        prediction = self.model.predict(
            source=source,
            source_time=source_time,
            target_time=target_time,
            n_samples=n_samples,
        )

        report = self.evaluator.evaluate(
            prediction=prediction,
            target=target,
            metrics=metrics,
        )

        return ExperimentResult(
            prediction=prediction,
            report=report,
        )