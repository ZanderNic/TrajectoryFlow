# std-lib imports

# 3 party imports
import pytest
import torch

# package imports
from trajectoryflow.evaluation.evaluator import Evaluator
from trajectoryflow.evaluation.default import make_default_evaluator
from trajectoryflow.models.base import TrajectoryPrediction
from trajectoryflow.evaluation.result import MetricRange


def test_register_duplicate_metric_raises():

    evaluator = Evaluator()

    def metric(predicted, target):
        return 1.0

    evaluator.register(
        name="test",
        function=metric,
        higher_is_better=True,
        value_range=MetricRange(-1, 1),
    )

    with pytest.raises(ValueError):
        evaluator.register(
            name="test",
            function=metric,
            higher_is_better=True,
            value_range=MetricRange(-1, 1),
        )
        
        

def test_evaluate_single_prediction():

    evaluator = Evaluator()

    evaluator.register(
        name="dummy",
        function=lambda predicted, target: 2.0,
        higher_is_better=False,
        value_range=MetricRange(-1, 1),
    )

    prediction = TrajectoryPrediction(
        states=torch.randn(
            1,
            10,
            3,
        ),
        source_time=0,
        target_time=2,
        metadata={
            "model": "test_model"
        },
    )

    target = torch.randn(20, 3)

    report = evaluator.evaluate(
        prediction=prediction,
        target=target,
    )

    assert report.model_name == "test_model"
    assert report.source_time == 0
    assert report.target_time == 2

    assert "dummy" in report.metrics

    result = report.metrics["dummy"]

    assert result.mean == pytest.approx(2.0)
    assert result.std == pytest.approx(0.0)
    assert result.values == [2.0]
    
    
def test_unknown_metric_raises():

    evaluator = make_default_evaluator()

    prediction = TrajectoryPrediction(
        states=torch.randn(
            1,
            10,
            3,
        ),
        source_time=0,
        target_time=2,
    )

    target = torch.randn(10, 3)

    with pytest.raises(KeyError):
        evaluator.evaluate(
            prediction,
            target,
            metrics=[
                "does_not_exist"
            ],
        )
        

def test_evaluator_rejects_feature_mismatch():

    evaluator = make_default_evaluator()

    prediction = TrajectoryPrediction(
        states=torch.randn(
            1,
            10,
            3,
        ),
        source_time=0,
        target_time=2,
    )

    target = torch.randn(
        10,
        4,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            prediction,
            target,
        )