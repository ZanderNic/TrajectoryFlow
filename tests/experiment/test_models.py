# std-lib imports

# 3 party imports
import pytest
import torch

# package imports
from trajectoryflow.experiment.config import ModelConfig
from trajectoryflow.experiment.models import NoChangeExperimentAdapter


def test_no_change_rejects_unused_trainer_params():
    config = ModelConfig(name="no_change", trainer_params={"epochs": 10})
    with pytest.raises(ValueError, match="does not use trainer_params"):
        NoChangeExperimentAdapter().build(config, data=None, device=torch.device("cpu"), seed=0)


def test_no_change_supports_backcast_prediction():
    from trajectoryflow.models.baselines.no_change import NoChangeBaseline

    source = torch.ones((2, 3))
    prediction = NoChangeBaseline().predict(source, source_time=10.0, target_time=5.0, n_samples=2)
    assert prediction.states.shape == (2, 2, 3)
    assert prediction.target_time == 5.0
