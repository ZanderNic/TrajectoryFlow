# std-lib imports
import json
from dataclasses import replace

# 3 party imports
import pandas as pd
import pytest
import torch

# package imports
from trajectoryflow.experiment.config import BenchmarkConfig, EvaluationConfig, EvaluationSpaceConfig, ModelConfig, RuntimeConfig
from trajectoryflow.experiment.evaluation import CallableEvaluationAdapter
from trajectoryflow.experiment.models import BuiltExperimentModel, ExperimentModelAdapter, ExperimentRegistry
from trajectoryflow.experiment.runner import ExperimentRunner
from trajectoryflow.experiment.split import ForecastTask, SplitSpec, resolve_split
from trajectoryflow.models.base import TrajectoryPrediction


class StochasticAdapter(ExperimentModelAdapter):
    def build(self, config, data, device, seed):
        return BuiltExperimentModel(model=torch.nn.Identity().to(device), trainer=None, device=device)

    def fit(self, built, config):
        return {"trained": False}

    def predict(self, built, source, source_time, target_time, n_samples):
        noise = torch.rand((n_samples, *source.shape), device=source.device)
        return TrajectoryPrediction(source[None] + noise, source_time, target_time)


class FailingAdapter(StochasticAdapter):
    def build(self, config, data, device, seed):
        raise RuntimeError("intentional build failure")


def mean_evaluator(prediction, target):
    return pd.DataFrame([{"metric": "prediction_mean", "value": float(prediction.states.mean()), "std": 0.0, "higher_is_better": False}])


def make_config(tmp_path, split_spec, **evaluation_overrides):
    evaluation = EvaluationConfig(n_source_cells=None, n_target_cells=None, n_samples=3, space=EvaluationSpaceConfig(transform="none"), **evaluation_overrides)
    return BenchmarkConfig(data_root=tmp_path / "data", output_dir=tmp_path / "runs", seeds=(0,), models=(ModelConfig(name="stochastic"),), splits=(split_spec,), evaluation=evaluation, runtime=RuntimeConfig(device="cpu"), save_checkpoints=False, save_predictions=False)


def make_split(fake_store):
    spec = SplitSpec(name="forecast", fit_timepoints=("5h",), test_tasks=(ForecastTask("5h", "10h"),))
    return spec, resolve_split(fake_store, spec, seed=0)


def make_registry(adapter=None):
    registry = ExperimentRegistry()
    registry.register_model("stochastic", adapter or StochasticAdapter())
    registry.register_evaluator(CallableEvaluationAdapter(mean_evaluator))
    return registry


def test_stochastic_metrics_do_not_depend_on_profiling_runs(tmp_path, fake_store):
    split_spec, split = make_split(fake_store)
    first = make_config(tmp_path / "a", split_spec, prediction_warmup_runs=0, prediction_timing_runs=1)
    second = make_config(tmp_path / "b", split_spec, prediction_warmup_runs=4, prediction_timing_runs=3)
    first_result = ExperimentRunner(first, make_registry(), fake_store).run(first.models[0], split, seed=0)
    second_result = ExperimentRunner(second, make_registry(), fake_store).run(second.models[0], split, seed=0)
    assert first_result.tasks[0].metrics == second_result.tasks[0].metrics


def test_fail_fast_persists_failed_result_before_reraising(tmp_path, fake_store):
    split_spec, split = make_split(fake_store)
    config = make_config(tmp_path, split_spec)
    registry = make_registry(FailingAdapter())
    with pytest.raises(RuntimeError, match="intentional build failure"):
        ExperimentRunner(config, registry, fake_store).run(config.models[0], split, seed=0)
    path = config.output_dir / "stochastic" / split.name / "seed_0" / "result.json"
    saved = json.loads(path.read_text())
    assert saved["status"] == "failed"
    assert "intentional build failure" in saved["error"]


def test_checkpoint_and_prediction_flags_create_artifacts(tmp_path, fake_store):
    split_spec, split = make_split(fake_store)
    config = replace(make_config(tmp_path, split_spec), save_checkpoints=True, save_predictions=True)
    ExperimentRunner(config, make_registry(), fake_store).run(config.models[0], split, seed=0)
    root = config.output_dir / "stochastic" / split.name / "seed_0"
    assert (root / "checkpoint.pt").exists()
    assert (root / "predictions" / "test__5h_to_10h.npz").exists()
