# std-lib imports
from pathlib import Path

# 3 party imports
import pandas as pd
import torch

# package imports
from trajectoryflow.experiment.config import (
    BenchmarkConfig,
    EvaluationConfig,
    EvaluationSpaceConfig,
    ModelConfig,
    RuntimeConfig,
)
from trajectoryflow.experiment.evaluation import CallableEvaluationAdapter
from trajectoryflow.experiment.models import DualScaleExperimentAdapter, ExperimentRegistry
from trajectoryflow.experiment.runner import ExperimentRunner
from trajectoryflow.experiment.split import ForecastTask, SplitSpec, resolve_split


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="dual_scale",
        model_params={
            "latent_dim": 3,
            "kinetic_dim": 2,
            "state_hidden_dims": (8,),
            "decoder_hidden_dims": (8,),
            "kinetic_projection_dim": 3,
            "kinetic_hidden_dims": (6,),
            "transition_hidden_dims": (6,),
            "alignment_hidden_dims": (4,),
        },
        trainer_params={
            "local_batch_size": 4,
            "global_batch_size": 4,
            "global_samples": 1,
            "sw_projections": 2,
            "mmd_bandwidths": (1.0,),
            "schedule": {
                "local_pretrain_epochs": 1,
                "global_pretrain_epochs": 0,
                "joint_epochs": 1,
                "local_steps_per_pretrain_epoch": 1,
                "global_steps_per_pretrain_epoch": 0,
                "local_steps_per_joint_cycle": 1,
                "global_steps_per_joint_cycle": 1,
                "joint_cycles_per_epoch": 1,
            },
        },
    )


def _mean_evaluator(prediction, target):
    return pd.DataFrame(
        [
            {
                "metric": "mean",
                "value": float(prediction.states.mean()),
                "std": 0.0,
                "higher_is_better": False,
            }
        ]
    )


def test_dual_scale_adapter_uses_only_fit_timepoints_for_global_training(fake_store):
    fake_store.__init__(timepoints=("0h", "2h", "4h", "6h", "8h"), n_cells=12, n_genes=4)
    spec = SplitSpec(
        name="extrapolation",
        fit_timepoints=("0h", "2h", "4h", "6h"),
        test_tasks=(ForecastTask("6h", "8h"),),
    )
    split = resolve_split(fake_store, spec, seed=0)
    adapter = DualScaleExperimentAdapter()
    from trajectoryflow.experiment.data import ExperimentData

    built = adapter.build(_model_config(), ExperimentData(fake_store, split), torch.device("cpu"), seed=0)

    assert built.metadata["training_transitions"] == (("0h", "2h"), ("2h", "4h"), ("4h", "6h"))
    assert built.metadata["past_training_constraints"] == (("0h", "2h"), ("2h", "4h"))
    assert all("8h" not in pair for pair in built.metadata["training_transitions"])


def test_dual_scale_adapter_uses_previous_snapshot_when_labeling_time_does_not_match_grid(fake_store):
    fake_store.__init__(timepoints=("0h", "5h", "10h", "15h"), n_cells=12, n_genes=4)
    spec = SplitSpec(
        name="extrapolation",
        fit_timepoints=("0h", "5h", "10h"),
        test_tasks=(ForecastTask("10h", "15h"),),
    )
    split = resolve_split(fake_store, spec, seed=0)
    adapter = DualScaleExperimentAdapter()
    from trajectoryflow.experiment.data import ExperimentData

    built = adapter.build(_model_config(), ExperimentData(fake_store, split), torch.device("cpu"), seed=0)

    assert built.metadata["training_transitions"] == (("0h", "5h"), ("5h", "10h"))
    assert built.metadata["past_training_constraints"] == (("0h", "5h"),)
    loaders = built.trainer.global_loader.loaders
    assert loaders[1].past_delta_time == 5.0
    assert loaders[1].past_source is not None


def test_dual_scale_adapter_runs_end_to_end_forecast(tmp_path: Path, fake_store):
    fake_store.__init__(timepoints=("0h", "2h", "4h", "6h", "8h"), n_cells=12, n_genes=4)
    spec = SplitSpec(
        name="extrapolation",
        fit_timepoints=("0h", "2h", "4h", "6h"),
        test_tasks=(ForecastTask("6h", "8h"),),
    )
    split = resolve_split(fake_store, spec, seed=0)
    model = _model_config()
    config = BenchmarkConfig(
        data_root=tmp_path / "data",
        output_dir=tmp_path / "runs",
        seeds=(0,),
        models=(model,),
        splits=(spec,),
        evaluation=EvaluationConfig(
            n_source_cells=6,
            n_target_cells=6,
            n_samples=2,
            sample_batch_size=1,
            prediction_warmup_runs=0,
            prediction_timing_runs=1,
            space=EvaluationSpaceConfig(transform="none"),
        ),
        runtime=RuntimeConfig(device="cpu"),
        save_checkpoints=False,
        save_predictions=False,
    )
    registry = ExperimentRegistry()
    registry.register_model("dual_scale", DualScaleExperimentAdapter())
    registry.register_evaluator(CallableEvaluationAdapter(_mean_evaluator))

    result = ExperimentRunner(config, registry, fake_store).run(model, split, seed=0)

    assert result.status == "completed"
    assert result.training_summary["local_steps"] == 2
    assert result.training_summary["global_steps"] == 1
    assert result.tasks[0].source == "6h"
    assert result.tasks[0].target == "8h"
    assert result.tasks[0].n_samples == 2
