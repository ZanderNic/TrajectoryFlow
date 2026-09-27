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
from trajectoryflow.experiment.models import (
    DualScaleExperimentAdapter,
    ExperimentRegistry,
)
from trajectoryflow.experiment.runner import ExperimentRunner
from trajectoryflow.experiment.split import (
    ForecastTask,
    SplitSpec,
    resolve_split,
)


def _model_config(
    *,
    use_kinetic_encoder: bool = True,
    reconstruction_weight: float = 1.0,
    local_kinetic_weight: float = 1.0,
    population_weight: float = 1.0,
) -> ModelConfig:
    """
    Small dual-scale configuration for tests.

    The architecture and loss weights can be changed independently so the same
    helper can also be used for kinetic-branch ablation tests.
    """
    return ModelConfig(
        name="dual_scale",
        model_params={
            "latent_dim": 3,
            "kinetic_dim": 2,
            "use_kinetic_encoder": use_kinetic_encoder,
            "state_hidden_dims": (8,),
            "decoder_hidden_dims": (8,),
            "state_projection_dim": 3,
            "kinetic_hidden_dims": (6,),
            "transition_noise_dim": 2,
            "transition_hidden_dim": 6,
            "transition_blocks": 1,
            "time_scale_hours": 2.0,
        },
        trainer_params={
            "local_batch_size": 4,
            "global_batch_size": 4,
            "global_samples": 1,
            "sw_projections": 2,
            "loss_weights": {
                "reconstruction": reconstruction_weight,
                "local_kinetic": local_kinetic_weight,
                "population_sliced_wasserstein": population_weight,
            },
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


def test_dual_scale_adapter_uses_only_fit_timepoints_for_global_training(
    fake_store,
):
    fake_store.__init__(
        timepoints=("0h", "2h", "4h", "6h", "8h"),
        n_cells=12,
        n_genes=4,
    )

    spec = SplitSpec(
        name="extrapolation",
        fit_timepoints=("0h", "2h", "4h", "6h"),
        test_tasks=(ForecastTask("6h", "8h"),),
    )

    split = resolve_split(fake_store, spec, seed=0)
    adapter = DualScaleExperimentAdapter()

    from trajectoryflow.experiment.data import ExperimentData

    built = adapter.build(
        _model_config(),
        ExperimentData(fake_store, split),
        torch.device("cpu"),
        seed=0,
    )

    assert built.metadata["training_transitions"] == (
        ("0h", "2h"),
        ("2h", "4h"),
        ("4h", "6h"),
    )

    assert all(
        "8h" not in pair
        for pair in built.metadata["training_transitions"]
    )


def test_dual_scale_adapter_uses_observed_transition_intervals(
    fake_store,
):
    fake_store.__init__(
        timepoints=("0h", "5h", "10h", "15h"),
        n_cells=12,
        n_genes=4,
    )

    spec = SplitSpec(
        name="extrapolation",
        fit_timepoints=("0h", "5h", "10h"),
        test_tasks=(ForecastTask("10h", "15h"),),
    )

    split = resolve_split(fake_store, spec, seed=0)
    adapter = DualScaleExperimentAdapter()

    from trajectoryflow.experiment.data import ExperimentData

    built = adapter.build(
        _model_config(),
        ExperimentData(fake_store, split),
        torch.device("cpu"),
        seed=0,
    )

    assert built.metadata["training_transitions"] == (
        ("0h", "5h"),
        ("5h", "10h"),
    )

    loaders = built.trainer.global_loader.loaders
    assert tuple(loader.delta_time for loader in loaders) == (5.0, 5.0)
    assert built.metadata["transition"] == "stochastic_residual_mlp"


def test_dual_scale_adapter_supports_kinetic_encoder_ablation(
    fake_store,
):
    """
    Verify that use_kinetic_encoder changes the architecture and is not merely
    a loss-weight switch.
    """
    fake_store.__init__(
        timepoints=("0h", "2h", "4h"),
        n_cells=12,
        n_genes=4,
    )

    spec = SplitSpec(
        name="extrapolation",
        fit_timepoints=("0h", "2h"),
        test_tasks=(ForecastTask("2h", "4h"),),
    )

    split = resolve_split(fake_store, spec, seed=0)
    adapter = DualScaleExperimentAdapter()

    from trajectoryflow.experiment.data import ExperimentData

    full = adapter.build(
        _model_config(
            use_kinetic_encoder=True,
            local_kinetic_weight=1.0,
        ),
        ExperimentData(fake_store, split),
        torch.device("cpu"),
        seed=0,
    )

    no_kinetic = adapter.build(
        _model_config(
            use_kinetic_encoder=False,
            local_kinetic_weight=0.0,
        ),
        ExperimentData(fake_store, split),
        torch.device("cpu"),
        seed=0,
    )

    # Full model has the requested cell-specific kinetic representation.
    assert full.model.kinetic_encoder.kinetic_dim == 2

    # Ablated model returns a zero-width kinetic representation.
    assert no_kinetic.model.kinetic_encoder.kinetic_dim == 0

    # The loss configuration is independently switchable.
    assert (
        full.trainer.trainer.config.loss_weights.local_kinetic
        == 1.0
    )

    assert (
        no_kinetic.trainer.trainer.config.loss_weights.local_kinetic
        == 0.0
    )


def test_dual_scale_adapter_can_keep_encoder_but_disable_kinetic_loss(
    fake_store,
):
    """
    Important factorial-ablation condition:

        kinetic encoder = ON
        kinetic loss    = OFF

    This tests whether improvements come from metabolically supervising the
    kinetic representation rather than simply giving the transition another
    latent branch.
    """
    fake_store.__init__(
        timepoints=("0h", "2h", "4h"),
        n_cells=12,
        n_genes=4,
    )

    spec = SplitSpec(
        name="extrapolation",
        fit_timepoints=("0h", "2h"),
        test_tasks=(ForecastTask("2h", "4h"),),
    )

    split = resolve_split(fake_store, spec, seed=0)
    adapter = DualScaleExperimentAdapter()

    from trajectoryflow.experiment.data import ExperimentData

    built = adapter.build(
        _model_config(
            use_kinetic_encoder=True,
            local_kinetic_weight=0.0,
        ),
        ExperimentData(fake_store, split),
        torch.device("cpu"),
        seed=0,
    )

    assert built.model.kinetic_encoder.kinetic_dim == 2

    assert (
        built.trainer.trainer.config.loss_weights.local_kinetic
        == 0.0
    )

    assert (
        built.trainer.trainer.config.loss_weights.reconstruction
        == 1.0
    )

    assert (
        built.trainer.trainer.config.loss_weights.population_sliced_wasserstein
        == 1.0
    )


def test_dual_scale_adapter_runs_end_to_end_forecast(
    tmp_path: Path,
    fake_store,
):
    fake_store.__init__(
        timepoints=("0h", "2h", "4h", "6h", "8h"),
        n_cells=12,
        n_genes=4,
    )

    spec = SplitSpec(
        name="extrapolation",
        fit_timepoints=("0h", "2h", "4h", "6h"),
        test_tasks=(ForecastTask("6h", "8h"),),
    )

    split = resolve_split(fake_store, spec, seed=0)

    model = _model_config(
        use_kinetic_encoder=True,
        local_kinetic_weight=1.0,
    )

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
            space=EvaluationSpaceConfig(
                transform="none",
            ),
        ),
        runtime=RuntimeConfig(
            device="cpu",
        ),
        save_checkpoints=False,
        save_predictions=False,
    )

    registry = ExperimentRegistry()

    registry.register_model(
        "dual_scale",
        DualScaleExperimentAdapter(),
    )

    registry.register_evaluator(
        CallableEvaluationAdapter(_mean_evaluator)
    )

    result = ExperimentRunner(
        config,
        registry,
        fake_store,
    ).run(
        model,
        split,
        seed=0,
    )

    assert result.status == "completed"

    # 1 local pretraining step
    # + 1 local step during joint training
    assert result.training_summary["local_steps"] == 2

    # 1 global step during joint training
    assert result.training_summary["global_steps"] == 1

    assert result.tasks[0].source == "6h"
    assert result.tasks[0].target == "8h"
    assert result.tasks[0].n_samples == 2