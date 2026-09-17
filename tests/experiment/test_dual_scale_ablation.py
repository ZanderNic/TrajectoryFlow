# std-lib imports
from pathlib import Path

# 3 party imports

# package imports
from trajectoryflow.experiment.config import load_benchmark_config


def test_ablation_config_changes_only_requested_auxiliary_weights():
    root = Path(__file__).resolve().parents[2]
    config = load_benchmark_config(root / "configs" / "benchmark_dual_scale_ablation.toml")
    models = {model.name: model for model in config.models}

    assert set(models) == {
        "dual_scale_free",
        "dual_scale_gene",
        "dual_scale_gene_past",
        "dual_scale_direction",
    }

    free = models["dual_scale_free"].trainer_params["loss_weights"]
    gene = models["dual_scale_gene"].trainer_params["loss_weights"]
    past = models["dual_scale_gene_past"].trainer_params["loss_weights"]
    direction = models["dual_scale_direction"].trainer_params["loss_weights"]

    assert free["future_gene_sliced_wasserstein"] == 0.0
    assert gene["future_gene_sliced_wasserstein"] > 0.0
    assert gene["past_direction_alignment"] == 0.0
    assert past["past_direction_alignment"] > 0.0
    assert past["kinetic_direction_alignment"] == 0.0
    assert direction["kinetic_direction_alignment"] > 0.0

    common_model = models["dual_scale_free"].model_params
    common_schedule = models["dual_scale_free"].trainer_params["schedule"]
    for model in models.values():
        assert model.model_params == common_model
        assert model.trainer_params["schedule"] == common_schedule
