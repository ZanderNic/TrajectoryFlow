# std-lib imports

# 3 party imports
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

# package imports
from trajectoryflow.plotting.paper import (
    plot_cbd_transitions,
    plot_cell_type_composition,
    plot_efficiency_summary,
    plot_forecast_comparison,
    plot_metric_comparison,
    plot_velocity_fields,
)


def test_forecast_comparison_uses_real_pca_and_saves(tmp_path):
    rng = np.random.default_rng(0)
    source = rng.poisson(2, size=(30, 8))
    target = rng.poisson(3, size=(35, 8))
    predictions = {"model_a": rng.poisson(3, size=(32, 8)), "model_b": rng.poisson(3, size=(31, 8))}
    path = tmp_path / "forecast.png"
    fig, axes = plot_forecast_comparison(source, target, predictions, path=path, source_labels=np.repeat(["A", "B"], 15), target_labels=np.resize(["A", "B", "C"], 35), max_cells=20)
    assert path.exists()
    assert len(axes) == 4
    fig.clear()


def test_velocity_fields_accepts_only_available_component_pairs(tmp_path):
    rng = np.random.default_rng(1)
    positions = rng.normal(size=(40, 2))
    fields = {"Velvet": rng.normal(size=(40, 2)), "Reference": rng.normal(size=(40, 2))}
    path = tmp_path / "velocity.png"
    fig, axes = plot_velocity_fields(positions, fields, path=path, labels=np.resize(["A", "B"], 40))
    assert path.exists()
    assert axes.shape == (1, 2)
    fig.clear()


def test_metric_comparison_filters_overall_rows(tmp_path):
    metrics = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b", "a"],
            "seed": [0, 1, 0, 1, 0],
            "split": ["s"] * 5,
            "task": ["t"] * 5,
            "task_type": ["velocity"] * 5,
            "reference": ["pts"] * 5,
            "group": [np.nan, np.nan, np.nan, np.nan, "extra"],
            "metric": ["cosine_similarity"] * 5,
            "value": [0.2, 0.3, 0.5, 0.4, 0.99],
            "higher_is_better": [True] * 5,
        }
    )
    path = tmp_path / "metric.png"
    fig, ax = plot_metric_comparison(metrics, "cosine_similarity", path=path, split="s", task="t", reference="pts")
    assert path.exists()
    assert len(ax.get_xticklabels()) == 2
    fig.clear()


def test_cbd_transition_plot(tmp_path):
    metrics = pd.DataFrame(
        {
            "model": np.repeat(["Velvet", "Ours"], 4),
            "seed": [0, 1, 0, 1] * 2,
            "split": ["s"] * 8,
            "task": ["cbd"] * 8,
            "reference": ["cbd"] * 8,
            "group": ["NMP->Mesoderm", "NMP->Mesoderm", "pMN->MN", "pMN->MN"] * 2,
            "metric": ["cosine_similarity"] * 8,
            "value": np.linspace(0.1, 0.8, 8),
        }
    )
    path = tmp_path / "cbd.png"
    fig, ax = plot_cbd_transitions(metrics, path=path, split="s", task="cbd")
    assert path.exists()
    assert len(ax.get_yticklabels()) == 2
    fig.clear()


def test_cell_type_composition_and_efficiency(tmp_path):
    obs = pd.DataFrame({"timepoint": ["0h", "0h", "2h", "2h"], "cell_type": ["NMP", "NMP", "Neural", None]})
    composition = tmp_path / "composition.png"
    fig, _ = plot_cell_type_composition(obs, path=composition)
    assert composition.exists()
    fig.clear()

    experiments = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "split": ["s"] * 4,
            "status": ["completed"] * 4,
            "training_seconds": [1.0, 1.2, 2.0, 2.2],
            "trainable_parameters": [10, 10, 20, 20],
        }
    )
    efficiency = tmp_path / "efficiency.png"
    fig, axes = plot_efficiency_summary(experiments, path=efficiency, split="s")
    assert efficiency.exists()
    assert len(axes) == 2
    fig.clear()


def test_forecast_rejects_feature_mismatch():
    with pytest.raises(ValueError, match="same number of genes"):
        plot_forecast_comparison(np.ones((2, 3)), np.ones((2, 4)), {"x": np.ones((2, 3))})
