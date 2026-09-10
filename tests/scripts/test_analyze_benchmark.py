# std-lib imports
import importlib.util
from pathlib import Path

# 3 party imports
import pandas as pd
import pytest

# package imports


def load_script(name: str):
    path = Path(__file__).parents[2] / "scripts" / f"{name}.py"

    if not path.exists():
        pytest.skip(f"{path} not present.")

    spec = importlib.util.spec_from_file_location(
        f"test_{name}",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_rows():
    rows = []

    for model, values in {
        "a": [0.2, 0.3],
        "b": [0.5, 0.6],
    }.items():
        for seed, value in enumerate(values):
            rows.append(
                {
                    "model": model,
                    "split": "split",
                    "seed": seed,
                    "phase": "test",
                    "task": "5h_to_10h",
                    "task_kind": "observed_transition",
                    "source": "5h",
                    "target": "10h",
                    "delta_hours": 5.0,
                    "metric": "mmd",
                    "value": value,
                    "higher_is_better": False,
                }
            )

    for model, values in {
        "a": [0.7, 0.8],
        "b": [0.9, 0.95],
    }.items():
        for seed, value in enumerate(values):
            rows.append(
                {
                    "model": model,
                    "split": "split",
                    "seed": seed,
                    "phase": "test",
                    "task": "5h_to_10h",
                    "task_kind": "observed_transition",
                    "source": "5h",
                    "target": "10h",
                    "delta_hours": 5.0,
                    "metric": "correlation",
                    "value": value,
                    "higher_is_better": True,
                }
            )

    return pd.DataFrame(rows)


def test_metric_summary_and_ranking_respect_metric_direction():
    analysis = load_script("analyze_benchmark")
    metrics = metric_rows()

    summary = analysis.metric_summary(metrics)
    ranks = analysis.ranking(summary)

    mmd_winner = ranks[
        (ranks["metric"] == "mmd")
        & (ranks["rank"] == 1)
    ].iloc[0]

    correlation_winner = ranks[
        (ranks["metric"] == "correlation")
        & (ranks["rank"] == 1)
    ].iloc[0]

    assert mmd_winner["model"] == "a"
    assert correlation_winner["model"] == "b"


def test_paired_comparisons_pair_only_same_seeds():
    analysis = load_script("analyze_benchmark")
    comparisons = analysis.paired_comparisons(
        metric_rows()
    )

    mmd = comparisons[
        comparisons["metric"] == "mmd"
    ].iloc[0]

    assert mmd["n_paired_seeds"] == 2
    assert mmd["model_a"] == "a"
    assert mmd["model_b"] == "b"
    assert mmd["a_win_fraction"] == 1.0


def test_runtime_and_experiment_summary():
    analysis = load_script("analyze_benchmark")

    runtimes = pd.DataFrame(
        [
            {
                "model": "a",
                "split": "split",
                "phase": "training",
                "task": None,
                "wall_seconds": 1.0,
                "cpu_seconds": 0.5,
                "rss_peak_mb": 100.0,
            },
            {
                "model": "a",
                "split": "split",
                "phase": "training",
                "task": None,
                "wall_seconds": 3.0,
                "cpu_seconds": 1.5,
                "rss_peak_mb": 110.0,
            },
        ]
    )

    runtime = analysis.runtime_summary(runtimes)

    assert runtime.iloc[0]["wall_seconds_mean"] == 2.0
    assert runtime.iloc[0]["rss_peak_mb_mean"] == 105.0

    experiments = pd.DataFrame(
        [
            {
                "model": "a",
                "split": "split",
                "status": "completed",
                "training_seconds": 1.0,
                "total_seconds": 2.0,
            },
            {
                "model": "a",
                "split": "split",
                "status": "failed",
                "training_seconds": 3.0,
                "total_seconds": 4.0,
            },
        ]
    )

    summary = analysis.experiment_summary(
        experiments
    ).iloc[0]

    assert summary["n_runs"] == 2
    assert summary["n_completed"] == 1
    assert summary["n_failed"] == 1
    assert summary["training_seconds_mean"] == 2.0
