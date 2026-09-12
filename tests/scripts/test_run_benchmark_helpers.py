# std-lib imports
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

# 3 party imports
import pandas as pd
import pytest
import torch

# package imports


def load_run_benchmark():
    pytest.importorskip("trajectoryflow.data.store")
    pytest.importorskip(
        "trajectoryflow.evaluation.metrics.population"
    )

    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "run_benchmark.py"
    )

    if not path.exists():
        pytest.skip(f"{path} not present.")

    spec = importlib.util.spec_from_file_location(
        "test_run_benchmark",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_result_matches_requires_status_fingerprint_and_signature(tmp_path):
    benchmark = load_run_benchmark()
    run = tmp_path / "run"
    run.mkdir()
    assert not benchmark.completed_result_matches(run, "abc", "sig")

    (run / "result.json").write_text(json.dumps({"status": "completed", "split_fingerprint": "abc"}), encoding="utf-8")
    assert not benchmark.completed_result_matches(run, "abc", "sig")

    benchmark.save_resume_signature(run, "sig")
    assert benchmark.completed_result_matches(run, "abc", "sig")
    assert not benchmark.completed_result_matches(run, "different", "sig")
    assert not benchmark.completed_result_matches(run, "abc", "different")


def test_source_tree_hash_changes_with_source(tmp_path):
    benchmark = load_run_benchmark()
    package = tmp_path / "package"
    package.mkdir()
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = benchmark.source_tree_sha256(package)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert benchmark.source_tree_sha256(package) != first


def test_evaluate_prediction_returns_all_benchmark_metrics():
    benchmark = load_run_benchmark()

    target = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
            [3.0, 4.0, 2.0],
            [4.0, 8.0, 4.0],
        ]
    )

    states = torch.stack(
        [
            target,
            target + 0.1,
        ]
    )

    prediction = SimpleNamespace(
        states=states
    )

    frame = benchmark.evaluate_prediction(
        prediction,
        target,
    )

    assert set(frame["metric"]) == set(
        benchmark.METRICS
    )
    assert len(frame) == len(
        benchmark.METRICS
    )
    assert frame["value"].notna().all()


def test_aggregate_outputs_combines_per_experiment_files(
    tmp_path,
):
    benchmark = load_run_benchmark()

    for seed in (0, 1):
        run = (
            tmp_path
            / "model"
            / "split"
            / f"seed_{seed}"
        )
        run.mkdir(parents=True)

        pd.DataFrame(
            [
                {
                    "model": "model",
                    "split": "split",
                    "seed": seed,
                    "metric": "mmd",
                    "value": 0.1 + seed,
                }
            ]
        ).to_csv(
            run / "metrics.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "model": "model",
                    "split": "split",
                    "seed": seed,
                    "phase": "training",
                    "wall_seconds": 1.0,
                }
            ]
        ).to_csv(
            run / "runtime.csv",
            index=False,
        )

        (run / "result.json").write_text(
            json.dumps(
                {
                    "experiment_id": f"x{seed}",
                    "model": "model",
                    "split": "split",
                    "seed": seed,
                    "status": "completed",
                    "split_fingerprint": "abc",
                    "training": {
                        "wall_seconds": 1.0,
                    },
                    "total": {
                        "wall_seconds": 2.0,
                    },
                    "parameters": {
                        "total": 3,
                        "trainable": 3,
                    },
                    "training_summary": {
                        "stop_reason": "done",
                    },
                    "error": None,
                }
            ),
            encoding="utf-8",
        )

    benchmark.aggregate_outputs(tmp_path)

    assert len(
        pd.read_csv(tmp_path / "metrics.csv")
    ) == 2
    assert len(
        pd.read_csv(tmp_path / "runtimes.csv")
    ) == 2
    assert len(
        pd.read_csv(tmp_path / "experiments.csv")
    ) == 2
