# std-lib imports
import importlib.util
from pathlib import Path
from types import SimpleNamespace

# 3 party imports
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# package imports


def load_script():
    path = Path(__file__).parents[2] / "scripts" / "plot_benchmark.py"
    spec = importlib.util.spec_from_file_location("test_plot_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_metric_and_efficiency_figures(tmp_path):
    module = load_script()
    metrics = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "seed": [0, 1, 0, 1],
            "split": ["s"] * 4,
            "phase": ["test"] * 4,
            "task_type": ["forecast"] * 4,
            "task": ["2h_to_4h"] * 4,
            "reference": [np.nan] * 4,
            "group": [np.nan] * 4,
            "metric": ["mmd"] * 4,
            "value": [0.2, 0.3, 0.4, 0.5],
            "higher_is_better": [False] * 4,
        }
    )
    module.make_metric_figures(metrics, tmp_path)
    assert (tmp_path / "metrics" / "s__2h_to_4h__mmd.png").exists()

    experiments = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "split": ["s"] * 4,
            "status": ["completed"] * 4,
            "training_seconds": [1, 2, 3, 4],
            "trainable_parameters": [10, 10, 20, 20],
        }
    )
    module.make_efficiency_figures(experiments, tmp_path)
    assert (tmp_path / "efficiency" / "s.png").exists()


def test_reference_rows_uses_cbd_group_to_disambiguate_duplicates():
    module = load_script()
    reference = SimpleNamespace(
        cell_ids=np.array(["c1", "c1", "c2"]),
        groups=np.array(["A->B", "A->C", "A->B"]),
    )
    cells = pd.DataFrame({"cell_id": ["c1", "c1", "c2"], "group": ["A->C", "A->B", "A->B"]})
    np.testing.assert_array_equal(module._reference_rows(reference, cells), [1, 0, 2])
