import json

import pandas as pd

from trajectoryflow.training.progress import TrainingProgressReporter


def test_training_progress_writes_live_and_persistent_metrics(tmp_path):
    reporter = TrainingProgressReporter(tmp_path / "training", "demo", enabled=False)
    reporter.status("build", "ready")
    reporter.start(total=2)
    reporter.update("joint/local", epoch=1, metrics={"total": 2.0, "new_rna": 1.0})
    reporter.update(
        "joint/global",
        epoch=1,
        metrics={"total": 1.5, "future_sliced_wasserstein": 0.5},
    )
    reporter.epoch("joint", epoch=1, metrics={"local_total": 2.0, "global_total": 1.5})
    reporter.close()

    folder = tmp_path / "training"
    metrics = pd.read_csv(folder / "training_metrics.csv")
    summary = json.loads((folder / "training_summary.json").read_text())

    assert {"step", "epoch"}.issubset(set(metrics["granularity"]))
    assert "future_sliced_wasserstein" in set(metrics["metric"])
    assert summary["status"] == "completed"
    assert summary["steps"] == 2
    assert (folder / "training_events.jsonl").stat().st_size > 0
    assert "joint/global" in (folder / "training.log").read_text()
