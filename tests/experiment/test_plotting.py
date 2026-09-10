# std-lib imports

# 3 party imports
import numpy as np

# package imports
from trajectoryflow.experiment.plotting import (
    plot_velocity_alignment_histogram,
    plot_velocity_comparison,
)


def test_velocity_plots_are_written(tmp_path):
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.5],
            [3.0, 1.0],
        ]
    )
    predicted = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.2],
            [0.5, 1.0],
            [0.2, 1.0],
        ]
    )
    reference = predicted.copy()

    comparison = plot_velocity_comparison(
        positions=positions,
        predicted=predicted,
        reference=reference,
        labels=np.array(["A", "A", "B", "B"]),
        path=tmp_path / "velocity.png",
        max_cells=4,
        max_arrows=4,
        seed=0,
    )
    histogram = plot_velocity_alignment_histogram(
        cosine_similarity=np.array([1.0, 0.5, 0.0, -0.5]),
        path=tmp_path / "hist.png",
    )

    assert comparison.exists()
    assert comparison.stat().st_size > 0
    assert histogram.exists()
    assert histogram.stat().st_size > 0
