# std-lib imports

# 3 party imports
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# package imports
from trajectoryflow.data.collate import SnapshotCollator
from trajectoryflow.data.dataset import CellIndexDataset
from trajectoryflow.data.loader import make_timepoint_loader
from trajectoryflow.data.store import TimepointData


def _snapshot():
    expression = sparse.csr_matrix(np.array([[1.0, 1.0], [0.0, 2.0], [3.0, 0.0]], dtype=np.float32))
    new = expression.copy()
    return TimepointData("2h", expression, new, expression.copy(), pd.DataFrame({"cell_id": ["a", "b", "c"]}))


def test_cell_index_dataset_rejects_fractional_size():
    with pytest.raises(ValueError, match="integer"):
        CellIndexDataset(2.5)


def test_snapshot_collator_only_densifies_batch_and_normalizes():
    batch = SnapshotCollator(_snapshot())([0, 2])
    assert batch["expression"].shape == (2, 2)
    assert batch["indices"].tolist() == [0, 2]
    assert batch["timepoint"] == "2h"
    assert np.isfinite(batch["expression"].numpy()).all()


def test_timepoint_loader_seed_is_reproducible():
    first = [batch["indices"].tolist() for batch in make_timepoint_loader(_snapshot(), batch_size=2, seed=7)]
    second = [batch["indices"].tolist() for batch in make_timepoint_loader(_snapshot(), batch_size=2, seed=7)]
    assert first == second
