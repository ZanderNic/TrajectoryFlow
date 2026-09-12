# std-lib imports

# 3 party imports
import numpy as np

# package imports
from trajectoryflow.experiment.split import CellPartitions, ForecastTask, ResolvedSplit, SplitSpec


def _resolved(task: ForecastTask) -> ResolvedSplit:
    spec = SplitSpec(name="same", fit_timepoints=("0h",), test_tasks=(task,))
    partition = CellPartitions(np.arange(3), np.arange(3), np.array([], dtype=np.int64), np.array([], dtype=np.int64), False)
    return ResolvedSplit(spec=spec, seed=0, fit_timepoints=("0h",), partitions={"0h": partition, "2h": partition, "4h": partition})


def test_split_fingerprint_includes_task_definition():
    first = _resolved(ForecastTask("0h", "2h"))
    second = _resolved(ForecastTask("0h", "4h"))
    assert first.fingerprint != second.fingerprint
