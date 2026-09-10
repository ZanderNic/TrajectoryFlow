# std-lib imports
from dataclasses import dataclass

# 3 party imports
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# package imports


@dataclass
class FakeSnapshot:
    timepoint: str
    expression: sparse.csr_matrix
    new: sparse.csr_matrix
    ntr: sparse.csr_matrix
    obs: pd.DataFrame

    def __len__(self) -> int:
        return self.expression.shape[0]

    @property
    def time_hours(self) -> float:
        return float(self.timepoint.removesuffix("h"))


class FakeStore:

    def __init__(
        self,
        timepoints=("5h", "7.5h", "10h", "15h"),
        n_cells=20,
        n_genes=4,
    ):
        self.timepoints = tuple(timepoints)
        self.n_genes = n_genes
        self.genes = pd.DataFrame(
            {
                "gene": [f"gene_{index}" for index in range(n_genes)],
            }
        )
        self.preprocessing = {
            "normalization": "none",
            "selected_genes": n_genes,
        }

        self._snapshots = {}

        for timepoint in self.timepoints:
            hours = float(timepoint.removesuffix("h"))

            values = (
                np.arange(n_cells * n_genes, dtype=np.float32)
                .reshape(n_cells, n_genes)
                + 1.0
                + hours
            )

            expression = sparse.csr_matrix(values)
            new = sparse.csr_matrix(values * 0.25)
            ntr = sparse.csr_matrix(
                np.divide(
                    values * 0.25,
                    values,
                    out=np.zeros_like(values),
                    where=values != 0,
                )
            )

            self._snapshots[timepoint] = FakeSnapshot(
                timepoint=timepoint,
                expression=expression,
                new=new,
                ntr=ntr,
                obs=pd.DataFrame(
                    {
                        "cell_id": [
                            f"{timepoint}_{index}"
                            for index in range(n_cells)
                        ],
                        "rep": [1] * (n_cells // 2)
                        + [2] * (n_cells - n_cells // 2),
                        "cell_type": [
                            "A" if index % 2 == 0 else "B"
                            for index in range(n_cells)
                        ],
                    }
                ),
            )

    def load(self, timepoint: str) -> FakeSnapshot:
        return self._snapshots[timepoint]


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()
