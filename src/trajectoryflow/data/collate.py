# std-lib imports

# 3 party imports
import numpy as np
import torch

# package imports
from trajectoryflow.data.store import TimepointData


class SnapshotCollator:
    """
        Construct dense PyTorch minibatches from a sparse SCI-FATE2 snapshot.

        The complete timepoint remains sparse in memory. Only the selected
        cells of the current batch are converted to dense tensors.
    """

    def __init__(
        self,
        data: TimepointData,
        normalize_expression: bool = True,
        library_size: float = 10_000,
    ):
        if library_size <= 0:
            raise ValueError("library_size must be > 0.")

        self.data = data
        self.normalize_expression = normalize_expression
        self.library_size = library_size

    def __call__(
        self,
        indices: list[int],
    ) -> dict[str, torch.Tensor | str]:

        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        # Slice sparse matrices first.
        expression = self.data.expression[indices]
        ntr = self.data.ntr[indices]

        # Only the current minibatch becomes dense.
        expression = torch.from_numpy(
            expression.toarray()
        ).float()

        ntr = torch.from_numpy(
            ntr.toarray()
        ).float()

        if self.normalize_expression:
            expression = self._normalize_expression(
                expression
            )

        return {
            "expression": expression,
            "ntr": ntr,
            "indices": torch.from_numpy(indices),
            "timepoint": self.data.timepoint,
        }

    def _normalize_expression(
        self,
        expression: torch.Tensor,
    ) -> torch.Tensor:
        """
        Library-size normalize each cell and apply log1p.

        x <- log(1 + x / sum(x) * library_size)
        """

        cell_total = expression.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)

        expression = (
            expression
            / cell_total
            * self.library_size
        )

        return torch.log1p(expression)