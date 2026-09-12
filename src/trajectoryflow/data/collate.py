# std-lib imports

# 3 party imports
import numpy as np
import torch

# package imports
from trajectoryflow.data.store import TimepointData


class SnapshotCollator:
    """Densify only the selected rows of a sparse SCI-FATE2 snapshot."""

    def __init__(self, data: TimepointData, normalize_expression: bool = True, library_size: float = 10_000):
        if not isinstance(normalize_expression, bool):
            raise ValueError("normalize_expression must be boolean.")
        if isinstance(library_size, bool) or not isinstance(library_size, (int, float)) or library_size <= 0:
            raise ValueError("library_size must be a positive number.")
        self.data, self.normalize_expression, self.library_size = data, normalize_expression, float(library_size)

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor | str]:
        indices = np.asarray(indices, dtype=np.int64)
        expression = torch.from_numpy(self.data.expression[indices].toarray()).float()
        new = torch.from_numpy(self.data.new[indices].toarray()).float()
        ntr = torch.from_numpy(self.data.ntr[indices].toarray()).float()
        if self.normalize_expression:
            expression = self._normalize_expression(expression)
        return {"expression": expression, "new": new, "ntr": ntr, "indices": torch.from_numpy(indices), "timepoint": self.data.timepoint}

    def _normalize_expression(self, expression: torch.Tensor) -> torch.Tensor:
        cell_total = expression.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.log1p(expression / cell_total * self.library_size)
