# std-lib imports

# 3 party imports
import torch

# package imports
from trajectoryflow.data.loader import make_timepoint_loader
from trajectoryflow.data.store import TimepointData


def sample_cells(
    data: TimepointData,
    n_cells: int = 2048,
    normalize_expression: bool = True,
) -> torch.Tensor:

    batch_size = min(n_cells, len(data))

    loader = make_timepoint_loader(
        data=data,
        batch_size=batch_size,
        shuffle=True,
        normalize_expression=normalize_expression,
    )

    batch = next(iter(loader))

    return batch["expression"]