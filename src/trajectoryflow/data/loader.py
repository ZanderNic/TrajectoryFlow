# std-lib imports

# 3 party imports
import torch
from torch.utils.data import DataLoader

# package imports
from trajectoryflow.data.collate import SnapshotCollator
from trajectoryflow.data.dataset import CellIndexDataset
from trajectoryflow.data.store import TimepointData


def make_timepoint_loader(
    data: TimepointData,
    batch_size: int = 256,
    shuffle: bool = True,
    normalize_expression: bool = True,
    library_size: float = 10_000,
    drop_last: bool = False,
) -> DataLoader:

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    dataset = CellIndexDataset(
        n_cells=len(data)
    )

    collator = SnapshotCollator(
        data=data,
        normalize_expression=normalize_expression,
        library_size=library_size,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )