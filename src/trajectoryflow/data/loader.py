# std-lib imports

# 3 party imports
import torch
from torch.utils.data import DataLoader

# package imports
from trajectoryflow.data.collate import SnapshotCollator
from trajectoryflow.data.dataset import CellIndexDataset
from trajectoryflow.data.store import TimepointData


def make_timepoint_loader(data: TimepointData, batch_size: int = 256, shuffle: bool = True, normalize_expression: bool = True, library_size: float = 10_000, drop_last: bool = False, seed: int | None = None) -> DataLoader:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
        raise ValueError("shuffle and drop_last must be boolean.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None.")
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset=CellIndexDataset(len(data)), batch_size=batch_size, shuffle=shuffle, collate_fn=SnapshotCollator(data, normalize_expression, library_size), num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=drop_last, generator=generator)
