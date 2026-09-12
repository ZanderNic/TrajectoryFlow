# std-lib imports

# 3 party imports
from torch.utils.data import Dataset

# package imports


class CellIndexDataset(Dataset):
    """Lightweight index dataset; sparse matrix access stays in the collator."""

    def __init__(self, n_cells: int):
        if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells < 0:
            raise ValueError("n_cells must be a non-negative integer.")
        self.n_cells = n_cells

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < self.n_cells:
            raise IndexError(index)
        return index
