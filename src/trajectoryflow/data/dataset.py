# std-lib imports

# 3 party imports
from torch.utils.data import Dataset

# package imports


class CellIndexDataset(Dataset):
    """
        Dataset containing cell indices for one loaded timepoint.

        Actual expression data are accessed by the collator so that
        sparse matrices can be sliced batch-wise before densification.
    """

    def __init__(self, n_cells: int):
        if n_cells < 0:
            raise ValueError("n_cells must be >= 0.")

        self.n_cells = n_cells

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, index: int) -> int:
        return index