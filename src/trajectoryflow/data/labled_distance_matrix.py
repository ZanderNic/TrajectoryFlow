# std lib imports

# 3-party import
import pandas as pd
import numpy as np
import scipy

# projekt imports
from trajectoryflow.data import LabeledDenseMatrix



class LabeledDistanceMatrix(LabeledDenseMatrix):
    """
        Specialized labeled dense matrix for pairwise distances.

        Rows and columns use the same info and labels + the matrix is a square matrix that is symetrical.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        info: pd.DataFrame = None,              # col and row labels 
        name=None,
        idx=None,
        label: str = None,
        metadata: dict = None,
        validate_square: bool = True,
        validate_symmetric: bool = False,
        validate_diagonal: bool = False,
    ):

        matrix = np.asarray(matrix)

        if validate_square and matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Distance matrix must be square, got shape {matrix.shape}."
            )

        if info is not None and matrix.shape[0] != len(info):
            raise ValueError(
                f"Matrix size ({matrix.shape[0]}) does not match info ({len(info)})."
            )

        if validate_symmetric and not np.allclose(matrix, matrix.T):
            raise ValueError("Distance matrix must be symmetric.")

        if validate_diagonal and not np.allclose(np.diag(matrix), 0):
            raise ValueError("Distance matrix diagonal must be zero.")

        self.metadata = metadata or {}

        super().__init__(
            matrix=matrix,
            row_info=info,
            col_info=info,
            name=name,
            row_idx=idx,
            col_idx=idx,
            row_label=label,
            col_label=label,
        )


    @property
    def info(self):
        return self.row_info


    @property
    def labels(self):
        return self._get_labels(
            info=self.info,
            label=self.row_label,
        )


    def summary(self):
        return (
            f"LabeledDistanceMatrix(\n"
            f"  name={self.name},\n"
            f"  shape={self.shape},\n"
            f"  dtype={self.matrix_dense.dtype},\n"
            f"  is_view={self.is_view},\n"
            f"  label={self.row_label},\n"
            f"  info={list(self.info.columns) if self.info is not None else None},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )


    def copy(self):
        return LabeledDistanceMatrix(
            matrix=self.matrix_dense.copy(),
            info=self.info.copy() if self.info is not None else None,
            name=self.name,
            label=self.row_label,
            metadata=self.metadata.copy(),
        )


    def __getitem__(self, key):

        if not isinstance(key, tuple):
            row_key = key
            col_key = row_key
        else:
            row_key, col_key = key

        selected_rows = np.atleast_1d(self._row_idx[row_key])
        selected_cols = np.atleast_1d(self._col_idx[col_key])

        return LabeledDenseMatrix(
            matrix=self._matrix,
            row_info=self._row_info,
            col_info=self._col_info,
            name=self.name,
            row_idx=selected_rows,
            col_idx=selected_cols,
            row_label=self.row_label,
            col_label=self.col_label,
        )