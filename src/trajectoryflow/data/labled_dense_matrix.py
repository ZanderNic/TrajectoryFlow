# std lib imports

# 3-party import
import pandas as pd
import numpy as np

# projekt imports




class LabeledDenseMatrix:
    """
    A labeled dense matrix with row/column metadata and view-based indexing.
    """


    def __init__(
        self,
        matrix: np.ndarray,
        row_info: pd.DataFrame = None,
        col_info: pd.DataFrame = None,
        name=None,
        row_idx=None,
        col_idx=None,
        row_label: str = None,
        col_label: str = None,
    ):

        self.name = name

        self._matrix = np.asarray(matrix)

        self._row_info = row_info
        self._col_info = col_info

        self._row_idx = (
            np.arange(self._matrix.shape[0])
            if row_idx is None
            else np.asarray(row_idx)
        )

        self._col_idx = (np.arange(self._matrix.shape[1]) if col_idx is None else np.asarray(col_idx))

        self.row_label = row_label
        self.col_label = col_label

        self._validate_shapes()


    def _validate_shapes(self):
        if self.row_info is not None:
            if self.matrix.shape[0] != len(self.row_info):
                raise ValueError(
                    f"Number of rows in matrix ({self.matrix.shape[0]}) "
                    f"does not match row_info ({len(self.row_info)})"
                )

        if self.col_info is not None:
            if self.matrix.shape[1] != len(self.col_info):
                raise ValueError(
                    f"Number of columns in matrix ({self.matrix.shape[1]}) "
                    f"does not match col_info ({len(self.col_info)})"
                )


    # properties #

    @property
    def matrix_dense(self):
        matrix = self._matrix

        if self._row_idx is not None:
            matrix = matrix[self._row_idx, :]

        if self._col_idx is not None:
            matrix = matrix[:, self._col_idx]

        return matrix


    @property
    def matrix(self):
        matrix = self.matrix_dense

        row_names = self._get_labels(
            info=self.row_info,
            label=self.row_label,
        )

        col_names = self._get_labels(
            info=self.col_info,
            label=self.col_label,
        )

        return pd.DataFrame(
            matrix,
            index=row_names,
            columns=col_names,
        )


    @property
    def row_info(self):
        if self._row_idx is None:
            return self._row_info

        return self._row_info.iloc[self._row_idx]


    @property
    def col_info(self):
        if self._col_idx is None:
            return self._col_info

        return self._col_info.iloc[self._col_idx]


    @property
    def shape(self):
        return self.matrix_dense.shape


    @property
    def n_rows(self):
        return self.shape[0]


    @property
    def n_cols(self):
        return self.shape[1]


    @property
    def is_view(self):
        return (
            len(self._row_idx) != self._matrix.shape[0]
            or len(self._col_idx) != self._matrix.shape[1]
        )


    # selection #

    def _build_mask(
        self,
        info: pd.DataFrame,
        filters: dict,
        starting_mask: np.ndarray,
    ):

        mask = np.ones(len(info), dtype=bool) & starting_mask

        for key, value in filters.items():

            if key not in info.columns:
                raise ValueError(
                    f"the provided key ({key}) "
                    f"is not in the provided info ({info})"
                )

            value = as_list(value)

            mask &= np.isin(info[key].to_numpy(), value)

        return np.where(mask)[0]


    def select_rows(self, **filters):

        starting_mask = np.zeros(len(self._row_info), dtype=bool)

        if self._row_idx is None:
            starting_mask[:] = True
        else:
            starting_mask[self._row_idx] = True

        row_idx = self._build_mask(
            info=self._row_info,
            filters=filters,
            starting_mask=starting_mask,
        )

        return LabeledDenseMatrix(
            matrix=self._matrix,
            row_info=self._row_info,
            col_info=self._col_info,
            name=self.name,
            row_idx=row_idx,
            col_idx=self._col_idx,
            row_label=self.row_label,
            col_label=self.col_label,
        )


    def select_cols(self, **filters):

        starting_mask = np.zeros(len(self._col_info), dtype=bool)

        if self._col_idx is None:
            starting_mask[:] = True
        else:
            starting_mask[self._col_idx] = True

        col_idx = self._build_mask(
            info=self._col_info,
            filters=filters,
            starting_mask=starting_mask,
        )

        return LabeledDenseMatrix(
            matrix=self._matrix,
            row_info=self._row_info,
            col_info=self._col_info,
            name=self.name,
            row_idx=self._row_idx,
            col_idx=col_idx,
            row_label=self.row_label,
            col_label=self.col_label,
        )


    def select(self, rows=None, cols=None):
        result = self

        if rows is not None:
            result = result.select_rows(**rows)

        if cols is not None:
            result = result.select_cols(**cols)

        return result


    # info #

    def summary(self):
        return (
            f"LabeledDenseMatrix(\n"
            f"  name={self.name},\n"
            f"  shape={self.shape},\n"
            f"  dtype={self.matrix_dense.dtype},\n"
            f"  is_view={self.is_view},\n"
            f"  row_info={list(self.row_info.columns) if self.row_info is not None else None},\n"
            f"  col_info={list(self.col_info.columns) if self.col_info is not None else None}\n"
            f")"
        )


    def __repr__(self):
        try:
            return self.matrix.__repr__()
        except Exception:
            return self.summary()


    def __str__(self):
        return self.__repr__()


    def _get_labels(self, info, label):
        if info is None:
            return None

        if label is None:
            return info.index

        if label in info.columns:
            return info[label].values

        raise KeyError(
            f"Unknown label column '{label}'. "
            f"Available columns: {list(info.columns)}"
        )


    # utils #

    def copy(self):
        return LabeledDenseMatrix(
            matrix=self.matrix_dense.copy(),
            row_info=self.row_info.copy() if self.row_info is not None else None,
            col_info=self.col_info.copy() if self.col_info is not None else None,
            name=self.name,
            row_label=self.row_label,
            col_label=self.col_label,
        )


    def head(self, n=5):
        return self.matrix.iloc[:n, :n]


    def to_np(self):
        return self.matrix_dense


    # indexing #

    def __getitem__(self, key):

        if not isinstance(key, tuple):
            row_key = key
            col_key = slice(None)

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
        
        
# helper functions 

def as_list(list_like):
    """
    
    """
    if list_like is None:
        return None
    if isinstance(list_like, (list, tuple, set)):
        return list(list_like)
    return [list_like]

    
    