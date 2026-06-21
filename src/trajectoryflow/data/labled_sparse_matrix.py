# std lib imports

# 3-party import
import pandas as pd
import numpy as np
import scipy

# projekt imports



    
class LabeledSparseMatrix:
    """
    
    
    """  
    
    
    def __init__(
        self, 
        matrix: scipy.sparse._csr.csr_matrix, 
        row_info: pd.DataFrame = None, 
        col_info: pd.DataFrame = None, 
        name = None,
        row_idx = None,
        col_idx = None,
        row_label: str = None,                  # here the user can give the column from col_info that will be used for displaying
        col_label: str = None,                  # same same 
    ):
        
        self.name = name
        
        self._matrix = matrix               # we will use self.matrix as a local matrix for the user and _matrix as a global matrix for intern handling 
        self._row_info = row_info           # we will use self._row_info for the global info and self.row_info for the user as local info
        self._col_info = col_info           # same same         

        self._row_idx = np.arange(self._matrix.shape[0]) if row_idx is None else np.asarray(row_idx)    # here the original row idx [0: _matrix.num_rows] is saved so global idx 
        self._col_idx = np.arange(self._matrix.shape[1]) if col_idx is None else np.asarray(col_idx)    # same same 

        self.row_label = row_label
        self.col_label = col_label

        self._validate_shapes()


    def _validate_shapes(self):
        if self.row_info is not None:
            if self.matrix_sparse.shape[0] != len(self.row_info):
                raise ValueError(f"Number of rows in matrix ({self.matrix_sparse.shape[0]}) does not match row_info ({len(self.row_info)})")

        if self.col_info is not None:
            if self.matrix_sparse.shape[1] != len(self.col_info):
                raise ValueError(f"Number of columns in matrix ({self.matrix_sparse.shape[1]}) does not match col_info ({len(self.col_info)})")
    
    
    #  @propertys #
    
    @property
    def matrix_sparse(self):
        """
            A method that will make the matrix usable for the user because he will only see the part of the matrix that he wants to see (selected _row_idx and _col_idx)
        """
        matrix = self._matrix

        if self._row_idx is not None:
            matrix = matrix[self._row_idx, :]

        if self._col_idx is not None:
            matrix = matrix[:, self._col_idx]

        return matrix

    @property
    def matrix(self) -> pd.DataFrame:
        matrix = self.matrix_sparse

        row_names = self._get_labels(info=self.row_info, label=self.row_label)
        col_names = self._get_labels(info=self.col_info,label=self.col_label)

        return pd.DataFrame(
            matrix.toarray(),
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
        return self.matrix_sparse.shape


    @property
    def n_rows(self):
        return self.shape[0]


    @property
    def n_cols(self):
        return self.shape[1]
    
    
    @property
    def density(self):
        return self.matrix_sparse.nnz / (self.shape[0] * self.shape[1])
    
    
    @property
    def is_view(self):
        return len(self._row_idx) == self._matrix.shape[0] or len(self._col_idx) == self._matrix.shape[1]

    @property
    def T(self):
        return LabeledSparseMatrix(
            matrix=self.matrix_sparse.T,
            row_info=self.col_info,
            col_info=self.row_info,
            name=f"{self.name}_T" if self.name is not None else None,
            row_label=self.col_label,
            col_label=self.row_label,
        )


    def row_density(self):
        nnz_per_row = self.matrix_sparse.getnnz(axis=1)

        return nnz_per_row / self.n_cols


    def col_density(self):
            nnz_per_col = self.matrix_sparse.getnnz(axis=0)

            return nnz_per_col / self.n_rows
    
    
    def max_row_density(self):
        return np.max(self.row_density())


    def max_col_density(self):
        return np.max(self.col_density())


    # mathematical operations #

    def _binary_operation(self, other, op, op_name):
        if isinstance(other, LabeledSparseMatrix):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} != {other.shape}")

            if self.row_info is not None and other.row_info is not None:
                if not self.row_info.index.equals(other.row_info.index):
                    raise ValueError("Row indices do not match.")

            if self.col_info is not None and other.col_info is not None:
                if not self.col_info.index.equals(other.col_info.index):
                    raise ValueError("Column indices do not match.")

            matrix = op(self.matrix_sparse, other.matrix_sparse)
            name = f"({self.name}{op_name}{other.name})"

        elif np.isscalar(other):
            matrix = op(self.matrix_sparse, other)
            name = f"({self.name}{op_name}{other})"

        else:
            raise TypeError(f"Unsupported operation with type {type(other)}")

        return LabeledSparseMatrix(
            matrix=matrix,
            row_info=self.row_info.copy() if self.row_info is not None else None,
            col_info=self.col_info.copy() if self.col_info is not None else None,
            name=name,
            row_label=self.row_label,
            col_label=self.col_label,
        )


    def __mul__(self, other):
        return self._binary_operation(other, lambda a, b: a.multiply(b), "*")


    def __add__(self, other):
        return self._binary_operation(other, lambda a, b: a + b, "+")


    def __sub__(self, other):
        return self._binary_operation(other, lambda a, b: a - b, "-")


    def __truediv__(self, other):
        return self._binary_operation(other, lambda a, b: a / b, "/")


    def __rmul__(self, other):
        return self.__mul__(other)


    def __radd__(self, other):
        return self.__add__(other)


    def __rsub__(self, other):
        return (-1 * self).__add__(other)


    def __rtruediv__(self, other):
        if np.isscalar(other):
            matrix = other / self.matrix_sparse
            return LabeledSparseMatrix(
                matrix=matrix,
                row_info=self.row_info.copy() if self.row_info is not None else None,
                col_info=self.col_info.copy() if self.col_info is not None else None,
                name=f"({other}/{self.name})",
                row_label=self.row_label,
                col_label=self.col_label,
            )

        raise TypeError(f"Unsupported division with type {type(other)}")


    # selecting rows and cloumns #
    
    def _build_mask(
        self, 
        info: pd.DataFrame,                 # should be self.row_info or self.col_info 
        filters: dict,                      # filters a dict that has the form {key (a key from info): value, ....}
        starting_mask: np.array             # here we will give a starting mask corospodning to our local idx 
    ):
        """
            A method that will be used to select rows or columns by filtering trough the col and row info 
            
            info (pd.DataFrame):
                info about col
            filters (dict):
        """
        mask = np.ones(len(info), dtype=bool) & starting_mask
        
        for key, value in filters.items():
            if key not in info.columns:
                raise ValueError(f"the provided key ({key}) is not in the provided info ({info})")
            
            value = as_list(value)                                                                      # to make sure value is a list and not a 
            mask &= np.isin(info[key].to_numpy(), value)                                               
        
        return np.where(mask)[0]                                                                        # we dont want the mask we want the idx of the valid cols or rows 
        
    
    def select_rows(
        self,
        **filters
    ):
      
        starting_mask = np.zeros(len(self._row_info), dtype=bool)                                                                 # this staring mask will be used so we can always work with global idx 
        if self._row_idx is None:
            starting_mask[:] = True
        else:
            starting_mask[self._row_idx] = True
        
        row_idx = self._build_mask(info= self._row_info, filters= filters, starting_mask = starting_mask)
        
        return LabeledSparseMatrix(
            matrix=self._matrix,
            row_info=self._row_info,
            col_info=self._col_info,
            name=self.name,
            row_idx= row_idx,
            col_idx=self._col_idx,
            row_label = self.row_label,
            col_label = self.col_label
        ) 
    
    
    def select_cols(
        self,
        **filters
    ):
      
        starting_mask = np.zeros(len(self._col_info), dtype=bool)                                                                 # this staring mask will be used so we can always work with global idx 
        
        if self._col_idx is None:
            starting_mask[:] = True
        else:
            starting_mask[self._col_idx] = True
        
        col_idx = self._build_mask(info= self._col_info, filters= filters, starting_mask = starting_mask)
        
        return LabeledSparseMatrix(
            matrix=self._matrix,
            row_info=self._row_info,
            col_info=self._col_info,
            name=self.name,
            row_idx= self._row_idx,
            col_idx= col_idx,
            row_label = self.row_label,
            col_label = self.col_label
        ) 
    
    
    def select(self, rows=None, cols=None):
        result = self

        if rows is not None:
            result = result.select_rows(**rows)

        if cols is not None:
            result = result.select_cols(**cols)

        return result
    
    
    def to_df(self, max_rows=100, max_cols=100, col_label="gene_name"):
        matrix = self.matrix_sparse

        if matrix.shape[0] > max_rows or matrix.shape[1] > max_cols:
            raise ValueError(f"Matrix too large for dense DataFrame: {matrix.shape}. Filter first or increase max_rows/max_cols.")

        if self.col_info is not None and col_label in self.col_info.columns:
            columns = self.col_info[col_label].values
        elif self.col_info is not None:
            columns = self.col_info.index
        else:
            columns = None

        index = self.row_info.index if self.row_info is not None else None

        return pd.DataFrame(
            matrix.toarray(),
            index=index,
            columns=columns,
        )
    
    def summary(self):
        return (
            f"LabeledSparseMatrix(\n"
            f"  name={self.name},\n"
            f"  shape={self.shape},\n"
            f"  dtype={self.matrix_sparse.dtype},\n"
            f"  not_zero={self.matrix_sparse.nnz},\n"
            f"  density={self.density:.4%},\n"
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
    
    
    def _get_labels(self, info, label):
        """
            returns the sa
        """    
    
        if info is None:
            return None
        if label is None:
            return info.index
        if label in info.columns:
            return info[label].values

        raise KeyError(f"Unknown label column '{label}'. Available columns: {list(info.columns)}")
    
    
    def __str__(self):
        return self.__repr__()
    
    
    def copy(self):
        return LabeledSparseMatrix(
            matrix=self.matrix_sparse.copy(),
            row_info=self.row_info.copy() if self.row_info is not None else None,
            col_info=self.col_info.copy() if self.col_info is not None else None,
            name=self.name,
            row_label=self.row_label,
            col_label=self.col_label,
        )
    
    def head(self, n=5):
        row_idx = self._row_idx[:n]
        col_idx = self._col_idx[:n]

        matrix = self._matrix[row_idx, :][:, col_idx]

        row_info = self._row_info.iloc[row_idx]
        col_info = self._col_info.iloc[col_idx]

        row_names = self._get_labels(info=row_info, label=self.row_label)
        col_names = self._get_labels(info=col_info, label=self.col_label)

        return pd.DataFrame(
            matrix.toarray(),
            index=row_names,
            columns=col_names,
        )
    
    
    def __getitem__(self, key):
        """
            The __getitem__ function should select the global selected indexes corosponding to the key
        
        """

        if not isinstance(key, tuple):
            row_key = key
            col_key = slice(None)

        else:
            row_key, col_key = key

       
        selected_rows = np.atleast_1d(self._row_idx[row_key])
        selected_cols = np.atleast_1d(self._col_idx[col_key])
        
        return LabeledSparseMatrix(
            matrix=self._matrix,
            row_info=self._row_info,
            col_info=self._col_info,
            name=self.name,
            row_idx= selected_rows,
            col_idx= selected_cols,
            row_label = self.row_label,
            col_label = self.col_label
        ) 
        
        
    def to_np(self):
        return self.matrix_sparse.toarray()



# helper functions 

def as_list(list_like):
    """
    
    """
    if list_like is None:
        return None
    if isinstance(list_like, (list, tuple, set)):
        return list(list_like)
    return [list_like]

    
    