# std lib imports
from pathlib import Path
from unittest.mock import Base
import warnings
from typing import List

# 3-party import
import pandas as pd
from scipy.io import mmread

# projekt imports
from trajectoryflow.data import LabeledSparseMatrix, LabeledDistanceMatrix, Dataset, LabeledDenseMatrix



class ScifateDataset(Dataset):
    """
    
    """
    
    
    def __init__(
        self, 
        data: LabeledSparseMatrix, 
        ntr: LabeledSparseMatrix  = None, 
        cell_info=None, 
        gene_info=None, 
        name=None,
        trajectories = None
    ):
        super().__init__(
            data=data,
            sample_info=cell_info,
            feature_info=gene_info,
            name=name
        )

        self.trajectories = trajectories
        self.ntr = ntr
   
        self.distance_matrices = {}
        self.distance_scores = {}
        
        self.cluster_assignments = {}
        
        self.reduced_matrices = {}
        self.reductions = {}
        

    #***# @propertys #***#***#***#***#***#***#***#***#***#***#***#***#
    @property
    def cell_info(self):
        return self.sample_info


    @property
    def gene_info(self):
        return self.feature_info


    @property
    def expression_matrix(self) -> LabeledSparseMatrix:
        return self.data
    
    
    #***# trajectory handling #***#***#***#***#***#***#***#***#***#***#***#***#
    
    def get_trajectory_id(self, barcode):
        if self.trajectories is None:
            raise ValueError("No trajectories available.")

        mask = self.trajectories.eq(barcode)

        if not mask.any().any():
            raise ValueError(f"No trajectory found for barcode: {barcode}")

        return mask.any(axis=1).idxmax()
    
    
    def get_trajectory(self, trajectory_id):
        if self.trajectories is None:
            raise ValueError("No trajectories available.")

        trajectory = self.trajectories.loc[trajectory_id]
        sorted_columns = sorted(trajectory.index,key=lambda x: int(x.replace("h", "")))

        return trajectory[sorted_columns]
    
    
    def get_trajectory_cells(self, trajectory_id):
        trajectory = self.get_trajectory(trajectory_id)
        return trajectory.dropna().tolist()
    
    
    def get_trajectory_expression(self, trajectory_id: int = None, barcode: str = None) -> LabeledSparseMatrix:
        if barcode and trajectory_id is None:
            trajectory_id = self.get_trajectory_id(barcode=barcode)
            
        cells = self.get_trajectory_cells(trajectory_id)
        return self.expression_matrix.select_rows(barcode=cells)
    
     
    def get_trajectory_ntr(self, trajectory_id: int = None, barcode: str = None) -> LabeledSparseMatrix:
        if self.ntr is None: raise ValueError("No NTR matrix available.")
        
        if barcode and trajectory_id is None:
            trajectory_id = self.get_trajectory_id(barcode=barcode)
            
            
        cells = self.get_trajectory_cells(trajectory_id)
        return self.ntr.select_rows(barcode=cells)
    
    
    def _get_trajectory_vectors(
        self,
        data: str       #  "expression" | "ntr"
    ):
        vectors = []
        for trajectory_id in self.trajectories.index: 
            trajectory_matrix = None
            if data == "expression":
                trajectory_matrix = self.get_trajectory_expression(trajectory_id)
            elif data == "ntr":
                trajectory_matrix = self.get_trajectory_ntr(trajectory_id)

            vector = trajectory_matrix.to_np().reshape(-1)

            vectors.append(vector)

        return np.stack(vectors)
    
    def get_cell_trajectory_mapping(self):
        if self.trajectories is None:
            raise ValueError("No trajectories available.")

        mapping = {}

        for trajectory_id in self.trajectories.index:
            cells = self.get_trajectory_cells(trajectory_id)

            for barcode in cells:
                mapping[barcode] = trajectory_id

        return mapping
    
    def add_trajectory_info(self):
        mapping = self.get_cell_trajectory_mapping()

        self.cell_info["trajectory"] = (
            self.cell_info["barcode"]
            .map(mapping)
        )
    


    #***# info functions #***#***#***#***#***#***#***#***#***#***#***#***#
        
    def summary(self):
        n_cells, n_genes = self.data.shape if self.data is not None else (None, None)

        return str(
            f"ScifateDataset(\n"
            f"  name = {self.name},\n"
            f"  shape = {n_cells} cells x {n_genes} genes,\n"
            f"  ntr = {'yes' if self.ntr is not None else 'no'},\n"
            f"  cell_info = {'yes' if self.cell_info is not None else 'no'},\n"
            f"  gene_info = {'yes' if self.gene_info is not None else 'no'}\n"
            f"  trajectory = {'yes' if self.trajectories is not None else 'no'}\n"
            f")"
        )

    def __str__(self):
        return self.summary()


    def valid_dist_matrix(self) -> List[str]:
        return list(self.distance_matrices.keys())



    def _is_dataset_matrix_view(self, matrix: LabeledSparseMatrix) -> bool:
        if matrix is None:
            return False

        valid_matrices = [
            self.expression_matrix._matrix,
            self.expression_matrix.T._matrix
        ]

        if self.ntr is not None:
            valid_matrices.append(self.ntr._matrix)
            valid_matrices.append(self.ntr.T._matrix)

        return any(matrix._matrix is valid_matrix for valid_matrix in valid_matrices)


    def _get_entity_vectors(
        self, 
        entity: str,                            # "cell" | "gene" | "trajectory" 
        data: str                               # "expression" | "ntr" | "new_rna" | "old_rna"
    ):
        
        if data not in ["expression", "ntr", "new_rna", "old_rna"]:
            raise ValueError("data must be 'expression', 'ntr', 'new_rna', or 'old_rna'.")

        if data in ["ntr", "new_rna", "old_rna"] and self.ntr is None:
            raise ValueError("No NTR matrix available.")
        
        if data == "expression":
            matrix = self.expression_matrix
        elif data == "ntr":
            matrix = self.ntr
        elif data == "new_rna":
            matrix = self.expression_matrix * self.ntr
        elif data == "old_rna":
            matrix = self.expression_matrix - (self.expression_matrix * self.ntr)
            
        if entity == "cell":
            info = self.cell_info.copy()
            if (self.trajectories is not None and "trajectory" not in info.columns):
                info["trajectory"] = info["barcode"].map(self.get_cell_trajectory_mapping())

            return matrix.matrix_sparse, info, "barcode"
        elif entity == "gene":
            return matrix.T.matrix_sparse, self.gene_info, "gene_name"
        elif entity == "trajectory":
            if self.trajectories is None:
                raise ValueError("No trajectories available.")

            X = self._get_trajectory_vectors(data=data)

            trajectory_info = pd.DataFrame(
                {"trajectory_id": self.trajectories.index},
                index=self.trajectories.index,
            )

            return X, trajectory_info, "trajectory_id"

        raise ValueError(f"Unknown entity: {entity}")






class ScifateDataLoader:
    """
    
    
    
    """
    
    def load(
        self, 
        name: str = None,
        folder_path: str = None, 
        expression_matrix_path: str = None,
        ntr_matrix_path: str = None,
        barcodes_path: str = None,
        features_path: str = None,
        trajectories_path: str = None,
        
    ) -> ScifateDataset:
        """
        
        
        """
        
        if folder_path is not None:
            (
                expression_matrix_path,
                ntr_matrix_path,
                barcodes_path,
                features_path,
                trajectories_path,
            ) = self._detect_files_in_folder(
                folder_path=folder_path,
                expression_matrix_path=expression_matrix_path,
                ntr_matrix_path=ntr_matrix_path,
                barcodes_path=barcodes_path,
                features_path=features_path,
                trajectories_path=trajectories_path,
            )
            
        expression_matrix = self.load_expression_matrix(expression_matrix_path)
        ntr_matrix = self.load_ntr_matrix(ntr_matrix_path) if ntr_matrix_path else None
        barcodes = self.load_barcodes(barcodes_path)
        features = self.load_features(features_path)

        trajectories = None
        if trajectories_path is not None:
            trajectories = self.load_trajectories(trajectories_path)
      
        n_cells = len(barcodes) 
        n_genes = len(features)
      
        # we want to make sure that both matrix are n_cells x n_genes
        if expression_matrix is not None:
            if expression_matrix.shape == (n_genes, n_cells):
                expression_matrix = expression_matrix.tocsr()
            elif expression_matrix.shape == (n_cells, n_genes):
                pass
            else:
                raise ValueError(f"the Dimensions of the expression_matrix with shape {expression_matrix.shape} doesn't match n_cells: {n_cells} and n_genes: {n_genes}")
        
        if ntr_matrix is not None:
            if ntr_matrix.shape == (n_genes, n_cells):
                ntr_matrix = ntr_matrix.T.tocsr()
            elif ntr_matrix.shape == (n_cells, n_genes):
                pass
            else:
                raise ValueError(f"the Dimensions of the ntr_matrix with shape {ntr_matrix.shape} doesn't match n_cells: {n_cells} and n_genes: {n_genes}")

        cell_info = self.prepare_cell_info(barcodes)
        gene_info = self.prepare_gene_info(features)

        expression = LabeledSparseMatrix(
            matrix=expression_matrix,
            row_info=cell_info,
            col_info=gene_info,
            name="expression_matrix",
            col_label="gene_name",
            row_label="barcode",
        )

        ntr = None
        if ntr_matrix is not None:
            ntr = LabeledSparseMatrix(
                matrix=ntr_matrix,
                row_info=cell_info,
                col_info=gene_info,
                name="ntr",
                col_label="gene_name",
                row_label="barcode",
            )

        dataset = ScifateDataset(
            data=expression,
            ntr=ntr,
            cell_info=cell_info,
            gene_info=gene_info,
            name=name if name is not None else "SciFateDataset",
            trajectories=trajectories,
        )

        return dataset


    def prepare_cell_info(
        self,
        barcodes
    ):
        cell_info = pd.DataFrame({"barcode": barcodes})

        split = cell_info["barcode"].str.split(".", n=2, expand=True)

        cell_info["cell_line"] = split[0]
        cell_info["time"] = split[1]
        cell_info["cell_barcode"] = split[2]
                  
        return cell_info          
            
            
    def prepare_gene_info(
        self,
        features
    ):
        
       gene_info = features
       
       gene_info.columns = [
            "gene_id",
            "gene_name",
            "feature_type",
            "region",
            "length",
        ]
       
       return gene_info
       
        
    def _detect_files_in_folder(
        self,
        folder_path,
        expression_matrix_path=None,
        ntr_matrix_path=None,
        barcodes_path=None,
        features_path=None,
        trajectories_path=None,
    ):
        folder = Path(folder_path)

        files = list(folder.iterdir())

        for file in files:
            if not file.is_file():
                continue

            name = file.name.lower()

            if name == "matrix.mtx":
                if expression_matrix_path is not None:
                    warnings.warn("expression_matrix_path overwritten by file found in folder")
                expression_matrix_path = file

            elif name == "barcodes.tsv":
                if barcodes_path is not None:
                    warnings.warn("barcodes_path overwritten by file found in folder")
                barcodes_path = file

            elif name in ["features.tsv", "genes.tsv"]:
                if features_path is not None:
                    warnings.warn("features_path overwritten by file found in folder")
                features_path = file

            elif name.endswith(".mtx") and "ntr" in name:
                if ntr_matrix_path is not None:
                    warnings.warn("ntr_matrix_path overwritten by file found in folder")
                ntr_matrix_path = file

            elif name.endswith(".csv") and ("traj" in name or "trajectory" in name):
                if trajectories_path is not None:
                    warnings.warn("trajectories_path overwritten by file found in folder")
                trajectories_path = file

        return (
            expression_matrix_path,
            ntr_matrix_path,
            barcodes_path,
            features_path,
            trajectories_path,
        )


    def load_expression_matrix(self, expression_matrix_path):
        expression_matrix_path = Path(expression_matrix_path)

        if not expression_matrix_path.exists():
            raise FileNotFoundError(f"Expression matrix not found: {expression_matrix_path}")

        return mmread(expression_matrix_path).tocsr()


    def load_ntr_matrix(self, ntr_matrix_path):
        ntr_matrix_path = Path(ntr_matrix_path)

        if not ntr_matrix_path.exists():
            raise FileNotFoundError(f"NTR matrix not found: {ntr_matrix_path}")

        return mmread(ntr_matrix_path).tocsr()


    def load_barcodes(self, barcodes_path):
        barcodes_path = Path(barcodes_path)

        if not barcodes_path.exists():
            raise FileNotFoundError(f"Barcodes file not found: {barcodes_path}")

        barcodes = pd.read_csv(
            barcodes_path,
            sep="\t",
            header=None,
        )

        return barcodes.iloc[:, 0].astype(str).tolist()


    def load_features(self, features_path):
        features_path = Path(features_path)

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        features = pd.read_csv(
            features_path,
            sep="\t",
            header=None,
        )

        return features


    def load_trajectories(self, trajectories_path):
        trajectories_path = Path(trajectories_path)

        if not trajectories_path.exists():
            raise FileNotFoundError(f"Trajectories file not found: {trajectories_path}")

        return pd.read_csv(trajectories_path, index_col=0)
    
    
    
    # helper 
    
    
