# std-lib imports
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

# 3 party imports
import pandas as pd
from scipy import sparse

# package imports


@dataclass
class TimepointData:
    timepoint: str
    expression: sparse.csr_matrix
    new: sparse.csr_matrix
    ntr: sparse.csr_matrix
    obs: pd.DataFrame

    def __len__(self) -> int:
        return self.n_cells

    @property
    def n_cells(self) -> int:
        return self.expression.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.expression.shape

    @property
    def time_hours(self) -> float:
        return float(self.timepoint.removesuffix("h"))

    def __repr__(self) -> str:
        return f"TimepointData(timepoint={self.timepoint!r}, cells={self.n_cells:,}, genes={self.n_genes:,})"


class ScifateStore:
    """Lazy reader for a processed SCI-FATE2 dataset with a small LRU cache."""

    _MANIFEST_KEYS = {"dataset", "geo_accession", "n_cells", "n_genes", "timepoint_column", "genes", "preprocessing", "snapshots"}
    _SNAPSHOT_KEYS = {"timepoint", "n_cells", "expression", "new", "ntr", "obs"}

    def __init__(self, root: str | Path, cache_size: int = 2):
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0.")
        self.root, self.cache_size = Path(root), cache_size
        self.manifest = self._read_json(self.root / "manifest.json")
        self._validate_manifest()

        self.dataset_name = str(self.manifest["dataset"])
        self.geo_accession = str(self.manifest["geo_accession"])
        self.n_cells, self.n_genes = int(self.manifest["n_cells"]), int(self.manifest["n_genes"])
        self.timepoint_column = str(self.manifest["timepoint_column"])
        self.snapshots = {str(snapshot["timepoint"]): snapshot for snapshot in self.manifest["snapshots"]}
        self.genes = self._read_table(self.root / self.manifest["genes"])
        self.preprocessing = self._read_json(self.root / self.manifest["preprocessing"])
        self._cache: OrderedDict[str, TimepointData] = OrderedDict()
        self._validate_metadata()

    def __len__(self) -> int:
        return self.n_cells

    @property
    def timepoints(self) -> list[str]:
        return list(self.snapshots)

    @property
    def n_timepoints(self) -> int:
        return len(self.snapshots)

    @property
    def gene_names(self) -> list[str]:
        column = "gene_name" if "gene_name" in self.genes.columns else "gene"
        if column not in self.genes.columns:
            raise KeyError("Gene table needs a 'gene_name' or 'gene' column.")
        return self.genes[column].astype(str).tolist()

    @property
    def gene_ids(self) -> list[str]:
        column = "gene_id" if "gene_id" in self.genes.columns else "gene"
        if column not in self.genes.columns:
            raise KeyError("Gene table needs a 'gene_id' or 'gene' column.")
        return self.genes[column].astype(str).tolist()

    @property
    def cell_counts(self) -> dict[str, int]:
        return {timepoint: int(snapshot["n_cells"]) for timepoint, snapshot in self.snapshots.items()}

    @property
    def cached_timepoints(self) -> list[str]:
        return list(self._cache)

    @property
    def total_expression_nnz(self) -> int:
        return sum(int(snapshot.get("expression_nnz", 0)) for snapshot in self.snapshots.values())

    @property
    def total_ntr_nnz(self) -> int:
        return sum(int(snapshot.get("ntr_nnz", 0)) for snapshot in self.snapshots.values())

    def has_timepoint(self, timepoint) -> bool:
        return str(timepoint) in self.snapshots

    def n_cells_at(self, timepoint) -> int:
        return int(self.snapshots[self._get_timepoint_key(timepoint)]["n_cells"])

    def shape_at(self, timepoint) -> tuple[int, int]:
        return self.n_cells_at(timepoint), self.n_genes

    def snapshot_info(self, timepoint) -> dict:
        return dict(self.snapshots[self._get_timepoint_key(timepoint)])

    def gene_index(self, gene_name: str) -> int:
        names = pd.Series(self.gene_names)
        matches = names.index[names == str(gene_name)].to_numpy()
        if len(matches) == 0:
            raise KeyError(f"Unknown gene {gene_name!r}.")
        if len(matches) > 1:
            raise ValueError(f"Gene name {gene_name!r} is not unique.")
        row = self.genes.iloc[int(matches[0])]
        return int(row["gene_index"]) if "gene_index" in self.genes.columns else int(matches[0])

    def has_gene(self, gene_name: str) -> bool:
        return str(gene_name) in set(self.gene_names)

    def load(self, timepoint) -> TimepointData:
        key = self._get_timepoint_key(timepoint)
        if key in self._cache:
            data = self._cache.pop(key)
            self._cache[key] = data
            return data

        spec = self.snapshots[key]
        data = TimepointData(timepoint=key, expression=self._load_csr(spec["expression"]), new=self._load_csr(spec["new"]), ntr=self._load_csr(spec["ntr"]), obs=self._read_table(self.root / spec["obs"]))
        self._validate_timepoint_data(key, data)
        if self.cache_size:
            self._cache[key] = data
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return data

    def load_pair(self, source_time, target_time) -> tuple[TimepointData, TimepointData]:
        return self.load(source_time), self.load(target_time)

    def unload(self, timepoint) -> None:
        self._cache.pop(str(timepoint), None)

    def clear_cache(self) -> None:
        self._cache.clear()

    def summary(self) -> dict:
        return {"dataset": self.dataset_name, "geo_accession": self.geo_accession, "n_cells": self.n_cells, "n_genes": self.n_genes, "n_timepoints": self.n_timepoints, "timepoints": self.timepoints, "cell_counts": self.cell_counts, "timepoint_column": self.timepoint_column, "cached_timepoints": self.cached_timepoints}

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"timepoint": timepoint, "n_cells": int(spec["n_cells"]), "n_genes": int(spec.get("n_genes", self.n_genes)), "expression_nnz": int(spec.get("expression_nnz", 0)), "ntr_nnz": int(spec.get("ntr_nnz", 0)), "cached": timepoint in self._cache} for timepoint, spec in self.snapshots.items()])

    def info(self) -> None:
        print(f"Dataset:       {self.dataset_name}\nGEO accession: {self.geo_accession}\nCells:         {self.n_cells:,}\nGenes:         {self.n_genes:,}\nTimepoints:    {self.n_timepoints}\nCache size:    {self.cache_size}\n\nCells per timepoint:")
        for timepoint, count in self.cell_counts.items():
            print(f"  {timepoint:>10}: {count:,}{' [cached]' if timepoint in self._cache else ''}")

    def _validate_manifest(self) -> None:
        missing = self._MANIFEST_KEYS - self.manifest.keys()
        if missing:
            raise KeyError(f"manifest.json is missing required keys: {sorted(missing)}.")
        if int(self.manifest["n_cells"]) < 0 or int(self.manifest["n_genes"]) < 1:
            raise ValueError("Manifest n_cells must be >= 0 and n_genes must be >= 1.")
        snapshots = self.manifest["snapshots"]
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError("Manifest snapshots must be a non-empty list.")
        names = []
        for snapshot in snapshots:
            missing = self._SNAPSHOT_KEYS - snapshot.keys()
            if missing:
                raise KeyError(f"Snapshot is missing required keys: {sorted(missing)}.")
            names.append(str(snapshot["timepoint"]))
        if len(set(names)) != len(names):
            raise ValueError("Manifest timepoints must be unique.")

    def _validate_metadata(self) -> None:
        if len(self.genes) != self.n_genes:
            raise ValueError(f"Gene metadata has {len(self.genes)} rows, expected {self.n_genes}.")
        snapshot_cells = sum(self.cell_counts.values())
        if snapshot_cells != self.n_cells:
            raise ValueError(f"Snapshot cell counts sum to {snapshot_cells}, expected {self.n_cells}.")
        if any(count < 0 for count in self.cell_counts.values()):
            raise ValueError("Snapshot cell counts must be >= 0.")

    def _validate_timepoint_data(self, key: str, data: TimepointData) -> None:
        expected_shape = (self.n_cells_at(key), self.n_genes)
        shapes = {"expression": data.expression.shape, "new": data.new.shape, "ntr": data.ntr.shape}
        if any(shape != expected_shape for shape in shapes.values()):
            raise ValueError(f"Matrix shape mismatch for timepoint {key!r}: {shapes}, expected {expected_shape}.")
        if len(data.obs) != expected_shape[0]:
            raise ValueError(f"obs has {len(data.obs)} rows at {key!r}, expected {expected_shape[0]}.")

    def _load_csr(self, relative_path: str) -> sparse.csr_matrix:
        path = self.root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required matrix not found: {path}")
        return sparse.load_npz(path).tocsr()

    def _get_timepoint_key(self, timepoint) -> str:
        key = str(timepoint)
        if key not in self.snapshots:
            raise KeyError(f"Unknown timepoint {timepoint!r}. Available: {self.timepoints}")
        return key

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported table format: {path}")

    def __repr__(self) -> str:
        return f"ScifateStore(dataset={self.dataset_name!r}, cells={self.n_cells:,}, genes={self.n_genes:,}, timepoints={self.timepoints}, cached={self.cached_timepoints})"
