# std-lib imports
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

# 3 party imports
import pandas as pd
from scipy import sparse

# package imports


@dataclass
class TimepointData:
    """
        Data belonging to one loaded SCI-FATE2 timepoint.
    """

    timepoint: str
    expression: sparse.csr_matrix
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
        return (
            f"TimepointData("
            f"timepoint={self.timepoint!r}, "
            f"cells={self.n_cells:,}, "
            f"genes={self.n_genes:,}"
            f")"
        )


class ScifateStore:
    """
        Central interface to a processed SCI-FATE2 dataset.

        The store keeps lightweight global metadata in memory and loads
        expression/NTR matrices only when a specific timepoint is requested.

        Parameters
        ----------
        root:
            Root directory containing manifest.json.

        cache_size:
            Maximum number of loaded timepoints kept in memory.
    """

    def __init__(
        self,
        root: str | Path,
        cache_size: int = 2,
    ):
        self.root = Path(root)
        self.cache_size = cache_size

        if cache_size < 0:
            raise ValueError("cache_size must be >= 0.")

        self.manifest = self._read_json(
            self.root / "manifest.json"
        )

        self.dataset_name = self.manifest["dataset"]
        self.geo_accession = self.manifest["geo_accession"]

        self.n_cells = int(self.manifest["n_cells"])
        self.n_genes = int(self.manifest["n_genes"])

        self.timepoint_column = self.manifest[
            "timepoint_column"
        ]

        self.snapshots = {
            str(snapshot["timepoint"]): snapshot
            for snapshot in self.manifest["snapshots"]
        }

        self.genes = self._read_table(
            self.root / self.manifest["genes"]
        )

        self.preprocessing = self._read_json(
            self.root / self.manifest["preprocessing"]
        )

        self._cache: OrderedDict[
            str,
            TimepointData,
        ] = OrderedDict()

        self._validate_metadata()

    # ------------------------------------------------------------------
    # Global dataset properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_cells

    @property
    def timepoints(self) -> list[str]:
        return list(self.snapshots.keys())

    @property
    def n_timepoints(self) -> int:
        return len(self.snapshots)

    @property
    def gene_names(self) -> list[str]:
        return self.genes["gene_name"].astype(str).tolist()

    @property
    def gene_ids(self) -> list[str]:
        return self.genes["gene_id"].astype(str).tolist()

    @property
    def cell_counts(self) -> dict[str, int]:
        return {
            timepoint: int(snapshot["n_cells"])
            for timepoint, snapshot in self.snapshots.items()
        }

    @property
    def cached_timepoints(self) -> list[str]:
        return list(self._cache.keys())

    @property
    def total_expression_nnz(self) -> int:
        return sum(
            int(snapshot.get("expression_nnz", 0))
            for snapshot in self.snapshots.values()
        )

    @property
    def total_ntr_nnz(self) -> int:
        return sum(
            int(snapshot.get("ntr_nnz", 0))
            for snapshot in self.snapshots.values()
        )

    # ------------------------------------------------------------------
    # Timepoint metadata
    # ------------------------------------------------------------------

    def has_timepoint(self, timepoint) -> bool:
        return str(timepoint) in self.snapshots

    def n_cells_at(self, timepoint) -> int:
        key = self._get_timepoint_key(timepoint)
        return int(self.snapshots[key]["n_cells"])

    def shape_at(
        self,
        timepoint,
    ) -> tuple[int, int]:
        return (
            self.n_cells_at(timepoint),
            self.n_genes,
        )

    def snapshot_info(
        self,
        timepoint,
    ) -> dict:
        """
        Return lightweight metadata for a timepoint without loading
        its matrices.
        """
        key = self._get_timepoint_key(timepoint)
        return dict(self.snapshots[key])

    # ------------------------------------------------------------------
    # Gene information
    # ------------------------------------------------------------------

    def gene_index(
        self,
        gene_name: str,
    ) -> int:
        """
        Return the processed matrix column index for a gene name.
        """
        matches = self.genes.index[
            self.genes["gene_name"].astype(str) == str(gene_name)
        ]

        if len(matches) == 0:
            raise KeyError(
                f"Unknown gene '{gene_name}'."
            )

        if len(matches) > 1:
            raise ValueError(
                f"Gene name '{gene_name}' is not unique."
            )

        row = self.genes.loc[matches[0]]

        if "gene_index" in row:
            return int(row["gene_index"])

        return int(matches[0])

    def has_gene(
        self,
        gene_name: str,
    ) -> bool:
        return bool(
            (
                self.genes["gene_name"].astype(str)
                == str(gene_name)
            ).any()
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        timepoint,
    ) -> TimepointData:
        """
            Load one timepoint into memory.
            If already cached, the cached object is returned.
        """
        key = self._get_timepoint_key(timepoint)

        if key in self._cache:
            data = self._cache.pop(key)
            self._cache[key] = data
            return data

        spec = self.snapshots[key]

        expression = sparse.load_npz(
            self.root / spec["expression"]
        ).tocsr()

        ntr = sparse.load_npz(
            self.root / spec["ntr"]
        ).tocsr()

        obs = self._read_table(
            self.root / spec["obs"]
        )

        data = TimepointData(
            timepoint=key,
            expression=expression,
            ntr=ntr,
            obs=obs,
        )

        self._validate_timepoint_data(
            key=key,
            data=data,
        )

        if self.cache_size > 0:
            self._cache[key] = data

            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return data


    def load_pair(
        self,
        source_time,
        target_time,
    ) -> tuple[
        TimepointData,
        TimepointData,
    ]:
        """
            Load two timepoints, useful later for trajectory learning.
        """
        return (
            self.load(source_time),
            self.load(target_time),
        )


    def unload(
        self,
        timepoint,
    ) -> None:
        """
            Remove one timepoint from the in-memory cache.
        """
        key = str(timepoint)
        self._cache.pop(key, None)


    def clear_cache(self) -> None:
        self._cache.clear()


    # ------------------------------------------------------------------
    # Summary / inspection
    # ------------------------------------------------------------------

    def summary(
        self,
    ) -> dict:
        """
        Return a lightweight summary without loading expression matrices.
        """
        return {
            "dataset": self.dataset_name,
            "geo_accession": self.geo_accession,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "n_timepoints": self.n_timepoints,
            "timepoints": self.timepoints,
            "cell_counts": self.cell_counts,
            "timepoint_column": self.timepoint_column,
            "cached_timepoints": self.cached_timepoints,
        }

    def summary_frame(
        self,
    ) -> pd.DataFrame:
        """
        Return one row per timepoint.
        """
        rows = []

        for timepoint in self.timepoints:
            snapshot = self.snapshots[timepoint]

            rows.append(
                {
                    "timepoint": timepoint,
                    "n_cells": int(snapshot["n_cells"]),
                    "n_genes": int(
                        snapshot.get(
                            "n_genes",
                            self.n_genes,
                        )
                    ),
                    "expression_nnz": int(
                        snapshot.get(
                            "expression_nnz",
                            0,
                        )
                    ),
                    "ntr_nnz": int(
                        snapshot.get(
                            "ntr_nnz",
                            0,
                        )
                    ),
                    "cached": timepoint in self._cache,
                }
            )

        return pd.DataFrame(rows)

    def info(self) -> None:
        """
        Print a human-readable dataset overview.
        """
        print(f"Dataset:       {self.dataset_name}")
        print(f"GEO accession: {self.geo_accession}")
        print(f"Cells:         {self.n_cells:,}")
        print(f"Genes:         {self.n_genes:,}")
        print(f"Timepoints:    {self.n_timepoints}")
        print(f"Cache size:    {self.cache_size}")

        print("\nCells per timepoint:")

        for timepoint, count in self.cell_counts.items():
            cached = (
                " [cached]"
                if timepoint in self._cache
                else ""
            )

            print(
                f"  {timepoint:>10}: "
                f"{count:,}{cached}"
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_metadata(self) -> None:
        if len(self.genes) != self.n_genes:
            raise ValueError(
                "Gene metadata does not match manifest: "
                f"{len(self.genes)} rows vs "
                f"{self.n_genes} genes."
            )

        snapshot_cells = sum(self.cell_counts.values())

        if snapshot_cells != self.n_cells:
            raise ValueError(
                "Timepoint cell counts do not match "
                "the global cell count: "
                f"{snapshot_cells} vs {self.n_cells}."
            )

    def _validate_timepoint_data(
        self,
        key: str,
        data: TimepointData,
    ) -> None:
        expected_cells = self.n_cells_at(key)

        if data.expression.shape != data.ntr.shape:
            raise ValueError(
                f"Expression/NTR shape mismatch at "
                f"timepoint '{key}': "
                f"{data.expression.shape} vs "
                f"{data.ntr.shape}."
            )

        if data.n_cells != expected_cells:
            raise ValueError(
                f"Cell count mismatch at timepoint '{key}': "
                f"{data.n_cells} vs expected "
                f"{expected_cells}."
            )

        if data.n_genes != self.n_genes:
            raise ValueError(
                f"Gene count mismatch at timepoint '{key}': "
                f"{data.n_genes} vs expected "
                f"{self.n_genes}."
            )

        if len(data.obs) != data.n_cells:
            raise ValueError(
                f"obs row count mismatch at timepoint '{key}': "
                f"{len(data.obs)} vs "
                f"{data.n_cells} cells."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_timepoint_key(
        self,
        timepoint,
    ) -> str:
        key = str(timepoint)

        if key not in self.snapshots:
            raise KeyError(
                f"Unknown timepoint '{timepoint}'. "
                f"Available: {self.timepoints}"
            )

        return key

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _read_table(
        path: Path,
    ) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        if path.suffix == ".parquet":
            return pd.read_parquet(path)

        if path.suffix == ".csv":
            return pd.read_csv(path)

        raise ValueError(
            f"Unsupported table format: {path}"
        )

    def __repr__(self) -> str:
        return (
            f"ScifateStore("
            f"dataset={self.dataset_name!r}, "
            f"cells={self.n_cells:,}, "
            f"genes={self.n_genes:,}, "
            f"timepoints={self.timepoints}, "
            f"cached={self.cached_timepoints}"
            f")"
        )
