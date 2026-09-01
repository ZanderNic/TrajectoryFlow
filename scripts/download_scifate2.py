from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from tqdm import tqdm


GEO_ACCESSION = "GSE236512"

FILES: Dict[str, str] = {
    "counting": "GSE236512_processed_data_counting.h5ad.gz",
    "estimate": "GSE236512_processed_data_estimate.h5ad.gz",
    "splicing": "GSE236512_processed_data_splicing.h5ad.gz",
}


# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------


def build_geo_download_url(filename: str) -> str:
    return (
        "https://www.ncbi.nlm.nih.gov/geo/download/"
        f"?acc={GEO_ACCESSION}&file={filename}&format=file"
    )


def download_and_decompress_gzip(
    url: str,
    output_h5ad: Path,
    force: bool = False,
) -> None:
    """
        Download a .h5ad.gz file from GEO and decompress it directly to .h5ad.

        The compressed .gz file is not stored.
    """
    if output_h5ad.exists() and not force:
        print(f"[skip] Existing H5AD found: {output_h5ad}")
        return

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_h5ad.with_suffix(output_h5ad.suffix + ".tmp")

    if tmp_path.exists():
        print(f"[cleanup] Removing old temporary file: {tmp_path}")
        tmp_path.unlink()

    if output_h5ad.exists() and force:
        print(f"[cleanup] Removing existing H5AD: {output_h5ad}")
        output_h5ad.unlink()

    print(f"[download] {url}")
    print(f"[decompress] Writing to: {output_h5ad}")

    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()

            compressed_size = int(response.headers.get("content-length", 0))
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)

            with open(tmp_path, "wb") as output_file:
                with tqdm(
                    total=compressed_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading + decompressing",
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue

                        progress.update(len(chunk))

                        decompressed = decompressor.decompress(chunk)
                        if decompressed:
                            output_file.write(decompressed)

                    tail = decompressor.flush()
                    if tail:
                        output_file.write(tail)

        tmp_path.rename(output_h5ad)
        print(f"[done] Saved H5AD: {output_h5ad}")

    except Exception:
        if tmp_path.exists():
            print(f"[cleanup] Removing failed temporary file: {tmp_path}")
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------
# H5AD sparse layer access
# ---------------------------------------------------------------------


def get_encoding_type(node) -> str:
    value = node.attrs.get("encoding-type", "")

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


class H5LayerReader:
    """
        Row-wise reader for sparse CSR layers stored inside an H5AD file.

        This avoids loading the complete expression matrix into RAM.
    """

    def __init__(self, h5: h5py.File, layer_name: str):
        self.layer_name = layer_name
        self.node = h5["layers"][layer_name]
        self.encoding = get_encoding_type(self.node)

        if not isinstance(self.node, h5py.Group):
            raise TypeError(
                f"Layer '{layer_name}' is not stored as a sparse matrix group. "
                "This script expects sparse CSR layers."
            )

        if self.encoding != "csr_matrix":
            raise ValueError(
                f"Layer '{layer_name}' has encoding-type='{self.encoding}'. "
                "This script expects CSR matrices for row-wise processing."
            )

        self.data = self.node["data"]
        self.indices = self.node["indices"]
        self.indptr = self.node["indptr"]
        self.shape = tuple(int(x) for x in self.node.attrs["shape"])

    def get_row(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.indptr[row])
        end = int(self.indptr[row + 1])

        cols = self.indices[start:end][:]
        values = self.data[start:end][:]

        return (
            cols.astype(np.int64, copy=False),
            values.astype(np.float32, copy=False),
        )

    def count_nonzero_per_gene(
        self,
        chunk_size: int = 5_000_000,
    ) -> np.ndarray:
        """
        Count in how many cells each gene has a stored non-zero value.

        Explicit stored zeros and non-finite values are ignored.
        """
        n_genes = self.shape[1]
        counts = np.zeros(n_genes, dtype=np.int64)

        n_entries = self.indices.shape[0]

        for start in tqdm(
            range(0, n_entries, chunk_size),
            desc=f"Counting gene detection in '{self.layer_name}'",
        ):
            end = min(start + chunk_size, n_entries)

            cols = self.indices[start:end][:]
            values = self.data[start:end][:]

            valid = np.isfinite(values) & (values != 0)

            if valid.any():
                counts += np.bincount(cols[valid], minlength=n_genes)

        return counts


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------


def read_metadata(h5ad_path: Path):
    """
        Read only obs/var metadata in backed mode.

        Expression matrices remain on disk.
    """
    adata = ad.read_h5ad(h5ad_path, backed="r")

    cell_ids = adata.obs_names.astype(str).to_numpy()
    gene_ids = adata.var_names.astype(str).to_numpy()

    obs = adata.obs.copy()
    var = adata.var.copy()

    adata.file.close()

    obs = obs.copy()
    var = var.copy()

    obs.insert(0, "cell_id", cell_ids)
    var.insert(0, "gene_name", gene_ids)

    return cell_ids, gene_ids, obs, var


def clean_value(value) -> str:
    text = str(value).strip()
    text = text.replace(" ", "")
    text = text.replace("/", "-")
    return text


def safe_path_component(value) -> str:
    """
    Turn a metadata value into a safe folder name.

    Example:
        "4 h" -> "4h"
        "day/3" -> "day-3"
    """
    text = clean_value(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text or "unknown"


def normalize_scalar_for_json(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_dataframe(
    dataframe: pd.DataFrame,
    output_stem: Path,
) -> Path:
    """
    Prefer Parquet, but fall back to CSV if no Parquet engine is installed.
    """
    parquet_path = output_stem.with_suffix(".parquet")

    try:
        dataframe.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        csv_path = output_stem.with_suffix(".csv")
        print(
            "[warning] pyarrow/fastparquet not installed; "
            f"writing CSV instead: {csv_path}"
        )
        dataframe.to_csv(csv_path, index=False)
        return csv_path


# ---------------------------------------------------------------------
# Gene selection
# ---------------------------------------------------------------------


def select_gene_indices(
    activation_reader: H5LayerReader,
    min_cells: int,
    min_gene_nonzero_fraction: float,
    top_genes_by_detection: int,
) -> np.ndarray:
    """
        Select ONE global gene set for the complete dataset.
        The same selected genes and the same column order are used for every
        timepoint file.
    """
    n_cells, n_genes = activation_reader.shape

    if not 0 <= min_gene_nonzero_fraction <= 1:
        raise ValueError(
            "--min-gene-nonzero-fraction must be between 0 and 1."
        )

    if (
        min_cells <= 0
        and min_gene_nonzero_fraction <= 0
        and top_genes_by_detection <= 0
    ):
        print("[filter] No gene filtering.")
        return np.arange(n_genes, dtype=np.int64)

    counts = activation_reader.count_nonzero_per_gene()

    min_cells_from_fraction = int(
        np.ceil(min_gene_nonzero_fraction * n_cells)
    )
    required_min_cells = max(min_cells, min_cells_from_fraction)

    keep = np.ones(n_genes, dtype=bool)

    if required_min_cells > 0:
        keep &= counts >= required_min_cells
        print(
            f"[filter] Genes detected in >= {required_min_cells} cells: "
            f"{keep.sum()} / {n_genes}"
        )

    selected = np.where(keep)[0]

    if (
        top_genes_by_detection > 0
        and len(selected) > top_genes_by_detection
    ):
        ranked = selected[np.argsort(counts[selected])[::-1]]
        selected = ranked[:top_genes_by_detection]

        # Sort by original gene index so every downstream matrix has a stable
        # deterministic column order.
        selected = np.sort(selected)

        print(
            f"[filter] Keeping top {top_genes_by_detection} genes "
            "by detection frequency."
        )

    print(
        f"[filter] Final selected genes: "
        f"{len(selected)} / {n_genes}"
    )

    return selected.astype(np.int64)


def build_gene_map(
    n_genes: int,
    selected_genes: np.ndarray,
) -> np.ndarray:
    """
        Map original H5AD gene indices -> processed matrix column indices.
        Unselected genes map to -1.
    """
    gene_map = np.full(n_genes, -1, dtype=np.int64)
    gene_map[selected_genes] = np.arange(
        len(selected_genes),
        dtype=np.int64,
    )
    return gene_map


def build_gene_table(
    gene_ids: np.ndarray,
    selected_genes: np.ndarray,
    var: pd.DataFrame,
    gene_id_column: Optional[str],
) -> pd.DataFrame:
    records = []

    for processed_index, original_index in enumerate(selected_genes):
        gene_name = str(gene_ids[original_index])

        if (
            gene_id_column is not None
            and gene_id_column in var.columns
        ):
            gene_id = str(var.iloc[original_index][gene_id_column])
        else:
            gene_id = gene_name

        record = {
            "gene_index": processed_index,
            "original_gene_index": int(original_index),
            "gene_id": gene_id,
            "gene_name": gene_name,
        }

        # Preserve any additional var metadata if available.
        for column in var.columns:
            if column == "gene_name":
                continue

            value = var.iloc[original_index][column]

            if pd.isna(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()

            record[column] = value

        records.append(record)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------


def filter_and_remap_row(
    cols: np.ndarray,
    values: np.ndarray,
    gene_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapped_cols = gene_map[cols]

    keep = (
        (mapped_cols >= 0)
        & np.isfinite(values)
        & (values != 0)
    )

    return (
        mapped_cols[keep],
        values[keep],
        cols[keep],
    )


def prepare_row_outputs(
    row: int,
    activation_reader: H5LayerReader,
    new_reader: H5LayerReader,
    gene_map: np.ndarray,
    clip_ratio: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Process one cell.

        Returns:
            activation_cols
            activation_values
            ntr_cols
            ntr_values

        NTR is computed as new / total only for genes where both values exist
        and total > 0.
    """
    total_cols_orig, total_values = activation_reader.get_row(row)
    new_cols_orig, new_values = new_reader.get_row(row)

    (
        total_cols_new,
        total_values,
        total_cols_orig_kept,
    ) = filter_and_remap_row(
        cols=total_cols_orig,
        values=total_values,
        gene_map=gene_map,
    )

    (
        new_cols_new,
        new_values,
        new_cols_orig_kept,
    ) = filter_and_remap_row(
        cols=new_cols_orig,
        values=new_values,
        gene_map=gene_map,
    )

    if len(total_cols_new) > 0:
        activation_order = np.argsort(total_cols_new)
        activation_cols = total_cols_new[activation_order]
        activation_values = total_values[activation_order]
    else:
        activation_cols = np.array([], dtype=np.int64)
        activation_values = np.array([], dtype=np.float32)

    if (
        len(new_cols_new) == 0
        or len(total_cols_orig_kept) == 0
    ):
        return (
            activation_cols,
            activation_values,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    total_order = np.argsort(total_cols_orig_kept)
    total_cols_orig_sorted = total_cols_orig_kept[total_order]
    total_values_sorted = total_values[total_order]

    positions = np.searchsorted(
        total_cols_orig_sorted,
        new_cols_orig_kept,
    )

    valid = positions < len(total_cols_orig_sorted)
    valid[valid] &= (
        total_cols_orig_sorted[positions[valid]]
        == new_cols_orig_kept[valid]
    )

    if not valid.any():
        return (
            activation_cols,
            activation_values,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    matched_total = total_values_sorted[positions[valid]]

    ntr_cols = new_cols_new[valid]

    ntr_values = np.divide(
        new_values[valid],
        matched_total,
        out=np.zeros_like(
            new_values[valid],
            dtype=np.float32,
        ),
        where=matched_total > 1e-12,
    )

    if clip_ratio:
        ntr_values = np.clip(ntr_values, 0.0, 1.0)

    keep_ntr = (
        np.isfinite(ntr_values)
        & (ntr_values != 0)
    )

    ntr_cols = ntr_cols[keep_ntr]
    ntr_values = ntr_values[keep_ntr]

    if len(ntr_cols) > 0:
        ntr_order = np.argsort(ntr_cols)
        ntr_cols = ntr_cols[ntr_order]
        ntr_values = ntr_values[ntr_order]

    return (
        activation_cols,
        activation_values,
        ntr_cols,
        ntr_values,
    )


# ---------------------------------------------------------------------
# CSR construction
# ---------------------------------------------------------------------


class CSRBuilder:
    """
    Incrementally construct one CSR matrix row-by-row.

    Only the current timepoint is accumulated in RAM.
    """

    def __init__(self, n_cols: int):
        self.n_cols = n_cols
        self.data_parts: list[np.ndarray] = []
        self.index_parts: list[np.ndarray] = []
        self.indptr = [0]
        self.nnz = 0

    def append(
        self,
        cols: np.ndarray,
        values: np.ndarray,
    ) -> None:
        cols = np.asarray(cols, dtype=np.int32)
        values = np.asarray(values, dtype=np.float32)

        if len(cols) != len(values):
            raise ValueError(
                "CSRBuilder received different numbers "
                "of columns and values."
            )

        if len(values) > 0:
            self.index_parts.append(cols)
            self.data_parts.append(values)
            self.nnz += len(values)

        self.indptr.append(self.nnz)

    def build(self) -> sparse.csr_matrix:
        if self.data_parts:
            data = np.concatenate(self.data_parts).astype(
                np.float32,
                copy=False,
            )
            indices = np.concatenate(self.index_parts).astype(
                np.int32,
                copy=False,
            )
        else:
            data = np.array([], dtype=np.float32)
            indices = np.array([], dtype=np.int32)

        indptr = np.asarray(self.indptr, dtype=np.int64)

        n_rows = len(indptr) - 1

        return sparse.csr_matrix(
            (data, indices, indptr),
            shape=(n_rows, self.n_cols),
            dtype=np.float32,
        )


# ---------------------------------------------------------------------
# Timepoint export
# ---------------------------------------------------------------------


def get_timepoint_groups(
    obs: pd.DataFrame,
    timepoint_column: str,
) -> list[tuple[object, np.ndarray]]:
    if timepoint_column not in obs.columns:
        raise KeyError(
            f"Timepoint column '{timepoint_column}' not found in obs. "
            f"Available columns: {list(obs.columns)}"
        )

    values = obs[timepoint_column]

    if values.isna().any():
        n_missing = int(values.isna().sum())
        raise ValueError(
            f"Timepoint column '{timepoint_column}' contains "
            f"{n_missing} missing values."
        )

    groups = []

    # sort=False preserves the order in which timepoints first appear.
    for timepoint, group in obs.groupby(
        timepoint_column,
        sort=False,
        observed=True,
    ):
        row_indices = group.index.to_numpy()

        # obs index may not be positional integers, therefore convert through
        # pandas' positional lookup.
        positions = obs.index.get_indexer(row_indices)

        if (positions < 0).any():
            raise RuntimeError(
                f"Could not map all rows for timepoint {timepoint!r}."
            )

        groups.append(
            (
                timepoint,
                positions.astype(np.int64),
            )
        )

    return groups


def write_timepoint(
    timepoint,
    row_indices: np.ndarray,
    output_dir: Path,
    obs: pd.DataFrame,
    activation_reader: H5LayerReader,
    new_reader: H5LayerReader,
    gene_map: np.ndarray,
    n_selected_genes: int,
    clip_ratio: bool,
    compressed_npz: bool,
) -> dict:
    folder_name = safe_path_component(timepoint)
    timepoint_dir = output_dir / "timepoints" / folder_name
    timepoint_dir.mkdir(parents=True, exist_ok=True)

    expression_builder = CSRBuilder(n_cols=n_selected_genes)
    ntr_builder = CSRBuilder(n_cols=n_selected_genes)

    print(
        f"\n[timepoint] {timepoint!r} "
        f"-> {len(row_indices)} cells"
    )

    for row in tqdm(
        row_indices,
        desc=f"Building {folder_name}",
    ):
        (
            activation_cols,
            activation_values,
            ntr_cols,
            ntr_values,
        ) = prepare_row_outputs(
            row=int(row),
            activation_reader=activation_reader,
            new_reader=new_reader,
            gene_map=gene_map,
            clip_ratio=clip_ratio,
        )

        expression_builder.append(
            activation_cols,
            activation_values,
        )
        ntr_builder.append(
            ntr_cols,
            ntr_values,
        )

    expression = expression_builder.build()
    ntr = ntr_builder.build()

    expression_path = timepoint_dir / "expression.npz"
    ntr_path = timepoint_dir / "ntr.npz"

    print(f"[write] {expression_path}")
    print(f"[write] {ntr_path}")

    sparse.save_npz(
        expression_path,
        expression,
        compressed=compressed_npz,
    )
    sparse.save_npz(
        ntr_path,
        ntr,
        compressed=compressed_npz,
    )

    timepoint_obs = obs.iloc[row_indices].copy().reset_index(drop=True)

    obs_path = save_dataframe(
        timepoint_obs,
        timepoint_dir / "obs",
    )

    relative_expression = expression_path.relative_to(output_dir)
    relative_ntr = ntr_path.relative_to(output_dir)
    relative_obs = obs_path.relative_to(output_dir)

    return {
        "timepoint": normalize_scalar_for_json(timepoint),
        "folder": folder_name,
        "n_cells": int(len(row_indices)),
        "n_genes": int(n_selected_genes),
        "expression": str(relative_expression),
        "ntr": str(relative_ntr),
        "obs": str(relative_obs),
        "expression_nnz": int(expression.nnz),
        "ntr_nnz": int(ntr.nnz),
    }


# ---------------------------------------------------------------------
# Full preprocessing
# ---------------------------------------------------------------------


def write_processed_dataset(
    h5ad_path: Path,
    output_dir: Path,
    dataset: str,
    activation_layer: str,
    new_layer: str,
    min_cells: int,
    min_gene_nonzero_fraction: float,
    top_genes_by_detection: int,
    clip_ratio: bool,
    timepoint_column: str,
    gene_id_column: Optional[str],
    compressed_npz: bool,
) -> None:
    """
    Convert the source H5AD into a training-oriented layout:

        output_dir/
            manifest.json
            preprocessing.json
            genes.parquet | genes.csv
            selected_gene_indices.npy
            timepoints/
                <timepoint>/
                    expression.npz
                    ntr.npz
                    obs.parquet | obs.csv

    Every expression/NTR matrix has shape:
        [cells_at_timepoint, globally_selected_genes]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[metadata] Reading metadata from: {h5ad_path}")
    cell_ids, gene_ids, obs, var = read_metadata(h5ad_path)

    with h5py.File(h5ad_path, "r") as h5:
        if "layers" not in h5:
            raise KeyError(
                "No 'layers' group found in H5AD file."
            )

        available_layers = list(h5["layers"].keys())
        print(f"[layers] Available layers: {available_layers}")

        if activation_layer not in available_layers:
            raise KeyError(
                f"Activation layer '{activation_layer}' not found. "
                f"Available layers: {available_layers}"
            )

        if new_layer not in available_layers:
            raise KeyError(
                f"New RNA layer '{new_layer}' not found. "
                f"Available layers: {available_layers}"
            )

        activation_reader = H5LayerReader(
            h5,
            activation_layer,
        )
        new_reader = H5LayerReader(
            h5,
            new_layer,
        )

        if activation_reader.shape != new_reader.shape:
            raise ValueError(
                f"Shape mismatch: "
                f"activation={activation_reader.shape}, "
                f"new={new_reader.shape}"
            )

        n_cells, n_genes = activation_reader.shape

        if len(cell_ids) != n_cells:
            raise ValueError(
                f"Cell ID count mismatch: "
                f"{len(cell_ids)} IDs vs {n_cells} rows."
            )

        if len(gene_ids) != n_genes:
            raise ValueError(
                f"Gene ID count mismatch: "
                f"{len(gene_ids)} IDs vs {n_genes} columns."
            )

        print(
            f"[shape] Source H5AD: "
            f"{n_cells} cells x {n_genes} genes"
        )

        # -------------------------------------------------------------
        # 1. Select genes ONCE globally.
        # -------------------------------------------------------------
        selected_genes = select_gene_indices(
            activation_reader=activation_reader,
            min_cells=min_cells,
            min_gene_nonzero_fraction=min_gene_nonzero_fraction,
            top_genes_by_detection=top_genes_by_detection,
        )

        if len(selected_genes) == 0:
            raise ValueError(
                "No genes passed the filter. "
                "Lower --min-cells or "
                "--min-gene-nonzero-fraction."
            )

        gene_map = build_gene_map(
            n_genes=n_genes,
            selected_genes=selected_genes,
        )

        n_selected_genes = len(selected_genes)

        # -------------------------------------------------------------
        # 2. Save global gene definition.
        # -------------------------------------------------------------
        gene_table = build_gene_table(
            gene_ids=gene_ids,
            selected_genes=selected_genes,
            var=var,
            gene_id_column=gene_id_column,
        )

        genes_path = save_dataframe(
            gene_table,
            output_dir / "genes",
        )

        selected_gene_indices_path = (
            output_dir / "selected_gene_indices.npy"
        )
        np.save(
            selected_gene_indices_path,
            selected_genes,
        )

        # -------------------------------------------------------------
        # 3. Split cells by timepoint.
        # -------------------------------------------------------------
        groups = get_timepoint_groups(
            obs=obs,
            timepoint_column=timepoint_column,
        )

        print(
            "[timepoints] "
            + ", ".join(
                f"{timepoint} ({len(rows)} cells)"
                for timepoint, rows in groups
            )
        )

        snapshots = []

        for timepoint, row_indices in groups:
            snapshot = write_timepoint(
                timepoint=timepoint,
                row_indices=row_indices,
                output_dir=output_dir,
                obs=obs,
                activation_reader=activation_reader,
                new_reader=new_reader,
                gene_map=gene_map,
                n_selected_genes=n_selected_genes,
                clip_ratio=clip_ratio,
                compressed_npz=compressed_npz,
            )

            snapshots.append(snapshot)

    # -----------------------------------------------------------------
    # 4. Save reproducibility/configuration metadata.
    # -----------------------------------------------------------------
    preprocessing = {
        "dataset": dataset,
        "geo_accession": GEO_ACCESSION,
        "source_h5ad": h5ad_path.name,
        "activation_layer": activation_layer,
        "new_layer": new_layer,
        "n_original_cells": int(n_cells),
        "n_original_genes": int(n_genes),
        "n_selected_genes": int(n_selected_genes),
        "gene_selection": {
            "method": "detection_frequency",
            "min_cells": int(min_cells),
            "min_gene_nonzero_fraction": float(
                min_gene_nonzero_fraction
            ),
            "top_genes_by_detection": int(
                top_genes_by_detection
            ),
            "global_selection": True,
        },
        "ntr": {
            "definition": "new / total",
            "clip_to_0_1": bool(clip_ratio),
            "stored_sparse": True,
            "note": (
                "Only finite non-zero NTR values are explicitly stored. "
                "Use the total-expression matrix when a validity/expression "
                "mask is required."
            ),
        },
        "storage": {
            "matrix_format": "scipy_csr_npz",
            "matrix_orientation": "cells-by-genes",
            "dtype": "float32",
            "compressed_npz": bool(compressed_npz),
            "split_by": timepoint_column,
        },
    }

    preprocessing_path = output_dir / "preprocessing.json"

    with open(
        preprocessing_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            preprocessing,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------
    # 5. Manifest used later by ScifateStore.
    # -----------------------------------------------------------------
    manifest = {
        "format_version": 1,
        "dataset": dataset,
        "geo_accession": GEO_ACCESSION,
        "n_cells": int(n_cells),
        "n_genes": int(n_selected_genes),
        "timepoint_column": timepoint_column,
        "genes": str(
            genes_path.relative_to(output_dir)
        ),
        "selected_gene_indices": str(
            selected_gene_indices_path.relative_to(output_dir)
        ),
        "preprocessing": str(
            preprocessing_path.relative_to(output_dir)
        ),
        "snapshots": snapshots,
    }

    manifest_path = output_dir / "manifest.json"

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[done] Created SCI-FATE2 training dataset")
    print(f"  root             = {output_dir}")
    print(f"  manifest         = {manifest_path}")
    print(f"  genes            = {genes_path}")
    print(f"  preprocessing    = {preprocessing_path}")
    print(
        f"  selected genes   = "
        f"{selected_gene_indices_path}"
    )

    for snapshot in snapshots:
        print(
            f"  timepoint {snapshot['timepoint']}: "
            f"{snapshot['n_cells']} cells -> "
            f"{snapshot['folder']}/"
        )


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------


def prepare_scifate2(
    dataset: str,
    work_dir: Path,
    output_dir: Path,
    h5ad_path: Optional[Path],
    activation_layer: str,
    new_layer: str,
    min_cells: int,
    min_gene_nonzero_fraction: float,
    top_genes_by_detection: int,
    force_download: bool,
    force_process: bool,
    delete_h5ad: bool,
    clip_ratio: bool,
    timepoint_column: str,
    gene_id_column: Optional[str],
    compressed_npz: bool,
) -> None:
    if dataset not in FILES:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available: {list(FILES)}"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists() and not force_process:
        print(
            f"[skip] Processed dataset already exists: "
            f"{manifest_path}"
        )
        print(
            "[skip] Use --force-process to recreate it."
        )
        return

    if h5ad_path is None:
        filename_gz = FILES[dataset]
        h5ad_path = (
            work_dir / filename_gz.replace(".gz", "")
        )

    if h5ad_path.exists() and not force_download:
        print(
            f"[skip] Existing H5AD found: "
            f"{h5ad_path}"
        )
    else:
        filename_gz = FILES[dataset]
        url = build_geo_download_url(filename_gz)

        download_and_decompress_gzip(
            url=url,
            output_h5ad=h5ad_path,
            force=force_download,
        )

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"H5AD file does not exist: {h5ad_path}"
        )

    write_processed_dataset(
        h5ad_path=h5ad_path,
        output_dir=output_dir,
        dataset=dataset,
        activation_layer=activation_layer,
        new_layer=new_layer,
        min_cells=min_cells,
        min_gene_nonzero_fraction=min_gene_nonzero_fraction,
        top_genes_by_detection=top_genes_by_detection,
        clip_ratio=clip_ratio,
        timepoint_column=timepoint_column,
        gene_id_column=gene_id_column,
        compressed_npz=compressed_npz,
    )

    if delete_h5ad:
        print(f"[cleanup] Deleting H5AD: {h5ad_path}")
        h5ad_path.unlink()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare SCI-FATE2 GSE236512 as a "
            "timepoint-split sparse training dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=list(FILES.keys()),
        default="estimate",
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/scifate2"),
    )

    parser.add_argument(
        "--h5ad-path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--activation-layer",
        type=str,
        default="total",
    )

    parser.add_argument(
        "--new-layer",
        type=str,
        default="new_estimated",
    )

    parser.add_argument(
        "--timepoint-column",
        type=str,
        default="timepoint",
    )

    parser.add_argument(
        "--gene-id-column",
        type=str,
        default=None,
        help=(
            "Optional column in adata.var used as gene ID. "
            "If absent, var_names are used."
        ),
    )

    parser.add_argument(
        "--min-cells",
        type=int,
        default=0,
        help=(
            "Absolute minimum number of cells in which a "
            "gene must be non-zero."
        ),
    )

    parser.add_argument(
        "--min-gene-nonzero-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of ALL cells in which a gene "
            "must be non-zero."
        ),
    )

    parser.add_argument(
        "--top-genes-by-detection",
        type=int,
        default=10_000,
        help=(
            "Keep only the top N globally detected genes "
            "after filtering. Use 0 to keep all."
        ),
    )

    parser.add_argument(
        "--clip-ratio",
        action="store_true",
        help="Clip computed NTR values to [0, 1].",
    )

    parser.add_argument(
        "--uncompressed-npz",
        action="store_true",
        help=(
            "Write uncompressed NPZ matrices. Larger files, "
            "but potentially faster loading."
        ),
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
    )

    parser.add_argument(
        "--force-process",
        action="store_true",
    )

    parser.add_argument(
        "--delete-h5ad",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_scifate2(
        dataset=args.dataset,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        h5ad_path=args.h5ad_path,
        activation_layer=args.activation_layer,
        new_layer=args.new_layer,
        min_cells=args.min_cells,
        min_gene_nonzero_fraction=args.min_gene_nonzero_fraction,
        top_genes_by_detection=args.top_genes_by_detection,
        force_download=args.force_download,
        force_process=args.force_process,
        delete_h5ad=args.delete_h5ad,
        clip_ratio=args.clip_ratio,
        timepoint_column=args.timepoint_column,
        gene_id_column=args.gene_id_column,
        compressed_npz=not args.uncompressed_npz,
    )


if __name__ == "__main__":
    main()