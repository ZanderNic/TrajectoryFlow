# std-lib imports
from __future__ import annotations

import json
import re
import shutil
import zlib
from pathlib import Path

# 3 party imports
import h5py
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from tqdm import tqdm

# package imports
from trajectoryflow.data.scifate2_annotations import load_or_reconstruct_authors_cell_annotations


GEO_ACCESSION = "GSE236512"

FILES: dict[str, str] = {
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
                    if not decompressor.eof:
                        raise OSError("Downloaded gzip stream ended before its end-of-stream marker.")

        tmp_path.replace(output_h5ad)
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
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
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
    """Read obs/var metadata in backed mode without loading expression matrices."""
    try:
        import anndata as ad
    except ImportError as error:
        raise ImportError("SCI-FATE2 metadata reading requires anndata; install trajectoryflow[data].") from error
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs, var = adata.obs.copy(), adata.var.copy()
        obs_names = adata.obs_names.astype(str).to_numpy()
        gene_ids = adata.var_names.astype(str).to_numpy()
    finally:
        adata.file.close()

    if "cell_id" in obs.columns and obs["cell_id"].notna().all() and obs["cell_id"].astype(str).is_unique:
        cell_ids = obs["cell_id"].astype(str).to_numpy()
    else:
        cell_ids = obs_names
        obs["cell_id"] = cell_ids
    if len(set(cell_ids.tolist())) != len(cell_ids):
        raise ValueError("SCI-FATE2 cell IDs must be globally unique.")
    if "gene_name" not in var.columns:
        var.insert(0, "gene_name", gene_ids)
    return cell_ids, gene_ids, obs, var


def _read_annotation_table(path: Path, cell_id_column: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".h5ad":
        try:
            import anndata as ad
        except ImportError as error:
            raise ImportError("Reading H5AD cell annotations requires anndata; install trajectoryflow[data].") from error
        adata = ad.read_h5ad(path, backed="r")
        try:
            annotations = adata.obs.copy()
            if cell_id_column not in annotations.columns:
                annotations.insert(0, cell_id_column, adata.obs_names.astype(str))
            return annotations.reset_index(drop=True)
        finally:
            adata.file.close()
    raise ValueError(f"Unsupported cell annotation format: {path.suffix!r}.")


def merge_cell_annotations(
    obs: pd.DataFrame,
    path: Path,
    cell_id_column: str = "cell_id",
    cell_type_column: str | None = None,
) -> pd.DataFrame:
    """Merge external annotations by cell ID and expose a canonical ``cell_type`` column."""
    annotations = _read_annotation_table(path, cell_id_column)
    if cell_id_column not in annotations.columns:
        raise KeyError(f"Annotation table is missing cell ID column {cell_id_column!r}.")
    if annotations[cell_id_column].isna().any() or not annotations[cell_id_column].astype(str).is_unique:
        raise ValueError("Annotation cell IDs must be non-missing and unique.")

    annotations = annotations.copy()
    annotations[cell_id_column] = annotations[cell_id_column].astype(str)

    if cell_type_column is None and "cell_annotation" in annotations.columns:
        cell_type_column = "cell_annotation"
    if cell_type_column is not None:
        if cell_type_column not in annotations.columns:
            raise KeyError(f"Annotation table is missing requested cell-type column {cell_type_column!r}.")
        if cell_type_column != "cell_type":
            if "cell_type" in annotations.columns:
                conflict = annotations["cell_type"].notna() & annotations[cell_type_column].notna() & (annotations["cell_type"].astype(str) != annotations[cell_type_column].astype(str))
                if conflict.any():
                    raise ValueError(f"Annotation columns 'cell_type' and {cell_type_column!r} disagree.")
            else:
                annotations["cell_type"] = annotations[cell_type_column]

    base = obs.copy()
    base["cell_id"] = base["cell_id"].astype(str)
    matched = annotations[cell_id_column].isin(set(base["cell_id"]))
    if not matched.any():
        raise ValueError("No annotation cell IDs match the SCI-FATE2 dataset.")
    n_matched = int(matched.sum())
    print(f"[annotations] Matched {n_matched:,}/{len(base):,} SCI-FATE2 cells ({100 * n_matched / len(base):.1f}%).")
    extra = [column for column in annotations.columns if column != cell_id_column]
    indexed = annotations.set_index(cell_id_column)
    for column in extra:
        mapped = base["cell_id"].map(indexed[column])
        if column not in base.columns:
            base[column] = mapped
            continue
        conflict = base[column].notna() & mapped.notna() & (base[column].astype(str) != mapped.astype(str))
        if conflict.any():
            raise ValueError(f"External annotations conflict with existing obs column {column!r}.")
        base[column] = base[column].where(base[column].notna(), mapped)
    if "authors_qc_pass" in base:
        base["authors_qc_pass"] = base["authors_qc_pass"].eq(True)
    return base


def clean_value(value) -> str:
    text = str(value).strip()
    text = text.replace(" ", "")
    text = text.replace("/", "-")
    return text


def timepoint_to_hours(value) -> float:
    """
    Convert a SCI-FATE2 timepoint label to hours.

    Examples:
        "05h" -> 5.0
        "5h"  -> 5.0
        "D7"  -> 168.0
        "D8"  -> 192.0
    """
    if pd.isna(value):
        raise ValueError("Timepoint values must not be missing.")

    if isinstance(value, (int, float, np.integer, np.floating)):
        hours = float(value)
    else:
        text = str(value).strip().lower().replace(" ", "")

        hour_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)h(?:ours?)?", text)
        day_prefix_match = re.fullmatch(r"d(?:ay)?([0-9]+(?:\.[0-9]+)?)", text)
        day_suffix_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)d(?:ays?)?", text)
        numeric_match = re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text)

        if hour_match:
            hours = float(hour_match.group(1))
        elif day_prefix_match:
            hours = float(day_prefix_match.group(1)) * 24.0
        elif day_suffix_match:
            hours = float(day_suffix_match.group(1)) * 24.0
        elif numeric_match:
            hours = float(text)
        else:
            raise ValueError(
                f"Unsupported timepoint format {value!r}. "
                "Expected values such as '05h', '5h', 'D7', or '7d'."
            )

    if not np.isfinite(hours) or hours < 0:
        raise ValueError(f"Invalid timepoint {value!r}: time in hours must be finite and >= 0.")

    return hours


def normalize_timepoint_label(value) -> str:
    """Return a canonical '<hours>h' label with days converted to hours."""
    hours = timepoint_to_hours(value)

    if float(hours).is_integer():
        return f"{int(hours)}h"

    return f"{hours:g}h"


def safe_path_component(value) -> str:
    """
    Turn a metadata value into a safe folder name.

    Example:
        "4 h" -> "4h"
        "D7" -> "168h" after timepoint normalization
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
    if isinstance(min_cells, bool) or not isinstance(min_cells, int) or min_cells < 0:
        raise ValueError("--min-cells must be a non-negative integer.")
    if isinstance(top_genes_by_detection, bool) or not isinstance(top_genes_by_detection, int) or top_genes_by_detection < 0:
        raise ValueError("--top-genes-by-detection must be a non-negative integer.")
    if isinstance(min_gene_nonzero_fraction, bool) or not isinstance(min_gene_nonzero_fraction, (int, float)) or not np.isfinite(min_gene_nonzero_fraction):
        raise ValueError("--min-gene-nonzero-fraction must be a finite number.")

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
    gene_id_column: str | None,
) -> pd.DataFrame:
    if gene_id_column is not None and gene_id_column not in var.columns:
        raise KeyError(f"Requested gene ID column {gene_id_column!r} is missing from var.")
    records = []

    for processed_index, original_index in enumerate(selected_genes):
        row = var.iloc[original_index]
        raw_name = row.get("gene_name", gene_ids[original_index])
        gene_name = str(gene_ids[original_index] if pd.isna(raw_name) else raw_name)
        gene_id = str(row[gene_id_column]) if gene_id_column is not None and pd.notna(row[gene_id_column]) else str(gene_ids[original_index])

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


def _sorted_row(cols: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(cols):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    order = np.argsort(cols)
    return cols[order].astype(np.int64, copy=False), values[order].astype(np.float32, copy=False)


def _ntr_row(total_cols: np.ndarray, total_values: np.ndarray, new_cols: np.ndarray, new_values: np.ndarray, gene_map: np.ndarray, clip_ratio: bool) -> tuple[np.ndarray, np.ndarray]:
    if not len(total_cols) or not len(new_cols):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    order = np.argsort(total_cols)
    total_cols, total_values = total_cols[order], total_values[order]
    positions = np.searchsorted(total_cols, new_cols)
    valid = positions < len(total_cols)
    valid[valid] &= total_cols[positions[valid]] == new_cols[valid]
    if not valid.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    matched_total = total_values[positions[valid]]
    values = np.divide(new_values[valid], matched_total, out=np.zeros(valid.sum(), dtype=np.float32), where=matched_total > 1e-12)
    if clip_ratio:
        values = np.clip(values, 0.0, 1.0)
    keep = np.isfinite(values) & (values != 0)
    return _sorted_row(gene_map[new_cols[valid][keep]], values[keep])


def prepare_row_outputs(
    row: int,
    activation_reader: H5LayerReader,
    new_reader: H5LayerReader,
    gene_map: np.ndarray,
    clip_ratio: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return selected total/new RNA and NTR sparse values for one cell."""
    total_cols, total_values = activation_reader.get_row(row)
    new_cols, new_values = new_reader.get_row(row)
    if not np.isfinite(total_values).all() or not np.isfinite(new_values).all():
        raise ValueError(f"Non-finite RNA value in source row {row}.")
    if (total_values < 0).any() or (new_values < 0).any():
        raise ValueError(f"Negative RNA value in source row {row}; SCI-FATE2 total/new RNA must be non-negative.")

    total_mapped, total_values_filtered, total_original = filter_and_remap_row(total_cols, total_values, gene_map)
    new_mapped, new_values_filtered, new_original = filter_and_remap_row(new_cols, new_values, gene_map)
    expression_cols, expression_values = _sorted_row(total_mapped, total_values_filtered)
    new_output_cols, new_output_values = _sorted_row(new_mapped, new_values_filtered)
    ntr_cols, ntr_values = _ntr_row(total_original, total_values_filtered, new_original, new_values_filtered, gene_map, clip_ratio)
    return expression_cols, expression_values, new_output_cols, new_output_values, ntr_cols, ntr_values


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
) -> list[tuple[str, np.ndarray]]:
    if timepoint_column not in obs.columns:
        raise KeyError(
            f"Timepoint column '{timepoint_column}' not found in obs. "
            f"Available columns: {list(obs.columns)}"
        )

    values = obs[timepoint_column]

    if values.isna().any():
        n_missing = int(values.isna().sum())
        raise ValueError(
            f"Timepoint column '{timepoint_column}' contains {n_missing} missing values."
        )

    grouped = obs.groupby(timepoint_column, sort=False, observed=True).indices
    groups = [(str(timepoint), np.asarray(indices, dtype=np.int64)) for timepoint, indices in grouped.items()]
    groups.sort(key=lambda item: timepoint_to_hours(item[0]))
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
    new_builder = CSRBuilder(n_cols=n_selected_genes)
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
            new_cols,
            new_values,
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
        new_builder.append(
            new_cols,
            new_values,
        )
        ntr_builder.append(
            ntr_cols,
            ntr_values,
        )

    expression = expression_builder.build()
    new = new_builder.build()
    ntr = ntr_builder.build()

    expression_path = timepoint_dir / "expression.npz"
    new_path = timepoint_dir / "new.npz"
    ntr_path = timepoint_dir / "ntr.npz"

    print(f"[write] {expression_path}")
    print(f"[write] {new_path}")
    print(f"[write] {ntr_path}")

    sparse.save_npz(
        expression_path,
        expression,
        compressed=compressed_npz,
    )
    sparse.save_npz(
        new_path,
        new,
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
    relative_new = new_path.relative_to(output_dir)
    relative_ntr = ntr_path.relative_to(output_dir)
    relative_obs = obs_path.relative_to(output_dir)

    return {
        "timepoint": normalize_scalar_for_json(timepoint),
        "folder": folder_name,
        "n_cells": int(len(row_indices)),
        "n_genes": int(n_selected_genes),
        "expression": str(relative_expression),
        "new": str(relative_new),
        "ntr": str(relative_ntr),
        "obs": str(relative_obs),
        "expression_nnz": int(expression.nnz),
        "new_nnz": int(new.nnz),
        "ntr_nnz": int(ntr.nnz),
    }


# ---------------------------------------------------------------------
# Full preprocessing
# ---------------------------------------------------------------------


def _normalize_timepoints(obs: pd.DataFrame, column: str) -> dict[str, str]:
    if column not in obs.columns:
        raise KeyError(f"Timepoint column {column!r} not found in obs. Available columns: {list(obs.columns)}")
    values = obs[column].copy()
    if values.isna().any():
        raise ValueError(f"Timepoint column {column!r} contains {int(values.isna().sum())} missing values.")
    mapping = {str(value): normalize_timepoint_label(value) for value in pd.unique(values)}
    obs[column] = values.map(normalize_timepoint_label)
    print("[timepoints] Normalized to hours:")
    for original, normalized in sorted(mapping.items(), key=lambda item: timepoint_to_hours(item[1])):
        print(f"  {original!r} -> {normalized!r}")
    return mapping


def _layer_readers(h5: h5py.File, activation_layer: str, new_layer: str) -> tuple[H5LayerReader, H5LayerReader]:
    if "layers" not in h5:
        raise KeyError("No 'layers' group found in H5AD file.")
    available = list(h5["layers"].keys())
    print(f"[layers] Available layers: {available}")
    for layer in (activation_layer, new_layer):
        if layer not in available:
            raise KeyError(f"Layer {layer!r} not found. Available layers: {available}")
    activation_reader, new_reader = H5LayerReader(h5, activation_layer), H5LayerReader(h5, new_layer)
    if activation_reader.shape != new_reader.shape:
        raise ValueError(f"Layer shape mismatch: activation={activation_reader.shape}, new={new_reader.shape}")
    return activation_reader, new_reader


def _save_gene_definition(output_dir: Path, activation_reader: H5LayerReader, gene_ids: np.ndarray, var: pd.DataFrame, gene_id_column: str | None, min_cells: int, min_gene_nonzero_fraction: float, top_genes_by_detection: int):
    selected = select_gene_indices(activation_reader, min_cells, min_gene_nonzero_fraction, top_genes_by_detection)
    if len(selected) == 0:
        raise ValueError("No genes passed the filter. Lower --min-cells or --min-gene-nonzero-fraction.")
    gene_map = build_gene_map(activation_reader.shape[1], selected)
    genes_path = save_dataframe(build_gene_table(gene_ids, selected, var, gene_id_column), output_dir / "genes")
    selected_path = output_dir / "selected_gene_indices.npy"
    np.save(selected_path, selected)
    return selected, gene_map, genes_path, selected_path


def _export_snapshots(output_dir: Path, obs: pd.DataFrame, timepoint_column: str, activation_reader: H5LayerReader, new_reader: H5LayerReader, gene_map: np.ndarray, n_selected_genes: int, clip_ratio: bool, compressed_npz: bool) -> list[dict]:
    groups = get_timepoint_groups(obs, timepoint_column)
    print("[timepoints] " + ", ".join(f"{timepoint} ({len(rows)} cells)" for timepoint, rows in groups))
    return [write_timepoint(timepoint=timepoint, row_indices=rows, output_dir=output_dir, obs=obs, activation_reader=activation_reader, new_reader=new_reader, gene_map=gene_map, n_selected_genes=n_selected_genes, clip_ratio=clip_ratio, compressed_npz=compressed_npz) for timepoint, rows in groups]


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def _preprocessing_metadata(dataset: str, h5ad_path: Path, activation_layer: str, new_layer: str, n_cells: int, n_genes: int, n_selected: int, min_cells: int, min_fraction: float, top_genes: int, clip_ratio: bool, timepoint_column: str, timepoint_mapping: dict[str, str], compressed_npz: bool, cell_annotations: Path | None, annotation_cell_id_column: str, annotation_cell_type_column: str | None, obs: pd.DataFrame) -> dict:
    effective_cell_type_column = annotation_cell_type_column or ("cell_annotation" if cell_annotations is not None and "cell_annotation" in obs.columns else None)
    return {
        "dataset": dataset, "geo_accession": GEO_ACCESSION, "source_h5ad": h5ad_path.name,
        "activation_layer": activation_layer, "new_layer": new_layer, "n_original_cells": int(n_cells), "n_original_genes": int(n_genes), "n_selected_genes": int(n_selected),
        "gene_selection": {"method": "detection_frequency", "min_cells": int(min_cells), "min_gene_nonzero_fraction": float(min_fraction), "top_genes_by_detection": int(top_genes), "global_selection": True},
        "new_rna": {"source_layer": new_layer, "stored_sparse": True, "note": "The selected-gene subset of the source new-RNA layer is stored directly for models such as Velvet that require total and new RNA."},
        "ntr": {"definition": "new / total", "clip_to_0_1": bool(clip_ratio), "stored_sparse": True, "note": "Only finite non-zero NTR values are explicitly stored. Use total expression when a validity/expression mask is required."},
        "cell_annotations": {
            "source": None if cell_annotations is None else str(cell_annotations), "cell_id_column": annotation_cell_id_column,
            "cell_type_source_column": effective_cell_type_column, "cell_type_available": "cell_type" in obs.columns,
            "annotated_cells": int(obs["cell_type"].notna().sum()) if "cell_type" in obs.columns else 0,
            "annotated_fraction": float(obs["cell_type"].notna().mean()) if "cell_type" in obs.columns and len(obs) else 0.0,
        },
        "timepoints": {"column": timepoint_column, "unit": "hours", "format": "<hours>h", "mapping_from_source": timepoint_mapping, "note": "Processed timepoints use hours; day labels are multiplied by 24 and leading zeros are removed."},
        "storage": {"matrix_format": "scipy_csr_npz", "matrix_orientation": "cells-by-genes", "dtype": "float32", "compressed_npz": bool(compressed_npz), "split_by": timepoint_column},
    }


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
    gene_id_column: str | None,
    compressed_npz: bool,
    cell_annotations: Path | None = None,
    annotation_cell_id_column: str = "cell_id",
    annotation_cell_type_column: str | None = None,
) -> None:
    """Convert one source H5AD into the timepoint-split sparse TrajectoryFlow layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[metadata] Reading metadata from: {h5ad_path}")
    cell_ids, gene_ids, obs, var = read_metadata(h5ad_path)
    if cell_annotations is not None:
        print(f"[metadata] Merging cell annotations: {cell_annotations}")
        obs = merge_cell_annotations(obs, Path(cell_annotations), annotation_cell_id_column, annotation_cell_type_column)
    timepoint_mapping = _normalize_timepoints(obs, timepoint_column)

    with h5py.File(h5ad_path, "r") as h5:
        activation_reader, new_reader = _layer_readers(h5, activation_layer, new_layer)
        n_cells, n_genes = activation_reader.shape
        if len(cell_ids) != n_cells or len(gene_ids) != n_genes:
            raise ValueError(f"Metadata/matrix shape mismatch: {len(cell_ids)} cell IDs, {len(gene_ids)} gene IDs, matrix={activation_reader.shape}.")
        print(f"[shape] Source H5AD: {n_cells} cells x {n_genes} genes")
        selected, gene_map, genes_path, selected_path = _save_gene_definition(output_dir, activation_reader, gene_ids, var, gene_id_column, min_cells, min_gene_nonzero_fraction, top_genes_by_detection)
        snapshots = _export_snapshots(output_dir, obs, timepoint_column, activation_reader, new_reader, gene_map, len(selected), clip_ratio, compressed_npz)

    preprocessing = _preprocessing_metadata(dataset, h5ad_path, activation_layer, new_layer, n_cells, n_genes, len(selected), min_cells, min_gene_nonzero_fraction, top_genes_by_detection, clip_ratio, timepoint_column, timepoint_mapping, compressed_npz, cell_annotations, annotation_cell_id_column, annotation_cell_type_column, obs)
    preprocessing_path = output_dir / "preprocessing.json"
    _write_json(preprocessing_path, preprocessing)
    manifest = {
        "format_version": 3, "dataset": dataset, "geo_accession": GEO_ACCESSION, "n_cells": int(n_cells), "n_genes": int(len(selected)),
        "timepoint_column": timepoint_column, "timepoint_unit": "hours", "timepoint_format": "<hours>h", "genes": str(genes_path.relative_to(output_dir)),
        "selected_gene_indices": str(selected_path.relative_to(output_dir)), "preprocessing": str(preprocessing_path.relative_to(output_dir)), "snapshots": snapshots,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    print(f"\n[done] Created SCI-FATE2 training dataset\n  root             = {output_dir}\n  manifest         = {manifest_path}\n  genes            = {genes_path}\n  preprocessing    = {preprocessing_path}\n  selected genes   = {selected_path}")
    for snapshot in snapshots:
        print(f"  timepoint {snapshot['timepoint']}: {snapshot['n_cells']} cells -> {snapshot['folder']}/")


def _commit_processed_dataset(staging_dir: Path, output_dir: Path) -> None:
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
    if backup_dir.exists() and not output_dir.exists():
        backup_dir.rename(output_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        output_dir.rename(backup_dir)
    try:
        staging_dir.rename(output_dir)
    except Exception:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------


def prepare_scifate2(
    dataset: str,
    work_dir: Path,
    output_dir: Path,
    h5ad_path: Path | None,
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
    gene_id_column: str | None,
    compressed_npz: bool,
    cell_annotations: Path | None = None,
    annotation_cell_id_column: str = "cell_id",
    annotation_cell_type_column: str | None = None,
    authors_cell_types: bool = False,
    authors_counting_h5ad: Path | None = None,
    force_authors_cell_types: bool = False,
    authors_chunk_size: int = 2048,
) -> None:
    if dataset not in FILES:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available: {list(FILES)}"
        )
    if authors_cell_types and cell_annotations is not None:
        raise ValueError("Use either --authors-cell-types or --cell-annotations, not both.")

    work_dir.mkdir(parents=True, exist_ok=True)
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

    if authors_cell_types:
        if authors_counting_h5ad is None:
            authors_counting_h5ad = h5ad_path if dataset == "counting" else work_dir / FILES["counting"].removesuffix(".gz")
        if not authors_counting_h5ad.exists():
            download_and_decompress_gzip(
                build_geo_download_url(FILES["counting"]),
                authors_counting_h5ad,
                force=False,
            )
        annotation_cache = work_dir / "GSE236512_authors_cell_annotations.csv"
        cell_annotations = load_or_reconstruct_authors_cell_annotations(
            authors_counting_h5ad, annotation_cache, force=force_authors_cell_types, chunk_size=authors_chunk_size,
        )
        annotation_cell_id_column = "cell_id"
        annotation_cell_type_column = "cell_annotation"

    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    try:
        write_processed_dataset(
            h5ad_path=h5ad_path, output_dir=staging_dir, dataset=dataset,
            activation_layer=activation_layer, new_layer=new_layer, min_cells=min_cells,
            min_gene_nonzero_fraction=min_gene_nonzero_fraction, top_genes_by_detection=top_genes_by_detection,
            clip_ratio=clip_ratio, timepoint_column=timepoint_column, gene_id_column=gene_id_column,
            compressed_npz=compressed_npz, cell_annotations=cell_annotations,
            annotation_cell_id_column=annotation_cell_id_column, annotation_cell_type_column=annotation_cell_type_column,
        )
        _commit_processed_dataset(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    if delete_h5ad:
        print(f"[cleanup] Deleting H5AD: {h5ad_path}")
        h5ad_path.unlink()
