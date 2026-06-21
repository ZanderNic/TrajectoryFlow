from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import h5py
import numpy as np
import requests
from tqdm import tqdm


GEO_ACCESSION = "GSE236512"

FILES: Dict[str, str] = {
    "counting": "GSE236512_processed_data_counting.h5ad.gz",
    "estimate": "GSE236512_processed_data_estimate.h5ad.gz",
    "splicing": "GSE236512_processed_data_splicing.h5ad.gz",
}


def build_geo_download_url(filename: str) -> str:
    return (
        "https://www.ncbi.nlm.nih.gov/geo/download/"
        f"?acc={GEO_ACCESSION}&file={filename}&format=file"
    )


def download_and_decompress_gzip(url: str, output_h5ad: Path, force: bool = False) -> None:
    """
    Download a .h5ad.gz file from GEO and decompress it directly to .h5ad.

    The compressed .gz file is not stored.
    If the decompressed .h5ad already exists, it is reused unless force=True.
    """
    if output_h5ad.exists() and not force:
        print(f"[skip] Existing H5AD found, no download needed: {output_h5ad}")
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


def get_encoding_type(node) -> str:
    value = node.attrs.get("encoding-type", "")

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


class H5LayerReader:
    """
    Row-wise reader for sparse CSR layers inside an .h5ad file.

    This avoids loading the full matrix into memory.
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

        return cols.astype(np.int64, copy=False), values.astype(np.float32, copy=False)

    def count_nonzero_per_gene(self, chunk_size: int = 5_000_000) -> np.ndarray:
        """
        Count in how many cells each gene has a stored non-zero value.
        Explicit stored zeros are ignored.
        """
        n_genes = self.shape[1]
        counts = np.zeros(n_genes, dtype=np.int64)

        n_entries = self.indices.shape[0]

        for start in tqdm(
            range(0, n_entries, chunk_size),
            desc=f"Counting gene detection in layer '{self.layer_name}'",
        ):
            end = min(start + chunk_size, n_entries)

            cols = self.indices[start:end][:]
            values = self.data[start:end][:]

            valid = np.isfinite(values) & (values != 0)

            if valid.any():
                counts += np.bincount(cols[valid], minlength=n_genes)

        return counts


def read_metadata(h5ad_path: Path):
    """
    Read only metadata in backed mode.
    The expression matrices are not loaded into RAM here.
    """
    adata = ad.read_h5ad(h5ad_path, backed="r")

    cell_ids = adata.obs_names.astype(str).to_numpy()
    gene_ids = adata.var_names.astype(str).to_numpy()
    obs = adata.obs.copy()
    var = adata.var.copy()

    adata.file.close()

    return cell_ids, gene_ids, obs, var


def clean_value(value) -> str:
    text = str(value)
    text = text.replace(" ", "")
    text = text.replace("/", "-")
    return text


def build_barcode(
    original_cell_id: str,
    obs_row,
    sample_column: str,
    timepoint_column: str,
    strip_sample_prefix: bool = True,
) -> str:
    """
    Builds barcodes similar to:
    A549.4h.AAAAACTCTCTCAA

    For this dataset this becomes for example:
    MAI5081A385.4h.CGCTTGTTAT

    depending on the actual values in obs[sample_column] and obs[timepoint_column].
    """
    if sample_column in obs_row.index:
        sample = clean_value(obs_row[sample_column])
    else:
        sample = original_cell_id.split("_")[0]

    if timepoint_column in obs_row.index:
        timepoint = clean_value(obs_row[timepoint_column])
    else:
        timepoint = "unknown"

    if strip_sample_prefix and "_" in original_cell_id:
        cell_barcode = original_cell_id.split("_")[-1]
    else:
        cell_barcode = original_cell_id

    return f"{sample}.{timepoint}.{cell_barcode}"


def write_barcodes(
    path: Path,
    cell_ids: np.ndarray,
    obs,
    sample_column: str,
    timepoint_column: str,
) -> None:
    print(f"[write] {path}")

    with open(path, "w", encoding="utf-8") as file:
        for i, original_cell_id in enumerate(cell_ids):
            barcode = build_barcode(
                original_cell_id=original_cell_id,
                obs_row=obs.iloc[i],
                sample_column=sample_column,
                timepoint_column=timepoint_column,
                strip_sample_prefix=True,
            )
            file.write(f"{barcode}\n")


def write_features(
    path: Path,
    gene_ids: np.ndarray,
    selected_genes: np.ndarray,
    var,
    gene_id_column: Optional[str],
    feature_type: str,
    feature_region: str,
) -> None:
    """
    Write features.tsv in a format similar to:

    ENSG00000225602    MTOR-AS1    Gene Expression    Exonic (h.ens90)    705

    This H5AD has no var columns, so by default gene_id == gene_name.
    The final column is the original gene index because real gene lengths are not available.
    """
    print(f"[write] {path}")

    selected_gene_names = gene_ids[selected_genes]

    with open(path, "w", encoding="utf-8") as file:
        for output_index, original_gene_index in enumerate(selected_genes):
            gene_name = str(selected_gene_names[output_index])

            if gene_id_column is not None and gene_id_column in var.columns:
                gene_id = str(var.iloc[original_gene_index][gene_id_column])
            else:
                gene_id = gene_name

            file.write(
                f"{gene_id}\t"
                f"{gene_name}\t"
                f"{feature_type}\t"
                f"{feature_region}\t"
                f"{int(original_gene_index)}\n"
            )


def select_gene_indices(
    activation_reader: H5LayerReader,
    min_cells: int,
    min_gene_nonzero_fraction: float,
    top_genes_by_detection: int,
) -> np.ndarray:
    """
    Select genes based on detection frequency.

    min_cells:
        Absolute minimum number of cells in which a gene must be non-zero.

    min_gene_nonzero_fraction:
        Fraction of cells in which a gene must be non-zero.
        Example: 0.05 means at least 5% of all cells.

    top_genes_by_detection:
        Optional cap. Keeps the most frequently detected genes after filtering.
        Use 0 to disable the cap.
    """
    n_cells, n_genes = activation_reader.shape

    if min_gene_nonzero_fraction < 0 or min_gene_nonzero_fraction > 1:
        raise ValueError("--min-gene-nonzero-fraction must be between 0 and 1.")

    if (
        min_cells <= 0
        and min_gene_nonzero_fraction <= 0
        and top_genes_by_detection <= 0
    ):
        print("[filter] No gene filtering.")
        return np.arange(n_genes, dtype=np.int64)

    counts = activation_reader.count_nonzero_per_gene()

    min_cells_from_fraction = int(np.ceil(min_gene_nonzero_fraction * n_cells))
    required_min_cells = max(min_cells, min_cells_from_fraction)

    keep = np.ones(n_genes, dtype=bool)

    if required_min_cells > 0:
        keep &= counts >= required_min_cells
        print(
            f"[filter] Genes detected in >= {required_min_cells} cells: "
            f"{keep.sum()} / {n_genes}"
        )

    selected = np.where(keep)[0]

    if top_genes_by_detection > 0 and len(selected) > top_genes_by_detection:
        ranked = selected[np.argsort(counts[selected])[::-1]]
        selected = ranked[:top_genes_by_detection]
        selected = np.sort(selected)

        print(
            f"[filter] Keeping top {top_genes_by_detection} genes "
            "by detection frequency."
        )

    print(f"[filter] Final selected genes: {len(selected)} / {n_genes}")

    return selected.astype(np.int64)


def build_gene_map(n_genes: int, selected_genes: np.ndarray) -> np.ndarray:
    gene_map = np.full(n_genes, -1, dtype=np.int64)
    gene_map[selected_genes] = np.arange(len(selected_genes), dtype=np.int64)
    return gene_map


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

    return mapped_cols[keep], values[keep], cols[keep]


def prepare_row_outputs(
    row: int,
    activation_reader: H5LayerReader,
    new_reader: H5LayerReader,
    gene_map: np.ndarray,
    clip_ratio: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process one cell row.

    Returns:
    - activation selected gene columns
    - activation values
    - ratio selected gene columns
    - ratio values
    """
    total_cols_orig, total_values = activation_reader.get_row(row)
    new_cols_orig, new_values = new_reader.get_row(row)

    total_cols_new, total_values, total_cols_orig_kept = filter_and_remap_row(
        cols=total_cols_orig,
        values=total_values,
        gene_map=gene_map,
    )

    new_cols_new, new_values, new_cols_orig_kept = filter_and_remap_row(
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

    if len(new_cols_new) == 0 or len(total_cols_orig_kept) == 0:
        return (
            activation_cols,
            activation_values,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    total_order = np.argsort(total_cols_orig_kept)
    total_cols_orig_sorted = total_cols_orig_kept[total_order]
    total_values_sorted = total_values[total_order]

    positions = np.searchsorted(total_cols_orig_sorted, new_cols_orig_kept)

    valid = positions < len(total_cols_orig_sorted)
    valid[valid] &= total_cols_orig_sorted[positions[valid]] == new_cols_orig_kept[valid]

    if not valid.any():
        return (
            activation_cols,
            activation_values,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    matched_total = total_values_sorted[positions[valid]]

    ratio_cols = new_cols_new[valid]
    ratio_values = np.divide(
        new_values[valid],
        matched_total,
        out=np.zeros_like(new_values[valid], dtype=np.float32),
        where=matched_total > 1e-12,
    )

    if clip_ratio:
        ratio_values = np.clip(ratio_values, 0.0, 1.0)

    keep_ratio = np.isfinite(ratio_values) & (ratio_values != 0)

    ratio_cols = ratio_cols[keep_ratio]
    ratio_values = ratio_values[keep_ratio]

    if len(ratio_cols) > 0:
        ratio_order = np.argsort(ratio_cols)
        ratio_cols = ratio_cols[ratio_order]
        ratio_values = ratio_values[ratio_order]

    return activation_cols, activation_values, ratio_cols, ratio_values


def count_output_entries(
    activation_reader: H5LayerReader,
    new_reader: H5LayerReader,
    gene_map: np.ndarray,
    clip_ratio: bool,
) -> tuple[int, int]:
    """
    MatrixMarket needs the number of non-zero entries in the header.
    Therefore we do one counting pass before the writing pass.
    """
    activation_nnz = 0
    ratio_nnz = 0

    n_cells = activation_reader.shape[0]

    for row in tqdm(range(n_cells), desc="Counting MatrixMarket entries"):
        activation_cols, activation_values, ratio_cols, ratio_values = prepare_row_outputs(
            row=row,
            activation_reader=activation_reader,
            new_reader=new_reader,
            gene_map=gene_map,
            clip_ratio=clip_ratio,
        )

        activation_nnz += len(activation_values)
        ratio_nnz += len(ratio_values)

    print(f"[count] activation nnz: {activation_nnz}")
    print(f"[count] ratio nnz: {ratio_nnz}")

    return activation_nnz, ratio_nnz


def write_matrix_market_header(file, n_rows: int, n_cols: int, nnz: int) -> None:
    file.write("%%MatrixMarket matrix coordinate real general\n")
    file.write("% Generated from SCI-FATE2 GSE236512\n")
    file.write(f"{n_rows} {n_cols} {nnz}\n")


def write_matrix_market_entries(
    file,
    cell_index: int,
    gene_cols: np.ndarray,
    values: np.ndarray,
    orientation: str,
) -> None:
    if len(values) == 0:
        return

    lines = []

    if orientation == "genes-by-cells":
        for gene_col, value in zip(gene_cols, values):
            lines.append(f"{gene_col + 1} {cell_index + 1} {value:.8g}\n")

    elif orientation == "cells-by-genes":
        for gene_col, value in zip(gene_cols, values):
            lines.append(f"{cell_index + 1} {gene_col + 1} {value:.8g}\n")

    else:
        raise ValueError("orientation must be 'genes-by-cells' or 'cells-by-genes'.")

    file.writelines(lines)


def write_mtx_outputs(
    h5ad_path: Path,
    output_dir: Path,
    activation_layer: str,
    new_layer: str,
    min_cells: int,
    min_gene_nonzero_fraction: float,
    top_genes_by_detection: int,
    clip_ratio: bool,
    orientation: str,
    sample_column: str,
    timepoint_column: str,
    gene_id_column: Optional[str],
    feature_type: str,
    feature_region: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "matrix.mtx"
    ntr_path = output_dir / "4sU.Binom.ntr.mtx"
    barcodes_path = output_dir / "barcodes.tsv"
    features_path = output_dir / "features.tsv"

    print(f"[metadata] Reading cell and gene IDs from: {h5ad_path}")
    cell_ids, gene_ids, obs, var = read_metadata(h5ad_path)

    with h5py.File(h5ad_path, "r") as h5:
        if "layers" not in h5:
            raise KeyError("No 'layers' group found in H5AD file.")

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

        activation_reader = H5LayerReader(h5, activation_layer)
        new_reader = H5LayerReader(h5, new_layer)

        if activation_reader.shape != new_reader.shape:
            raise ValueError(
                f"Shape mismatch: activation={activation_reader.shape}, "
                f"new={new_reader.shape}"
            )

        n_cells, n_genes = activation_reader.shape

        if len(cell_ids) != n_cells:
            raise ValueError(f"Cell ID count mismatch: {len(cell_ids)} IDs vs {n_cells} rows.")

        if len(gene_ids) != n_genes:
            raise ValueError(f"Gene ID count mismatch: {len(gene_ids)} IDs vs {n_genes} columns.")

        selected_genes = select_gene_indices(
            activation_reader=activation_reader,
            min_cells=min_cells,
            min_gene_nonzero_fraction=min_gene_nonzero_fraction,
            top_genes_by_detection=top_genes_by_detection,
        )

        if len(selected_genes) == 0:
            raise ValueError(
                "No genes passed the filter. "
                "Lower --min-cells or --min-gene-nonzero-fraction."
            )

        gene_map = build_gene_map(n_genes=n_genes, selected_genes=selected_genes)

        n_selected_genes = len(selected_genes)

        if orientation == "genes-by-cells":
            matrix_rows = n_selected_genes
            matrix_cols = n_cells
        elif orientation == "cells-by-genes":
            matrix_rows = n_cells
            matrix_cols = n_selected_genes
        else:
            raise ValueError("orientation must be 'genes-by-cells' or 'cells-by-genes'.")

        print(f"[shape] H5AD matrix shape: {n_cells} cells x {n_genes} genes")
        print(f"[shape] Output matrix shape: {matrix_rows} rows x {matrix_cols} columns")
        print(f"[orientation] {orientation}")
        print(f"[filter] min_cells={min_cells}")
        print(f"[filter] min_gene_nonzero_fraction={min_gene_nonzero_fraction}")
        print(f"[filter] top_genes_by_detection={top_genes_by_detection}")

        activation_nnz, ratio_nnz = count_output_entries(
            activation_reader=activation_reader,
            new_reader=new_reader,
            gene_map=gene_map,
            clip_ratio=clip_ratio,
        )

        write_barcodes(
            path=barcodes_path,
            cell_ids=cell_ids,
            obs=obs,
            sample_column=sample_column,
            timepoint_column=timepoint_column,
        )

        write_features(
            path=features_path,
            gene_ids=gene_ids,
            selected_genes=selected_genes,
            var=var,
            gene_id_column=gene_id_column,
            feature_type=feature_type,
            feature_region=feature_region,
        )

        print(f"[write] {matrix_path}")
        print(f"[write] {ntr_path}")

        with open(matrix_path, "w", encoding="utf-8") as matrix_file:
            with open(ntr_path, "w", encoding="utf-8") as ntr_file:
                write_matrix_market_header(
                    file=matrix_file,
                    n_rows=matrix_rows,
                    n_cols=matrix_cols,
                    nnz=activation_nnz,
                )
                write_matrix_market_header(
                    file=ntr_file,
                    n_rows=matrix_rows,
                    n_cols=matrix_cols,
                    nnz=ratio_nnz,
                )

                for row in tqdm(range(n_cells), desc="Writing MatrixMarket files"):
                    activation_cols, activation_values, ratio_cols, ratio_values = prepare_row_outputs(
                        row=row,
                        activation_reader=activation_reader,
                        new_reader=new_reader,
                        gene_map=gene_map,
                        clip_ratio=clip_ratio,
                    )

                    write_matrix_market_entries(
                        file=matrix_file,
                        cell_index=row,
                        gene_cols=activation_cols,
                        values=activation_values,
                        orientation=orientation,
                    )

                    write_matrix_market_entries(
                        file=ntr_file,
                        cell_index=row,
                        gene_cols=ratio_cols,
                        values=ratio_values,
                        orientation=orientation,
                    )

    print("\n[done] Created MatrixMarket-style dataset:")
    print(f"  expression_matrix_path = {matrix_path}")
    print(f"  ntr_matrix_path        = {ntr_path}")
    print(f"  barcodes_path          = {barcodes_path}")
    print(f"  features_path          = {features_path}")


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
    orientation: str,
    sample_column: str,
    timepoint_column: str,
    gene_id_column: Optional[str],
    feature_type: str,
    feature_region: str,
) -> None:
    if dataset not in FILES:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(FILES)}")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "matrix.mtx"
    ntr_path = output_dir / "4sU.Binom.ntr.mtx"
    barcodes_path = output_dir / "barcodes.tsv"
    features_path = output_dir / "features.tsv"

    final_files_exist = (
        matrix_path.exists()
        and ntr_path.exists()
        and barcodes_path.exists()
        and features_path.exists()
    )

    if final_files_exist and not force_process:
        print(f"[skip] Final output files already exist in: {output_dir}")
        print("[skip] Use --force-process to recreate them.")
        return

    if h5ad_path is None:
        filename_gz = FILES[dataset]
        h5ad_path = work_dir / filename_gz.replace(".gz", "")

    if h5ad_path.exists() and not force_download:
        print(f"[skip] Existing H5AD found, no download needed: {h5ad_path}")
    else:
        filename_gz = FILES[dataset]
        url = build_geo_download_url(filename_gz)

        download_and_decompress_gzip(
            url=url,
            output_h5ad=h5ad_path,
            force=force_download,
        )

    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file does not exist: {h5ad_path}")

    write_mtx_outputs(
        h5ad_path=h5ad_path,
        output_dir=output_dir,
        activation_layer=activation_layer,
        new_layer=new_layer,
        min_cells=min_cells,
        min_gene_nonzero_fraction=min_gene_nonzero_fraction,
        top_genes_by_detection=top_genes_by_detection,
        clip_ratio=clip_ratio,
        orientation=orientation,
        sample_column=sample_column,
        timepoint_column=timepoint_column,
        gene_id_column=gene_id_column,
        feature_type=feature_type,
        feature_region=feature_region,
    )

    if delete_h5ad:
        print(f"[cleanup] Deleting H5AD: {h5ad_path}")
        h5ad_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SCI-FATE2 GSE236512 data as MatrixMarket files: "
            "matrix.mtx, 4sU.Binom.ntr.mtx, barcodes.tsv, features.tsv."
        )
    )

    parser.add_argument("--dataset", choices=list(FILES.keys()), default="estimate")
    parser.add_argument("--work-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/scifate2_mtx"))
    parser.add_argument("--h5ad-path", type=Path, default=None)

    parser.add_argument("--activation-layer", type=str, default="total")
    parser.add_argument("--new-layer", type=str, default="new_estimated")

    parser.add_argument(
        "--min-cells",
        type=int,
        default=0,
        help="Absolute minimum number of cells in which a gene must be non-zero.",
    )
    parser.add_argument(
        "--min-gene-nonzero-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of cells in which a gene must be non-zero. "
            "Example: 0.05 means at least 5 percent of cells."
        ),
    )
    parser.add_argument(
        "--top-genes-by-detection",
        type=int,
        default=10000,
        help=(
            "Keep only the top N genes by detection frequency after filtering. "
            "Use 0 to keep all genes passing the filters."
        ),
    )

    parser.add_argument(
        "--orientation",
        choices=["genes-by-cells", "cells-by-genes"],
        default="cells-by-genes",
        help=(
            "Matrix orientation. Default is cells-by-genes because your loader expects "
            "n_cells x n_genes and this avoids an expensive transpose."
        ),
    )

    parser.add_argument("--sample-column", type=str, default="sample")
    parser.add_argument("--timepoint-column", type=str, default="timepoint")

    parser.add_argument(
        "--gene-id-column",
        type=str,
        default=None,
        help="Optional column in adata.var used as gene ID. If missing, gene name is used.",
    )
    parser.add_argument("--feature-type", type=str, default="Gene Expression")
    parser.add_argument("--feature-region", type=str, default="Exonic (mgi)")

    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--delete-h5ad", action="store_true")
    parser.add_argument("--clip-ratio", action="store_true")

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
        orientation=args.orientation,
        sample_column=args.sample_column,
        timepoint_column=args.timepoint_column,
        gene_id_column=args.gene_id_column,
        feature_type=args.feature_type,
        feature_region=args.feature_region,
    )


if __name__ == "__main__":
    main()