# std-lib imports
from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

# 3 party imports
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

# package imports


AUTHORS_ANNOTATION_PIPELINE_VERSION = 3
AUTHORS_EXPECTED_CELLS = 47_243
AUTHORS_EXPECTED_GENES = 2_137
AUTHORS_CLUSTER_MIN_SUPPORT = 0.50
AUTHORS_CLUSTER_MIN_MARGIN = 0.10

# Exact marker-gene order of Maizels2023aa/data/celltype_knowledge_matrix.csv.
AUTHORS_KNOWLEDGE_GENES = (
    "Foxa2", "Shh", "Arx", "Sox2", "T", "Fgf8", "Lef1", "Nkx1-2", "Pax6", "Irx3", "Olig2", "Nkx2-2", "Nkx6-1",
    "Neurog2", "Isl1", "Sim1", "Elavl3", "Tubb3", "Stmn2", "Map2", "Mnx1", "Slc10a4", "Slc18a3", "Foxc2", "Twist1",
    "Meox1", "Meox2", "Tbx6",
)

# Each tuple is (authors' basic-classification name, robust canonical label, genes expected >= 1 UMI).
# Row order is significant because the authors used np.argmin(), so ties resolve to the first matching row.
AUTHORS_KNOWLEDGE_TEMPLATES = (
    ("NMP", "NMP", ("Sox2", "T", "Fgf8", "Lef1")),
    ("Neural", "Early_Neural", ("Sox2", "Nkx1-2")),
    ("Neural", "Neural", ("Sox2", "Pax6", "Irx3")),
    ("Floor plate", "FP", ("Foxa2", "Shh", "Arx", "Sox2")),
    ("pMN", "pMN", ("Sox2", "Olig2", "Nkx6-1")),
    ("p3", "p3", ("Sox2", "Nkx2-2", "Nkx6-1")),
    ("pMN", "pMN", ("Sox2", "Olig2", "Neurog2", "Isl1")),
    ("p3", "p3", ("Sox2", "Nkx2-2")),
    ("V3 Interneuron", "V3", ("Nkx2-2", "Sim1", "Elavl3", "Tubb3", "Stmn2", "Map2")),
    ("V3 Interneuron", "V3", ("Nkx2-2", "Sim1", "Elavl3", "Tubb3", "Stmn2", "Map2")),
    ("Mesoderm", "Mesoderm", ("Foxc2", "Tbx6")),
    ("Mesoderm", "Mesoderm", ("Foxc2", "Twist1", "Meox1", "Meox2")),
    ("Motor neuron", "MN", ("Isl1", "Elavl3", "Tubb3", "Stmn2", "Map2")),
    ("Motor neuron", "MN", ("Isl1", "Elavl3", "Tubb3", "Stmn2", "Map2", "Mnx1", "Slc10a4")),
    ("Motor neuron", "MN", ("Elavl3", "Tubb3", "Stmn2", "Map2", "Slc10a4", "Slc18a3")),
    ("Motor neuron", "MN", ("Elavl3", "Tubb3", "Stmn2", "Map2", "Slc18a3")),
    ("Motor neuron", "MN", ("Isl1", "Elavl3", "Tubb3", "Stmn2", "Map2", "Mnx1", "Slc10a4", "Slc18a3")),
    ("null_progenitor", "other", ("Sox2",)),
    ("null_neuron", "other", ("Elavl3", "Tubb3", "Stmn2", "Map2")),
    ("null_cell", "other", ()),
    ("null_mixed1", "other", ("Elavl3", "Tubb3", "Stmn2", "Map2", "Foxc2", "Twist1", "Meox1", "Meox2", "Tbx6")),
    ("null_mixed2", "other", ("T", "Fgf8", "Lef1", "Elavl3", "Tubb3", "Stmn2", "Map2")),
)

# Gene programs scored by the authors while manually inspecting Leiden clusters in A1.2.
AUTHORS_MARKER_PROGRAMS = {
    "NMP": ("Sox2", "T", "Fgf8", "Lef1"),
    "Early_Neural": ("Sox2", "Nkx1-2", "Pax6", "Irx3", "Sema3e"),
    "FP": ("Shh", "Arx", "Foxa2"),
    "pMN": ("Sox2", "Olig2", "Nkx6-1"),
    "p3": ("Sox2", "Nkx2-2", "Nkx6-1"),
    "MN": ("Neurog2", "Isl1", "Elavl3", "Tubb3", "Stmn2", "Map2", "Mnx1", "Slc10a4", "Slc18a3"),
    "V3": ("Sim1", "Elavl3", "Tubb3", "Stmn2", "Map2"),
    "Mesoderm": ("Rspo3", "Foxc2", "Twist1", "Meox1", "Meox2", "Tbx6"),
}


# ---------------------------------------------------------------------
# Dependencies / low-memory H5AD access
# ---------------------------------------------------------------------


def _annotation_dependencies():
    try:
        import anndata as ad
        import scanpy as sc
        import scrublet as scr
    except ImportError as error:
        raise ImportError(
            "Reconstructing SCI-FATE2 marker annotations requires anndata, scanpy, scrublet, scikit-misc, igraph and "
            "leidenalg; install trajectoryflow[annotations]."
        ) from error
    return ad, sc, scr


def _encoding_type(node) -> str:
    value = node.attrs.get("encoding-type", "")
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class _BackedLayerReader:
    """Read contiguous row blocks from an H5AD layer without materializing the full matrix."""

    def __init__(self, h5: h5py.File, layer: str):
        if "layers" not in h5 or layer not in h5["layers"]:
            raise KeyError(f"Authors' annotation workflow requires layer {layer!r} in the counting H5AD.")
        self.layer, self.node = layer, h5["layers"][layer]
        self.encoding = _encoding_type(self.node)
        if isinstance(self.node, h5py.Group):
            if self.encoding != "csr_matrix":
                raise ValueError(
                    f"Layer {layer!r} uses {self.encoding!r}; low-memory annotation reconstruction currently requires "
                    "CSR or dense H5AD layers."
                )
            self.data, self.indices, self.indptr = self.node["data"], self.node["indices"], self.node["indptr"]
            self.shape = tuple(int(value) for value in self.node.attrs["shape"])
        elif isinstance(self.node, h5py.Dataset):
            self.shape = tuple(int(value) for value in self.node.shape)
        else:
            raise TypeError(f"Unsupported H5AD storage for layer {layer!r}.")

    def rows(self, start: int, end: int, columns: np.ndarray | None = None) -> sparse.csr_matrix:
        if not 0 <= start <= end <= self.shape[0]:
            raise IndexError(f"Invalid row block [{start}:{end}] for layer {self.layer!r} with {self.shape[0]} rows.")
        if isinstance(self.node, h5py.Dataset):
            matrix = sparse.csr_matrix(np.asarray(self.node[start:end], dtype=np.float32))
        else:
            data_start, data_end = int(self.indptr[start]), int(self.indptr[end])
            data = np.asarray(self.data[data_start:data_end], dtype=np.float32)
            indices = np.asarray(self.indices[data_start:data_end], dtype=np.int32)
            indptr = np.asarray(self.indptr[start:end + 1], dtype=np.int64) - data_start
            matrix = sparse.csr_matrix((data, indices, indptr), shape=(end - start, self.shape[1]), dtype=np.float32)
        if columns is not None:
            matrix = matrix[:, columns]
        matrix.eliminate_zeros()
        return matrix


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("authors annotation chunk_size must be a positive integer.")


def _validate_counts(matrix: sparse.csr_matrix, layer: str) -> None:
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"Non-finite values found in counting layer {layer!r}.")
    if (matrix.data < 0).any():
        raise ValueError(f"Negative values found in counting layer {layer!r}.")


def _stream_qc_metrics(
    total_reader: _BackedLayerReader,
    new_reader: _BackedLayerReader,
    mitochondrial: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the authors' QC statistics with bounded memory."""
    n_cells = total_reader.shape[0]
    total_counts, n_genes = np.empty(n_cells, dtype=np.float64), np.empty(n_cells, dtype=np.int32)
    pct_mt, label_rate = np.empty(n_cells, dtype=np.float64), np.empty(n_cells, dtype=np.float64)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        total, new = total_reader.rows(start, end), new_reader.rows(start, end)
        _validate_counts(total, "total")
        _validate_counts(new, "new")
        total_sum = np.asarray(total.sum(axis=1)).ravel().astype(np.float64, copy=False)
        new_sum = np.asarray(new.sum(axis=1)).ravel().astype(np.float64, copy=False)
        mt_sum = np.asarray(total[:, mitochondrial].sum(axis=1)).ravel() if mitochondrial.any() else np.zeros(end - start)
        total_counts[start:end] = total_sum
        n_genes[start:end] = total.getnnz(axis=1)
        pct_mt[start:end] = np.divide(100.0 * mt_sum, total_sum, out=np.zeros(end - start), where=total_sum != 0)
        label_rate[start:end] = np.divide(new_sum, total_sum, out=np.zeros(end - start), where=total_sum != 0)
    return total_counts, n_genes, pct_mt, label_rate


def _collect_rows(
    reader: _BackedLayerReader,
    keep: np.ndarray,
    chunk_size: int,
    columns: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Materialize only selected rows (and optionally columns) from a backed layer."""
    if keep.dtype != bool or keep.shape != (reader.shape[0],):
        raise ValueError("keep must be a boolean mask matching the source cell count.")
    n_columns = reader.shape[1] if columns is None else len(columns)
    parts = []
    for start in range(0, reader.shape[0], chunk_size):
        end = min(start + chunk_size, reader.shape[0])
        local = np.flatnonzero(keep[start:end])
        if len(local):
            parts.append(reader.rows(start, end, columns=columns)[local])
    return sparse.vstack(parts, format="csr").astype(np.float32, copy=False) if parts else sparse.csr_matrix((0, n_columns), dtype=np.float32)


# ---------------------------------------------------------------------
# Authors' preprocessing + marker annotation
# ---------------------------------------------------------------------


def _replicate_hvgs(matrix: sparse.csr_matrix, gene_names: np.ndarray, ad, sc) -> set[str]:
    adata = ad.AnnData(X=matrix, var=pd.DataFrame(index=gene_names))
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", subset=False)
    return set(adata.var_names[adata.var["highly_variable"]].astype(str))


def _normalize_log_selected(matrix: sparse.csr_matrix, total_counts: np.ndarray) -> sparse.csr_matrix:
    """Match normalize_total(1e4) -> log1p before gene subsetting without loading all genes."""
    if (total_counts <= 0).any():
        raise ValueError("Authors' retained cells must have positive total counts.")
    matrix = matrix.multiply((1e4 / total_counts).astype(np.float32)[:, None]).tocsr()
    np.log1p(matrix.data, out=matrix.data)
    return matrix


def authors_qc_mask(obs: pd.DataFrame) -> np.ndarray:
    """Reproduce the percentile QC filters from Maizels et al. notebook A1.0."""
    required = ("total_counts", "n_genes_by_counts", "label_rate", "pct_counts_mt")
    missing = [column for column in required if column not in obs]
    if missing:
        raise KeyError(f"Missing authors' QC columns: {missing}.")
    total_lo, total_hi = np.percentile(obs["total_counts"], [10, 90])
    genes_lo, genes_hi = np.percentile(obs["n_genes_by_counts"], [10, 90])
    label_lo, mt_hi = np.percentile(obs["label_rate"], 10), np.percentile(obs["pct_counts_mt"], 90)
    return (
        (obs["total_counts"].to_numpy() > total_lo) & (obs["total_counts"].to_numpy() < total_hi)
        & (obs["n_genes_by_counts"].to_numpy() > genes_lo) & (obs["n_genes_by_counts"].to_numpy() < genes_hi)
        & (obs["label_rate"].to_numpy() > label_lo) & (obs["pct_counts_mt"].to_numpy() < mt_hi)
    )


def _template_matrix() -> np.ndarray:
    gene_to_index = {gene: index for index, gene in enumerate(AUTHORS_KNOWLEDGE_GENES)}
    matrix = np.zeros((len(AUTHORS_KNOWLEDGE_TEMPLATES), len(AUTHORS_KNOWLEDGE_GENES)), dtype=np.int16)
    for row, (_, _, positive_genes) in enumerate(AUTHORS_KNOWLEDGE_TEMPLATES):
        matrix[row, [gene_to_index[gene] for gene in positive_genes]] = 1
    return matrix


def classify_authors_markers(marker_counts: sparse.spmatrix | np.ndarray) -> pd.DataFrame:
    """Vectorized reproduction of Maizels et al. ``basic_classification``.

    The publication notebook binarizes each of the 28 marker genes at one UMI,
    computes Euclidean distance to every binary row of ``celltype_knowledge_matrix.csv``,
    and takes ``argmin``. Since all values are binary, minimizing Euclidean distance
    is exactly equivalent to minimizing Hamming distance, which we compute here
    without constructing a cell × template × gene tensor. Tie-breaking remains the
    authors' original first-row ``argmin`` behavior.
    """
    counts = marker_counts.toarray() if sparse.issparse(marker_counts) else np.asarray(marker_counts)
    if counts.ndim != 2 or counts.shape[1] != len(AUTHORS_KNOWLEDGE_GENES):
        raise ValueError(f"marker_counts must have shape [n_cells, {len(AUTHORS_KNOWLEDGE_GENES)}].")
    expressed = (counts >= 1).astype(np.int16, copy=False)
    templates = _template_matrix()
    distances = expressed.sum(axis=1, dtype=np.int16)[:, None] + templates.sum(axis=1)[None, :] - 2 * (expressed @ templates.T)
    winners = np.argmin(distances, axis=1)
    nearest_two = np.partition(distances, kth=1, axis=1)[:, :2] if distances.shape[1] > 1 else np.column_stack([distances[:, 0], distances[:, 0]])
    raw_names = np.asarray([name.replace(" ", "") for name, _, _ in AUTHORS_KNOWLEDGE_TEMPLATES], dtype=object)
    canonical = np.asarray([cell_type for _, cell_type, _ in AUTHORS_KNOWLEDGE_TEMPLATES], dtype=object)
    return pd.DataFrame({
        "marker_based_classification": raw_names[winners],
        "marker_based_cell_type": canonical[winners],
        "marker_distance": distances[np.arange(len(winners)), winners].astype(np.int16),
        "marker_distance_margin": (nearest_two[:, 1] - nearest_two[:, 0]).astype(np.int16),
    })


def cluster_consensus_cell_types(
    leiden: pd.Series,
    marker_cell_type: pd.Series,
    min_support: float = AUTHORS_CLUSTER_MIN_SUPPORT,
    min_margin: float = AUTHORS_CLUSTER_MIN_MARGIN,
) -> pd.DataFrame:
    """Assign current Leiden clusters from marker evidence without relying on cluster IDs."""
    if len(leiden) != len(marker_cell_type):
        raise ValueError("leiden and marker_cell_type must have the same length.")
    if not 0 <= min_support <= 1 or not 0 <= min_margin <= 1:
        raise ValueError("min_support and min_margin must lie in [0, 1].")

    clusters = leiden.astype(str).to_numpy()
    labels = marker_cell_type.astype(str).to_numpy()
    assigned = np.empty(len(labels), dtype=object)
    confidence, margin = np.empty(len(labels), dtype=np.float64), np.empty(len(labels), dtype=np.float64)
    for cluster in pd.unique(clusters):
        mask = clusters == cluster
        fractions = pd.Series(labels[mask]).value_counts(normalize=True)
        top_label, top_support = str(fractions.index[0]), float(fractions.iloc[0])
        second_support = float(fractions.iloc[1]) if len(fractions) > 1 else 0.0
        top_margin = top_support - second_support
        final_label = top_label if top_label != "other" and top_support >= min_support and top_margin >= min_margin else "other"
        assigned[mask], confidence[mask], margin[mask] = final_label, top_support, top_margin
    return pd.DataFrame({"cell_type": assigned, "cell_type_support": confidence, "cell_type_margin": margin})


def _score_authors_programs(adata, sc) -> None:
    params = inspect.signature(sc.tl.score_genes).parameters
    for cell_type, markers in AUTHORS_MARKER_PROGRAMS.items():
        present = [gene for gene in markers if gene in adata.var_names]
        if not present:
            warnings.warn(f"No {cell_type} marker-program genes are present; skipping its score.", RuntimeWarning, stacklevel=2)
            continue
        kwargs = {
            "gene_list": present, "ctrl_size": 100, "n_bins": 25, "score_name": f"{cell_type}_score",
            "copy": False, "use_raw": False,
        }
        kwargs["random_state" if "random_state" in params else "rng"] = 0
        if "ctrl_as_ref" in params:
            kwargs["ctrl_as_ref"] = True  # Preserve Scanpy v1 behavior explicitly; Scanpy 2 changes this default.
        sc.tl.score_genes(adata, **kwargs)


def reconstruct_authors_cell_annotations(counting_h5ad: Path, chunk_size: int = 2048) -> pd.DataFrame:
    """Reconstruct robust SCI-FATE2 cell-type annotations from the authors' published marker logic.

    Provenance
    ----------
    This implementation is adapted from Rory Maizels et al.'s public SCI-FATE2 /
    Velvet analysis repository ``rorymaizels/Maizels2023aa``. The two relevant
    notebooks are:

    * ``analysis/A1.0_QC_and_filtering.ipynb`` for percentile QC and Scrublet.
    * ``analysis/A1.2_cell_type_classification.ipynb`` for gene selection,
      PCA/neighbors/Leiden, ``basic_classification``, marker-program scores, and
      the biological cell-type inspection workflow.

    In A1.2, ``basic_classification`` reads
    ``data/celltype_knowledge_matrix.csv``, thresholds each marker at >= 1 UMI,
    computes the Euclidean distance between that binary expression vector and
    every binary knowledge-matrix row, and chooses the nearest row. The exact 28
    genes and all 22 binary templates from that matrix are reproduced above.
    The authors also score eight marker programs with ``scanpy.tl.score_genes``;
    those programs are reproduced here with their published parameters and are
    saved in the annotation table for plotting/inspection.

    Why this does NOT reuse the paper's hard-coded Leiden mapping
    --------------------------------------------------------------
    The notebook's final ``cell_annotation`` was created manually after inspecting
    each *specific* Leiden cluster and then applying a dictionary such as
    ``'0' -> 'Mesoderm'`` and ``'1' -> 'NMP'``. Leiden cluster numbers are not
    biological identifiers: dependency versions, floating-point changes, graph
    construction, or the Leiden backend can change both cluster count and numeric
    labels. Applying that old dictionary to a new clustering can therefore assign
    confidently wrong biology. This function intentionally never maps by numeric
    Leiden ID.

    Robust adaptation used here
    ---------------------------
    Current single-cell annotation guidance recommends using marker evidence at
    the level of groups of transcriptionally similar cells rather than relying on
    single-cell marker thresholds alone, because scRNA-seq dropout makes individual
    marker detection noisy. We therefore:

    1. Reproduce the authors' QC, per-replicate Scrublet, stratified Seurat-v3 HVG
       selection, normalization, PCA, 10-neighbor graph, and Leiden resolution 1.6.
    2. Reproduce their per-cell binary ``basic_classification`` exactly (vectorized
       for speed and memory efficiency) and preserve it as
       ``marker_based_classification``.
    3. Convert each winning marker template to a canonical biological label, then
       label each *current* Leiden cluster by marker consensus. A cluster receives
       a biological label only when >= 50% of its cells support the same label and
       that support exceeds the runner-up by >= 10 percentage points; otherwise it
       is ``other``. ``cell_type_support`` and ``cell_type_margin`` preserve the
       evidence instead of hiding ambiguity.
    4. Keep the authors' marker-program scores as diagnostic columns. They are not
       treated as calibrated probabilities and are not blindly ``argmax``-ed.

    The two rows called ``Neural`` in the original binary knowledge matrix encode
    different developmental states. We preserve the authors' raw classification
    name, but refine the Sox2+Nkx1-2 template to ``Early_Neural`` and the
    Sox2+Pax6+Irx3 template to ``Neural`` for the canonical label. This matches the
    known NMP-to-neural progression in which Nkx1-2 is retained in pre-neural cells
    and later downregulated as Pax6/Irx3 are induced. This refinement is an explicit
    TrajectoryFlow adaptation; it is not claimed to be the authors' manually curated
    final ``cell_annotation``.

    Memory behavior
    ---------------
    The counting H5AD stays backed on disk. QC is streamed in ``chunk_size`` row
    blocks, Scrublet/HVG selection load one replicate at a time, marker
    classification loads only 28 columns, and PCA/Leiden materialize only the
    final ~2k selected genes. The authors reported 47,243 cells x 2,137 genes;
    modern dependency versions can differ slightly, so these values are diagnostic
    warnings rather than correctness gates now that no fixed cluster-ID mapping is
    used.

    References
    ----------
    Authors' notebook:
    https://github.com/rorymaizels/Maizels2023aa/blob/main/analysis/A1.2_cell_type_classification.ipynb

    Single-cell annotation best practices:
    https://www.sc-best-practices.org/cellular-structure/annotation/

    Scanpy marker/cluster annotation tutorial:
    https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html
    """
    _validate_chunk_size(chunk_size)
    ad, sc, scr = _annotation_dependencies()
    print(f"[authors-cell-types] Reading metadata in backed mode: {counting_h5ad}")
    backed = ad.read_h5ad(counting_h5ad, backed="r")
    try:
        if "rep" not in backed.obs:
            raise KeyError("Authors' annotation workflow requires obs['rep'].")
        missing_layers = [layer for layer in ("total", "new") if layer not in backed.layers]
        if missing_layers:
            raise KeyError(f"Authors' annotation workflow requires counting layers {missing_layers}.")
        obs = backed.obs.copy()
        cell_ids = backed.obs_names.astype(str).to_numpy()
        gene_names = backed.var_names.astype(str).to_numpy()
    finally:
        backed.file.close()

    missing_markers = sorted(set(AUTHORS_KNOWLEDGE_GENES) - set(gene_names))
    if missing_markers:
        raise KeyError(f"Counting H5AD is missing authors' knowledge-matrix marker genes: {missing_markers}.")

    with h5py.File(counting_h5ad, "r") as h5:
        total_reader, new_reader = _BackedLayerReader(h5, "total"), _BackedLayerReader(h5, "new")
        if total_reader.shape != new_reader.shape or total_reader.shape != (len(obs), len(gene_names)):
            raise ValueError("Counting H5AD total/new layer shapes do not match obs/var metadata.")

        print(f"[authors-cell-types] Streaming QC metrics in chunks of {chunk_size:,} cells...")
        mitochondrial = np.char.startswith(gene_names.astype(str), "mt")
        total_counts, n_genes, pct_mt, label_rate = _stream_qc_metrics(total_reader, new_reader, mitochondrial, chunk_size)
        obs["total_counts"], obs["n_genes_by_counts"] = total_counts, n_genes
        obs["pct_counts_mt"], obs["label_rate"] = pct_mt, label_rate
        qc_mask = authors_qc_mask(obs)
        print(f"[authors-cell-types] QC retained {int(qc_mask.sum()):,}/{len(obs):,} cells.")

        replicates = obs["rep"].astype(str).to_numpy()
        doublet_scores = np.full(len(obs), np.nan, dtype=np.float64)
        gene_sets = []
        for replicate in ("r1", "r2", "r3", "r4"):
            replicate_mask = qc_mask & (replicates == replicate)
            if not replicate_mask.any():
                raise ValueError(f"Authors' annotation workflow expects replicate {replicate!r}.")
            print(f"[authors-cell-types] Loading QC-passed {replicate} cells for Scrublet...")
            counts = _collect_rows(total_reader, replicate_mask, chunk_size)
            scrub = scr.Scrublet(counts, expected_doublet_rate=0.06, random_state=0)
            scores, _ = scrub.scrub_doublets()
            global_rows = np.flatnonzero(replicate_mask)
            doublet_scores[global_rows] = scores
            singlets = scores < 0.3
            print(f"[authors-cell-types] {replicate}: {int(singlets.sum()):,}/{len(scores):,} cells after doublet filtering.")
            if replicate != "r4":
                gene_sets.append(_replicate_hvgs(counts[singlets], gene_names, ad, sc))
            del counts, scrub, scores

        final_mask = qc_mask & np.isfinite(doublet_scores) & (doublet_scores < 0.3)
        selected_set = set.intersection(*gene_sets).union(AUTHORS_KNOWLEDGE_GENES).intersection(gene_names)
        selected_indices = np.asarray([i for i, gene in enumerate(gene_names) if gene in selected_set], dtype=np.int64)
        selected_genes = gene_names[selected_indices]
        if int(final_mask.sum()) != AUTHORS_EXPECTED_CELLS:
            warnings.warn(
                f"Authors reported {AUTHORS_EXPECTED_CELLS:,} retained cells; this environment produced {int(final_mask.sum()):,}. "
                "Continuing because annotation no longer depends on matching published Leiden IDs.", RuntimeWarning, stacklevel=2,
            )
        if len(selected_genes) != AUTHORS_EXPECTED_GENES:
            warnings.warn(
                f"Authors reported {AUTHORS_EXPECTED_GENES:,} selected genes; this environment produced {len(selected_genes):,}. "
                "Continuing because annotation no longer depends on matching published Leiden IDs.", RuntimeWarning, stacklevel=2,
            )

        final_rows = np.flatnonzero(final_mask)
        marker_indices = np.asarray([np.flatnonzero(gene_names == gene)[0] for gene in AUTHORS_KNOWLEDGE_GENES], dtype=np.int64)
        marker_calls = classify_authors_markers(_collect_rows(total_reader, final_mask, chunk_size, columns=marker_indices))
        print(f"[authors-cell-types] Loading final {len(final_rows):,} x {len(selected_genes):,} matrix for PCA/Leiden...")
        cluster_matrix = _collect_rows(total_reader, final_mask, chunk_size, columns=selected_indices)

    cluster_matrix = _normalize_log_selected(cluster_matrix, total_counts[final_rows])
    cluster_obs = obs.iloc[final_rows].copy()
    cluster_obs["doublet_score"] = doublet_scores[final_rows]
    cluster_obs.index = pd.Index(cell_ids[final_rows])
    adata = ad.AnnData(X=cluster_matrix, obs=cluster_obs, var=pd.DataFrame(index=selected_genes))
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack", dtype="float32")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50, random_state=0)
    random_state = np.random.get_state()
    try:
        np.random.seed(0)
        sc.tl.leiden(adata, resolution=1.6, random_state=0, flavor="leidenalg")
    finally:
        np.random.set_state(random_state)
    _score_authors_programs(adata, sc)

    marker_calls.index = adata.obs_names
    consensus = cluster_consensus_cell_types(adata.obs["leiden"], marker_calls["marker_based_cell_type"])
    consensus.index = adata.obs_names
    result = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "authors_leiden": adata.obs["leiden"].astype(str).to_numpy(),
        "marker_based_classification": marker_calls["marker_based_classification"].to_numpy(),
        "marker_based_cell_type": marker_calls["marker_based_cell_type"].to_numpy(),
        "marker_distance": marker_calls["marker_distance"].to_numpy(),
        "marker_distance_margin": marker_calls["marker_distance_margin"].to_numpy(),
        "cell_annotation": consensus["cell_type"].to_numpy(),
        "cell_type": consensus["cell_type"].to_numpy(),
        "cell_type_support": consensus["cell_type_support"].to_numpy(),
        "cell_type_margin": consensus["cell_type_margin"].to_numpy(),
        "annotation_method": "maizels_marker_cluster_consensus_v3",
        "authors_qc_pass": True,
        "doublet_score": adata.obs["doublet_score"].to_numpy(),
        "total_counts": adata.obs["total_counts"].to_numpy(),
        "n_genes_by_counts": adata.obs["n_genes_by_counts"].to_numpy(),
        "pct_counts_mt": adata.obs["pct_counts_mt"].to_numpy(),
        "label_rate": adata.obs["label_rate"].to_numpy(),
    })
    for score_name in (f"{cell_type}_score" for cell_type in AUTHORS_MARKER_PROGRAMS):
        if score_name in adata.obs:
            result[score_name] = adata.obs[score_name].to_numpy()
    print(
        f"[authors-cell-types] Annotated {len(result):,} cells across {result['cell_type'].nunique()} robust labels "
        f"({adata.obs['leiden'].nunique()} current Leiden clusters)."
    )
    return result


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------


def _cache_signature(counting_h5ad: Path) -> dict:
    stat = counting_h5ad.stat()
    return {
        "pipeline_version": AUTHORS_ANNOTATION_PIPELINE_VERSION,
        "source_name": counting_h5ad.name,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def load_or_reconstruct_authors_cell_annotations(
    counting_h5ad: Path,
    cache_csv: Path,
    force: bool = False,
    chunk_size: int = 2048,
) -> Path:
    """Return cached marker-derived SCI-FATE2 annotations, rebuilding with bounded memory when needed."""
    metadata_path = cache_csv.with_suffix(".json")
    signature = _cache_signature(counting_h5ad)
    if cache_csv.exists() and metadata_path.exists() and not force:
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if cached == signature:
            print(f"[authors-cell-types] Reusing annotation cache: {cache_csv}")
            return cache_csv

    annotations = reconstruct_authors_cell_annotations(counting_h5ad, chunk_size=chunk_size)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv, tmp_json = cache_csv.with_suffix(".csv.tmp"), metadata_path.with_suffix(".json.tmp")
    annotations.to_csv(tmp_csv, index=False)
    tmp_json.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    tmp_csv.replace(cache_csv)
    tmp_json.replace(metadata_path)
    print(f"[authors-cell-types] Saved annotation cache: {cache_csv}")
    return cache_csv
