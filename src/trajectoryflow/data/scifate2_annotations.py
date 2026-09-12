# std-lib imports
from __future__ import annotations

import json
from pathlib import Path

# 3 party imports
import numpy as np
import pandas as pd

# package imports


AUTHORS_ANNOTATION_PIPELINE_VERSION = 1
AUTHORS_EXPECTED_CELLS = 47_243
AUTHORS_EXPECTED_GENES = 2_137
AUTHORS_CURATED_GENES = (
    "Foxa2", "Shh", "Arx", "Sox2", "T", "Fgf8", "Lef1", "Nkx1-2", "Pax6", "Irx3", "Olig2", "Nkx2-2", "Nkx6-1",
    "Neurog2", "Isl1", "Sim1", "Elavl3", "Tubb3", "Stmn2", "Map2", "Mnx1", "Slc10a4", "Slc18a3", "Foxc2", "Twist1",
    "Meox1", "Meox2", "Tbx6",
)
AUTHORS_CLUSTER_ANNOTATIONS = {
    "0": "Mesoderm", "1": "NMP", "2": "Neural", "3": "NMP", "4": "Early_Neural", "5": "Neural", "6": "pMN",
    "7": "Mesoderm", "8": "pMN", "9": "NMP", "10": "p3", "11": "Mesoderm", "12": "Mesoderm", "13": "V3", "14": "FP",
    "15": "Early_Neural", "16": "p3", "17": "MN", "18": "pMN", "19": "pMN", "20": "MN", "21": "other", "22": "V3",
    "23": "Mesoderm", "24": "V3", "25": "MN", "26": "other", "27": "other", "28": "Neural", "29": "other", "30": "other",
    "31": "other", "32": "other",
}


def _annotation_dependencies():
    try:
        import anndata as ad
        import scanpy as sc
        import scrublet as scr
    except ImportError as error:
        raise ImportError(
            "Reconstructing the published SCI-FATE2 cell annotations requires anndata, scanpy, scrublet, scikit-misc, "
            "igraph and leidenalg; install trajectoryflow[annotations]."
        ) from error
    return ad, sc, scr


def authors_qc_mask(obs: pd.DataFrame) -> np.ndarray:
    """Reproduce the percentile QC filters from Maizels et al. notebook A1.0."""
    required = ("total_counts", "n_genes_by_counts", "label_rate", "pct_counts_mt")
    missing = [column for column in required if column not in obs]
    if missing:
        raise KeyError(f"Missing authors' QC columns: {missing}.")
    total_lo, total_hi = np.percentile(obs["total_counts"], [10, 90])
    genes_lo, genes_hi = np.percentile(obs["n_genes_by_counts"], [10, 90])
    label_lo = np.percentile(obs["label_rate"], 10)
    mt_hi = np.percentile(obs["pct_counts_mt"], 90)
    return (
        (obs["total_counts"].to_numpy() > total_lo) & (obs["total_counts"].to_numpy() < total_hi)
        & (obs["n_genes_by_counts"].to_numpy() > genes_lo) & (obs["n_genes_by_counts"].to_numpy() < genes_hi)
        & (obs["label_rate"].to_numpy() > label_lo) & (obs["pct_counts_mt"].to_numpy() < mt_hi)
    )


def map_authors_leiden(leiden: pd.Series, strict: bool = True) -> pd.Series:
    labels = leiden.astype(str)
    unknown = sorted(set(labels) - set(AUTHORS_CLUSTER_ANNOTATIONS))
    if unknown and strict:
        raise RuntimeError(
            "The reconstructed Leiden clustering does not match the authors' published 33-cluster solution; "
            f"unexpected cluster labels: {unknown}. Do not apply the published cluster mapping to this run."
        )
    return labels.map(AUTHORS_CLUSTER_ANNOTATIONS)


def _select_authors_genes(adata, sc) -> list[str]:
    """Reimplement velvetvae.pp.select_genes as used in A1.2, without depending on velvetvae."""
    source = adata[adata.obs["rep"].astype(str) != "r4"]
    gene_sets = []
    for replicate in source.obs["rep"].unique():
        subset = source[source.obs["rep"] == replicate].copy()
        sc.pp.highly_variable_genes(subset, n_top_genes=3000, flavor="seurat_v3", subset=True)
        gene_sets.append(set(subset.var_names.astype(str)))
    if not gene_sets:
        raise ValueError("No non-r4 cells are available for the authors' stratified gene selection.")
    selected = set.intersection(*gene_sets).union(AUTHORS_CURATED_GENES).intersection(adata.var_names.astype(str))
    return [gene for gene in adata.var_names.astype(str) if gene in selected]


def reconstruct_authors_cell_annotations(counting_h5ad: Path, strict: bool = True) -> pd.DataFrame:
    """Reproduce A1.0 + A1.2 and return the authors' final curated cell annotations.

    The published workflow performs QC and Scrublet on the counting dataset, removes
    cells with doublet_score >= 0.3, selects replicate-stratified HVGs, clusters with
    Leiden, then maps the 33 Leiden clusters to manually curated biological labels.
    """
    ad, sc, scr = _annotation_dependencies()
    print(f"[authors-cell-types] Reading counting dataset: {counting_h5ad}")
    adata = ad.read_h5ad(counting_h5ad)
    for column in ("rep",):
        if column not in adata.obs:
            raise KeyError(f"Authors' annotation workflow requires obs[{column!r}].")
    for layer in ("total", "new"):
        if layer not in adata.layers:
            raise KeyError(f"Authors' annotation workflow requires layer {layer!r} in the counting H5AD.")

    # A1.0_QC_and_filtering.ipynb
    adata.X = adata.layers["total"].copy()
    adata.var["mt"] = adata.var_names.astype(str).str.startswith("mt")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    total = np.asarray(adata.layers["total"].sum(axis=1)).ravel()
    new = np.asarray(adata.layers["new"].sum(axis=1)).ravel()
    adata.obs["label_rate"] = np.divide(new, total, out=np.zeros_like(new, dtype=np.float64), where=total != 0)
    adata = adata[authors_qc_mask(adata.obs)].copy()

    scores = pd.Series(index=adata.obs_names, dtype=np.float64)
    for replicate in ("r1", "r2", "r3", "r4"):
        mask = adata.obs["rep"].astype(str) == replicate
        if not mask.any():
            raise ValueError(f"Authors' annotation workflow expects replicate {replicate!r}.")
        counts = adata[mask].layers["total"]
        scrub = scr.Scrublet(counts, expected_doublet_rate=0.06, random_state=0)
        doublet_scores, _ = scrub.scrub_doublets()
        scores.loc[adata.obs_names[mask]] = doublet_scores
    adata.obs["doublet_score"] = scores.loc[adata.obs_names].to_numpy()
    adata = adata[adata.obs["doublet_score"] < 0.3].copy()

    # A1.2_cell_type_classification.ipynb
    genes = _select_authors_genes(adata, sc)
    if strict and len(genes) != AUTHORS_EXPECTED_GENES:
        raise RuntimeError(
            f"Authors' pipeline selected {len(genes)} genes, expected {AUTHORS_EXPECTED_GENES}. "
            "This usually indicates a dependency/version mismatch; refusing to apply fixed Leiden cluster labels."
        )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, genes].copy()
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50, random_state=0)
    random_state = np.random.get_state()
    try:
        np.random.seed(0)
        sc.tl.leiden(adata, resolution=1.6, random_state=0)
    finally:
        np.random.set_state(random_state)

    clusters = set(adata.obs["leiden"].astype(str))
    if strict and len(adata) != AUTHORS_EXPECTED_CELLS:
        raise RuntimeError(
            f"Authors' pipeline retained {len(adata):,} cells, expected {AUTHORS_EXPECTED_CELLS:,}. "
            "Refusing to apply fixed Leiden cluster labels because the preprocessing result differs from the publication."
        )
    if strict and clusters != set(AUTHORS_CLUSTER_ANNOTATIONS):
        raise RuntimeError(
            f"Authors' pipeline produced clusters {sorted(clusters)}, expected 0..32. "
            "Refusing to apply the publication's fixed cluster-to-cell-type mapping."
        )

    cell_annotation = map_authors_leiden(adata.obs["leiden"], strict=strict)
    result = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "authors_leiden": adata.obs["leiden"].astype(str).to_numpy(),
        "cell_annotation": cell_annotation.to_numpy(),
        "cell_type": cell_annotation.to_numpy(),
        "authors_qc_pass": True,
        "doublet_score": adata.obs["doublet_score"].to_numpy(),
        "total_counts": adata.obs["total_counts"].to_numpy(),
        "n_genes_by_counts": adata.obs["n_genes_by_counts"].to_numpy(),
        "pct_counts_mt": adata.obs["pct_counts_mt"].to_numpy(),
        "label_rate": adata.obs["label_rate"].to_numpy(),
    })
    print(f"[authors-cell-types] Reconstructed {len(result):,} curated cell annotations across {result['cell_type'].nunique()} labels.")
    return result


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
    strict: bool = True,
) -> Path:
    """Return a cached CSV of reconstructed published annotations, rebuilding when needed."""
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

    annotations = reconstruct_authors_cell_annotations(counting_h5ad, strict=strict)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv, tmp_json = cache_csv.with_suffix(".csv.tmp"), metadata_path.with_suffix(".json.tmp")
    annotations.to_csv(tmp_csv, index=False)
    tmp_json.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    tmp_csv.replace(cache_csv)
    tmp_json.replace(metadata_path)
    print(f"[authors-cell-types] Saved annotation cache: {cache_csv}")
    return cache_csv
