# std-lib imports
from __future__ import annotations

import hashlib
from pathlib import Path

# 3 party imports
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors

# package imports
from trajectoryflow.experiment.evaluation import (
    VelocityReference,
    save_velocity_reference,
)


TIMEPOINT_COLUMN = "_trajectoryflow_timepoint"

SCIFATE2_PUBLISHED_CBD_TRANSITIONS: tuple[tuple[str, str], ...] = (
    ("Early_Neural", "Neural"),
    ("NMP", "Early_Neural"),
    ("NMP", "Mesoderm"),
    ("Neural", "pMN"),
    ("pMN", "MN"),
    ("pMN", "p3"),
    ("p3", "V3"),
)


def _cell_ids(obs: pd.DataFrame, timepoint: str) -> np.ndarray:
    for column in ("cell_id", "barcode", "cell"):
        if column in obs.columns:
            values = obs[column].astype(str).to_numpy()
            if len(np.unique(values)) == len(values):
                return values

    index = obs.index.astype(str).to_numpy()
    if len(np.unique(index)) == len(index):
        return index

    return np.asarray(
        [f"{timepoint}:{i}" for i in range(len(obs))],
        dtype=str,
    )


def _gene_names(store) -> np.ndarray:
    if hasattr(store, "gene_names"):
        values = np.asarray(store.gene_names).astype(str)
        if values.shape == (store.n_genes,):
            return values

    genes = store.genes

    if isinstance(genes, pd.DataFrame):
        for column in ("gene", "gene_name", "symbol", "gene_id", "feature_name"):
            if column in genes.columns:
                values = genes[column].astype(str).to_numpy()
                if values.shape == (store.n_genes,):
                    return values

        values = genes.index.astype(str).to_numpy()
        if values.shape == (store.n_genes,):
            return values

    values = np.asarray(genes).astype(str)
    if values.shape == (store.n_genes,):
        return values

    raise ValueError("Could not determine gene names from ScifateStore.genes.")


def _sample_indices(n_cells: int, max_cells: int | None, seed: int) -> np.ndarray:
    if max_cells is None or max_cells <= 0 or n_cells <= max_cells:
        return np.arange(n_cells, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_cells, size=max_cells, replace=False))


def store_to_anndata(
    data_root: str | Path,
    max_cells_per_timepoint: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Reconstruct one AnnData view directly from the processed SCI-FATE2 store."""
    from trajectoryflow.data.store import ScifateStore

    store = ScifateStore(Path(data_root), cache_size=1)

    matrices = []
    obs_frames = []

    for timepoint_index, timepoint in enumerate(store.timepoints):
        snapshot = store.load(timepoint)
        indices = _sample_indices(
            len(snapshot),
            max_cells=max_cells_per_timepoint,
            seed=seed + timepoint_index,
        )

        expression = snapshot.expression[indices].astype(np.float32).tocsr()
        obs = snapshot.obs.iloc[indices].copy().reset_index(drop=True)
        ids = _cell_ids(obs, timepoint)

        obs["cell_id"] = ids
        obs[TIMEPOINT_COLUMN] = str(timepoint)
        obs.index = pd.Index(ids, name="cell_id")

        matrices.append(expression)
        obs_frames.append(obs)

    expression = sparse.vstack(matrices, format="csr")
    obs = pd.concat(obs_frames, axis=0)
    if obs.index.has_duplicates:
        duplicates = obs.index[obs.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"Reconstructed reference data contains duplicate cell IDs, e.g. {duplicates}. Use globally unique cell IDs before PTS/CRS matching.")
    genes = _gene_names(store)

    adata = ad.AnnData(
        X=expression,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.layers["counts"] = expression.copy()
    adata.uns["trajectoryflow_data_root"] = str(Path(data_root).resolve())
    adata.uns["trajectoryflow_timepoints"] = list(map(str, store.timepoints))

    return adata


def library_log1p_sparse(
    expression: sparse.csr_matrix,
    library_size: float = 10_000.0,
) -> sparse.csr_matrix:
    if library_size <= 0:
        raise ValueError("library_size must be > 0.")
    expression = expression.astype(np.float64).tocsr(copy=True)
    if expression.ndim != 2 or 0 in expression.shape:
        raise ValueError("expression must be a non-empty two-dimensional matrix.")
    if not np.isfinite(expression.data).all() or (expression.data < 0).any():
        raise ValueError("expression must contain finite non-negative values.")
    library = np.asarray(expression.sum(axis=1)).ravel()
    scale = np.divide(
        library_size,
        library,
        out=np.zeros_like(library, dtype=np.float64),
        where=library > 0,
    )
    expression = sparse.diags(scale) @ expression
    expression = expression.tocsr()
    expression.data = np.log1p(expression.data)
    return expression


def _incremental_batches(n_rows: int, batch_size: int, min_batch: int):
    batch_size = max(int(batch_size), int(min_batch))
    start = 0

    while start < n_rows:
        end = min(start + batch_size, n_rows)

        if n_rows - end and n_rows - end < min_batch:
            end = n_rows

        yield slice(start, end)
        start = end


def fit_reference_pca(
    adata: ad.AnnData,
    n_components: int = 50,
    batch_size: int = 1024,
    library_size: float = 10_000.0,
) -> None:
    if adata.n_obs < 2:
        raise ValueError("At least two cells are required for PCA.")
    if n_components < 1 or batch_size < 1 or library_size <= 0:
        raise ValueError("n_components, batch_size and library_size must be > 0.")

    n_components = min(int(n_components), adata.n_obs - 1, adata.n_vars)
    if n_components < 2:
        raise ValueError("At least two PCA components are required.")

    transformed = library_log1p_sparse(
        adata.layers["counts"].tocsr(),
        library_size=library_size,
    )

    pca = IncrementalPCA(
        n_components=n_components,
        batch_size=max(batch_size, n_components),
    )

    batches = list(
        _incremental_batches(
            adata.n_obs,
            batch_size=max(batch_size, n_components),
            min_batch=n_components,
        )
    )

    for batch in batches:
        pca.partial_fit(transformed[batch].toarray())

    positions = np.empty((adata.n_obs, n_components), dtype=np.float32)
    for batch in batches:
        positions[batch] = pca.transform(transformed[batch].toarray()).astype(np.float32)

    adata.X = transformed.astype(np.float32)
    adata.obsm["X_pca"] = positions
    adata.varm["PCs"] = pca.components_.T.astype(np.float64)
    adata.uns["pca_mean"] = pca.mean_.astype(np.float64)
    adata.uns["trajectoryflow_pca_library_size"] = float(library_size)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_cache_config(data_root: Path, max_cells_per_timepoint: int | None, n_components: int, pca_batch_size: int, library_size: float, seed: int) -> dict:
    return {
        "data_root": str(data_root.resolve()),
        "manifest_sha256": _sha256_file(data_root / "manifest.json"),
        "max_cells_per_timepoint": -1 if max_cells_per_timepoint is None else int(max_cells_per_timepoint),
        "n_components": int(n_components),
        "pca_batch_size": int(pca_batch_size),
        "library_size": float(library_size),
        "seed": int(seed),
    }


def _cache_matches(adata: ad.AnnData, expected: dict) -> bool:
    cached = adata.uns.get("trajectoryflow_reference_config")
    if not isinstance(cached, dict):
        return False
    normalized = {key: value.item() if isinstance(value, np.generic) else value for key, value in cached.items()}
    return normalized == expected


def load_or_build_reference_adata(
    data_root: str | Path,
    cache_path: str | Path | None = None,
    rebuild: bool = False,
    max_cells_per_timepoint: int | None = None,
    n_components: int = 50,
    pca_batch_size: int = 1024,
    library_size: float = 10_000.0,
    seed: int = 0,
) -> ad.AnnData:
    data_root = Path(data_root)
    if cache_path is None:
        cache_path = data_root / "velocity_references" / "reference_adata.h5ad"
    cache_path = Path(cache_path)
    expected_cache = _reference_cache_config(data_root, max_cells_per_timepoint, n_components, pca_batch_size, library_size, seed)

    if cache_path.exists() and not rebuild:
        try:
            cached = ad.read_h5ad(cache_path)
        except OSError:
            cached = None
        if cached is not None and _cache_matches(cached, expected_cache):
            return cached

    adata = store_to_anndata(
        data_root=data_root,
        max_cells_per_timepoint=max_cells_per_timepoint,
        seed=seed,
    )
    fit_reference_pca(
        adata,
        n_components=n_components,
        batch_size=pca_batch_size,
        library_size=library_size,
    )
    adata.uns["trajectoryflow_reference_config"] = expected_cache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp")
    try:
        adata.write_h5ad(temporary, compression="gzip")
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return adata


def _timepoint_hours(value: str) -> float:
    value = str(value)
    if value.endswith("h"):
        return float(value[:-1])
    if value.startswith("D"):
        return float(value[1:]) * 24.0
    return float(value)


def earliest_timepoint_mask(adata: ad.AnnData) -> np.ndarray:
    values = adata.obs[TIMEPOINT_COLUMN].astype(str).to_numpy()
    hours = np.asarray([_timepoint_hours(value) for value in values])
    return hours == hours.min()


def _principal_graph_arrays(adata: ad.AnnData) -> tuple[np.ndarray, sparse.csr_matrix]:
    graph = adata.uns.get("graph")
    if graph is None:
        raise KeyError("scFates did not create adata.uns['graph'].")

    if "F" not in graph or "B" not in graph:
        raise KeyError("scFates graph must contain 'F' coordinates and 'B' adjacency.")

    raw_adjacency = graph["B"]
    adjacency = (
        raw_adjacency.tocsr()
        if sparse.issparse(raw_adjacency)
        else sparse.csr_matrix(np.asarray(raw_adjacency))
    )
    coords = np.asarray(graph["F"], dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] == 0:
        raise ValueError("scFates graph adjacency must be a non-empty square matrix.")
    if coords.ndim != 2 or not np.isfinite(coords).all():
        raise ValueError("scFates principal-point coordinates must be a finite 2D array.")

    if coords.shape[0] == adjacency.shape[0]:
        node_positions = coords
    elif coords.shape[1] == adjacency.shape[0]:
        node_positions = coords.T
    else:
        raise ValueError("Could not align scFates principal-point coordinates to graph adjacency.")

    return node_positions, adjacency


def choose_scfates_root(adata: ad.AnnData) -> int:
    node_positions, adjacency = _principal_graph_arrays(adata)
    centroid = np.asarray(adata.obsm["X_pca"])[earliest_timepoint_mask(adata)].mean(axis=0)
    degree = np.diff(adjacency.indptr)
    candidates = np.flatnonzero((degree == 1) | (degree > 2))
    if not len(candidates):
        candidates = np.arange(len(node_positions))
    distances = np.linalg.norm(node_positions[candidates] - centroid[None, :], axis=1)
    return int(candidates[np.argmin(distances)])


def principal_tree_reference_vectors(
    adata: ad.AnnData,
    root: int,
) -> np.ndarray:
    """Map cells to local forward directions of the rooted scFates principal tree."""
    node_positions, adjacency = _principal_graph_arrays(adata)
    if root < 0 or root >= adjacency.shape[0]:
        raise ValueError(f"root must lie in [0, {adjacency.shape[0] - 1}].")

    rows, cols = adjacency.nonzero()
    weights = np.linalg.norm(node_positions[rows] - node_positions[cols], axis=1)
    weighted = sparse.csr_matrix((weights, (rows, cols)), shape=adjacency.shape)
    distance_from_root = np.asarray(dijkstra(weighted, indices=root)).ravel()
    if not np.isfinite(distance_from_root).all():
        raise ValueError("scFates principal graph must be connected to the selected root.")

    node_vectors = np.zeros_like(node_positions, dtype=np.float64)

    for node in range(adjacency.shape[0]):
        neighbors = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        forward = neighbors[distance_from_root[neighbors] > distance_from_root[node] + 1e-12]

        if not len(forward):
            continue

        displacement = node_positions[forward] - node_positions[node]
        norms = np.linalg.norm(displacement, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-12

        if np.any(valid):
            node_vectors[node] = (displacement[valid] / norms[valid]).mean(axis=0)

    if "X_R" in adata.obsm:
        raw_assignment = adata.obsm["X_R"]
        assignment = (
            raw_assignment.toarray()
            if sparse.issparse(raw_assignment)
            else np.asarray(raw_assignment, dtype=np.float64)
        )
        if assignment.shape[1] == len(node_vectors):
            return assignment @ node_vectors

    nearest = NearestNeighbors(n_neighbors=1).fit(node_positions)
    nearest_nodes = nearest.kneighbors(
        np.asarray(adata.obsm["X_pca"]),
        return_distance=False,
    )[:, 0]
    return node_vectors[nearest_nodes]


def run_scfates(
    adata: ad.AnnData,
    nodes: int = 200,
    seed: int = 1,
    device: str = "cpu",
    ppt_sigma: float = 0.1,
    ppt_lambda: float = 1.0,
    pseudotime_maps: int = 1,
) -> int:
    if nodes < 2 or ppt_sigma <= 0 or ppt_lambda <= 0 or pseudotime_maps < 1:
        raise ValueError("nodes must be >= 2, ppt_sigma/ppt_lambda > 0, and pseudotime_maps >= 1.")
    if not str(device).strip():
        raise ValueError("device must not be empty.")
    try:
        import scFates as scf
    except ImportError as error:
        raise ImportError(
            "PTS/CRS generation requires scFates. Install it with `pip install scFates`."
        ) from error

    scf.tl.tree(
        adata,
        Nodes=int(nodes),
        use_rep="X_pca",
        method="ppt",
        device=device,
        ppt_sigma=float(ppt_sigma),
        ppt_lambda=float(ppt_lambda),
        seed=int(seed),
    )

    root = choose_scfates_root(adata)
    scf.tl.root(adata, root)
    scf.tl.pseudotime(
        adata,
        n_map=int(pseudotime_maps),
        seed=int(seed),
    )
    return root


def reference_from_vectors(
    adata: ad.AnnData,
    name: str,
    vectors: np.ndarray,
    groups: np.ndarray | None = None,
    alignment_mode: str = "cosine",
) -> VelocityReference:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != adata.n_obs:
        raise ValueError("vectors must have shape [n_cells, n_components].")
    if "cell_id" not in adata.obs.columns:
        raise KeyError("Reference generation requires adata.obs['cell_id'].")
    cell_ids = adata.obs["cell_id"].astype(str).to_numpy()
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("PTS/CRS reference generation requires unique cell IDs.")
    n_components = vectors.shape[1]
    return VelocityReference(
        name=name,
        cell_ids=cell_ids,
        vectors=vectors,
        pca_components=np.asarray(adata.varm["PCs"], dtype=np.float64).T[:n_components],
        pca_mean=np.asarray(adata.uns["pca_mean"], dtype=np.float64),
        positions=np.asarray(adata.obsm["X_pca"], dtype=np.float64)[:, :n_components],
        groups=groups,
        genes=np.asarray(adata.var_names).astype(str),
        expression_transform="library_log1p",
        library_size=float(adata.uns.get("trajectoryflow_pca_library_size", 10_000.0)),
        projection_mode="finite_difference",
        velocity_epsilon=1e-3,
        alignment_mode=alignment_mode,
    )


def build_pts_reference(
    adata: ad.AnnData,
    output: str | Path,
    nodes: int = 200,
    seed: int = 1,
    device: str = "cpu",
    ppt_sigma: float = 0.1,
    ppt_lambda: float = 1.0,
    pseudotime_maps: int = 1,
) -> VelocityReference:
    root = run_scfates(
        adata,
        nodes=nodes,
        seed=seed,
        device=device,
        ppt_sigma=ppt_sigma,
        ppt_lambda=ppt_lambda,
        pseudotime_maps=pseudotime_maps,
    )
    vectors = principal_tree_reference_vectors(adata, root=root)
    reference = reference_from_vectors(adata, name="pts", vectors=vectors)
    save_velocity_reference(Path(output), reference)
    return reference


def transition_matrix_reference_vectors(
    transition,
    positions: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    return np.asarray(transition @ positions - positions, dtype=np.float64)


def build_crs_reference(
    adata: ad.AnnData,
    output: str | Path,
    n_neighbors: int = 30,
    threshold_scheme: str = "hard",
    frac_to_keep: float = 0.3,
) -> VelocityReference:
    if n_neighbors < 1 or not 0 < frac_to_keep <= 1:
        raise ValueError("n_neighbors must be >= 1 and frac_to_keep must lie in (0, 1].")
    if adata.n_obs < 2 or n_neighbors >= adata.n_obs:
        raise ValueError("CRS n_neighbors must be smaller than the number of cells.")
    if threshold_scheme not in ("hard", "soft"):
        raise ValueError("threshold_scheme must be 'hard' or 'soft'.")
    if "t" not in adata.obs.columns:
        raise KeyError("CRS generation requires scFates pseudotime in adata.obs['t'].")

    try:
        import scanpy as sc
        import cellrank as cr
    except ImportError as error:
        raise ImportError(
            "CRS generation requires scanpy and cellrank. Install them with "
            "`pip install scanpy cellrank`."
        ) from error

    adata.obs["trajectoryflow_pseudotime"] = adata.obs["t"].astype(float)
    sc.pp.neighbors(
        adata,
        n_neighbors=int(n_neighbors),
        use_rep="X_pca",
    )

    kernel = cr.kernels.PseudotimeKernel(
        adata,
        time_key="trajectoryflow_pseudotime",
    )
    kernel.compute_transition_matrix(
        threshold_scheme=threshold_scheme,
        frac_to_keep=float(frac_to_keep),
    )

    transition = kernel.transition_matrix
    positions = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    vectors = transition_matrix_reference_vectors(transition, positions)

    reference = reference_from_vectors(adata, name="crs", vectors=vectors)
    save_velocity_reference(Path(output), reference)
    return reference


def build_cbd_reference(
    adata: ad.AnnData,
    output: str | Path,
    cluster_column: str,
    transitions: list[tuple[str, str]],
    n_neighbors: int = 100,
    min_target_neighbors: int = 1,
    name: str = "cbd",
) -> VelocityReference:
    if not transitions:
        raise ValueError("At least one CBD transition is required.")
    if len(set(transitions)) != len(transitions):
        raise ValueError("CBD transitions must be unique.")
    if n_neighbors < 1 or min_target_neighbors < 1:
        raise ValueError("n_neighbors and min_target_neighbors must be >= 1.")
    if min_target_neighbors > n_neighbors:
        raise ValueError("min_target_neighbors cannot exceed n_neighbors.")
    if any(source == target for source, target in transitions):
        raise ValueError("CBD transitions must connect different clusters.")
    if cluster_column not in adata.obs.columns:
        raise KeyError(f"Missing adata.obs[{cluster_column!r}].")

    if "cell_id" not in adata.obs.columns:
        raise KeyError("CBD reference generation requires adata.obs['cell_id'].")
    positions_all = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    cluster_series = adata.obs[cluster_column].astype("string")
    annotated_mask = cluster_series.notna() & cluster_series.str.strip().ne("")
    if not annotated_mask.any():
        raise ValueError(f"adata.obs[{cluster_column!r}] contains no annotated cells.")
    annotated_indices = np.flatnonzero(annotated_mask.to_numpy())
    positions_all = positions_all[annotated_indices]
    clusters = cluster_series.iloc[annotated_indices].astype(str).to_numpy()
    all_cell_ids = adata.obs["cell_id"].astype(str).to_numpy()[annotated_indices]
    if len(np.unique(all_cell_ids)) != len(all_cell_ids):
        raise ValueError("CBD source data requires unique cell IDs.")

    n_neighbors = min(int(n_neighbors) + 1, len(annotated_indices))
    if n_neighbors < 2:
        raise ValueError("CBD requires at least two annotated cells.")
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(positions_all)
    neighbor_indices = neighbors.kneighbors(positions_all, return_distance=False)

    cell_ids = []
    vectors = []
    positions = []
    groups = []

    for source, target in transitions:
        source_indices = np.flatnonzero(clusters == source)
        if not len(source_indices):
            raise ValueError(f"Transition source cluster {source!r} has no cells.")
        if not np.any(clusters == target):
            raise ValueError(f"Transition target cluster {target!r} has no cells.")

        group = f"{source}->{target}"

        for source_index in source_indices:
            candidate = neighbor_indices[source_index]
            candidate = candidate[candidate != source_index]
            target_neighbors = candidate[clusters[candidate] == target]

            if len(target_neighbors) < min_target_neighbors:
                continue

            displacement = positions_all[target_neighbors] - positions_all[source_index]
            norms = np.linalg.norm(displacement, axis=1, keepdims=True)
            valid = np.isfinite(displacement).all(axis=1) & (norms[:, 0] > 1e-12)

            if not np.any(valid):
                continue

            vectors.append((displacement[valid] / norms[valid]).mean(axis=0))
            cell_ids.append(all_cell_ids[source_index])
            positions.append(positions_all[source_index])
            groups.append(group)

    if not cell_ids:
        raise ValueError("No cross-boundary source cells were found.")

    reference = VelocityReference(
        name=name,
        cell_ids=np.asarray(cell_ids),
        vectors=np.asarray(vectors, dtype=np.float64),
        pca_components=np.asarray(adata.varm["PCs"], dtype=np.float64).T,
        pca_mean=np.asarray(adata.uns["pca_mean"], dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float64),
        groups=np.asarray(groups),
        genes=np.asarray(adata.var_names).astype(str),
        expression_transform="library_log1p",
        library_size=float(adata.uns.get("trajectoryflow_pca_library_size", 10_000.0)),
        projection_mode="finite_difference",
        velocity_epsilon=1e-3,
        alignment_mode="mean_unit_direction",
    )
    save_velocity_reference(Path(output), reference)
    return reference
