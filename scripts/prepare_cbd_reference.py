#!/usr/bin/env python3

# std-lib imports
import argparse
from pathlib import Path

# 3 party imports
import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

# package imports
from trajectoryflow.experiment.evaluation import (
    VelocityReference,
    save_velocity_reference,
)


def parse_transition(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Transitions must use SOURCE:TARGET, e.g. 'NMP:Mesoderm'."
        )

    source, target = (part.strip() for part in value.split(":", 1))

    if not source or not target:
        raise argparse.ArgumentTypeError(
            "Both SOURCE and TARGET must be non-empty."
        )

    return source, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Cross-Boundary Direction (CBD) reference from known "
            "source->target cluster transitions."
        )
    )
    parser.add_argument("input", type=Path, help="Input .h5ad file.")
    parser.add_argument("output", type=Path, help="Output .npz reference file.")
    parser.add_argument("--name", default="cbd")
    parser.add_argument(
        "--cluster-column",
        required=True,
        help="obs column containing the cluster/cell-type annotation.",
    )
    parser.add_argument(
        "--transition",
        action="append",
        required=True,
        type=parse_transition,
        metavar="SOURCE:TARGET",
        help="Known directed transition. Repeat for multiple transitions.",
    )
    parser.add_argument(
        "--position-key",
        default="X_pca",
        help="adata.obsm key used to define neighborhood geometry.",
    )
    parser.add_argument(
        "--components-key",
        default="PCs",
        help="adata.varm key containing PCA loadings [genes, PCs].",
    )
    parser.add_argument(
        "--cell-id-column",
        default=None,
        help="Stable cell ID column; defaults to obs_names.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=100,
        help="Global kNN size used to identify cross-boundary neighbors.",
    )
    parser.add_argument(
        "--min-target-neighbors",
        type=int,
        default=1,
        help="Minimum target-cluster neighbors required for a source cell.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of PCA dimensions used for CBD scoring.",
    )
    parser.add_argument(
        "--expression-transform",
        choices=("none", "library_log1p"),
        default="none",
    )
    parser.add_argument("--library-size", type=float, default=10_000.0)
    parser.add_argument(
        "--projection-mode",
        choices=("linear", "finite_difference"),
        default="linear",
    )
    parser.add_argument("--velocity-epsilon", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_neighbors < 2:
        raise ValueError("n_neighbors must be >= 2.")
    if args.min_target_neighbors < 1:
        raise ValueError("min_target_neighbors must be >= 1.")
    if args.n_components < 2:
        raise ValueError("n_components must be >= 2.")

    adata = ad.read_h5ad(args.input)

    if args.cluster_column not in adata.obs.columns:
        raise KeyError(f"Missing adata.obs[{args.cluster_column!r}].")
    if args.position_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{args.position_key!r}].")
    if args.components_key not in adata.varm:
        raise KeyError(f"Missing adata.varm[{args.components_key!r}].")

    positions_all = np.asarray(adata.obsm[args.position_key], dtype=np.float64)
    components_all = np.asarray(adata.varm[args.components_key], dtype=np.float64).T

    n_components = min(
        args.n_components,
        positions_all.shape[1],
        components_all.shape[0],
    )

    if n_components < 2:
        raise ValueError("At least two PCA components are required.")

    positions_all = positions_all[:, :n_components]
    components = components_all[:n_components]
    clusters = adata.obs[args.cluster_column].astype(str).to_numpy()

    if args.cell_id_column is None:
        all_cell_ids = np.asarray(adata.obs_names).astype(str)
    else:
        if args.cell_id_column not in adata.obs.columns:
            raise KeyError(f"Missing adata.obs[{args.cell_id_column!r}].")
        all_cell_ids = adata.obs[args.cell_id_column].astype(str).to_numpy()

    n_neighbors = min(args.n_neighbors + 1, adata.n_obs)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(positions_all)
    neighbor_indices = neighbors.kneighbors(return_distance=False)

    cell_ids = []
    vectors = []
    positions = []
    groups = []

    for source, target in args.transition:
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

            if len(target_neighbors) < args.min_target_neighbors:
                continue

            displacement = positions_all[target_neighbors] - positions_all[source_index]
            norms = np.linalg.norm(displacement, axis=1, keepdims=True)
            valid = np.isfinite(displacement).all(axis=1) & (norms[:, 0] > 1e-12)

            if not np.any(valid):
                continue

            unit_displacement = displacement[valid] / norms[valid]

            # If v_hat is the normalized predicted velocity, then
            # v_hat @ mean(unit_displacement) equals the mean cosine
            # similarity to the target-cluster boundary neighbors.
            mean_unit_direction = unit_displacement.mean(axis=0)

            cell_ids.append(all_cell_ids[source_index])
            vectors.append(mean_unit_direction)
            positions.append(positions_all[source_index])
            groups.append(group)

    if not cell_ids:
        raise ValueError(
            "No cross-boundary source cells were found. Increase n_neighbors, "
            "check the transition labels, or lower min_target_neighbors."
        )

    pca_mean = np.asarray(
        adata.uns.get("pca_mean", np.zeros(adata.n_vars)),
        dtype=np.float64,
    )

    if pca_mean.shape != (adata.n_vars,):
        raise ValueError(
            "adata.uns['pca_mean'] must have shape [n_genes] if provided."
        )

    reference = VelocityReference(
        name=args.name,
        cell_ids=np.asarray(cell_ids),
        vectors=np.asarray(vectors),
        pca_components=components,
        pca_mean=pca_mean,
        positions=np.asarray(positions),
        groups=np.asarray(groups),
        genes=np.asarray(adata.var_names).astype(str),
        expression_transform=args.expression_transform,
        library_size=args.library_size,
        projection_mode=args.projection_mode,
        velocity_epsilon=args.velocity_epsilon,
        alignment_mode="mean_unit_direction",
    )

    save_velocity_reference(args.output, reference)

    print("CBD reference written")
    print(f"  transitions : {len(args.transition)}")
    print(f"  rows        : {len(reference.cell_ids)}")
    print(f"  components  : {reference.n_components}")
    print(f"  output      : {args.output.resolve()}")


if __name__ == "__main__":
    main()
