#!/usr/bin/env python3

# std-lib imports
import argparse
from pathlib import Path

# 3 party imports
import anndata as ad
import numpy as np

# package imports
from trajectoryflow.experiment.evaluation import (
    VelocityReference,
    save_velocity_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package precomputed PTS/CRS/CBD reference directions from AnnData "
            "into TrajectoryFlow's velocity-reference format."
        )
    )
    parser.add_argument("input", type=Path, help="Input .h5ad file.")
    parser.add_argument("output", type=Path, help="Output .npz reference file.")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--vector-key",
        required=True,
        help="adata.obsm key containing reference vectors in PCA space.",
    )
    parser.add_argument(
        "--position-key",
        default="X_pca",
        help="adata.obsm key containing PCA cell positions.",
    )
    parser.add_argument(
        "--components-key",
        default="PCs",
        help="adata.varm key containing PCA loadings [genes, PCs].",
    )
    parser.add_argument(
        "--cell-id-column",
        default=None,
        help="obs column containing stable cell IDs; defaults to obs_names.",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="Optional obs column for CBD transition/group labels.",
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
    parser.add_argument(
        "--alignment-mode",
        choices=("cosine", "mean_unit_direction"),
        default="cosine",
        help=(
            "Use cosine for PTS/CRS. mean_unit_direction is intended for "
            "precomputed CBD mean-neighbour direction references."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.input)

    if args.vector_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{args.vector_key!r}].")
    if args.position_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{args.position_key!r}].")
    if args.components_key not in adata.varm:
        raise KeyError(f"Missing adata.varm[{args.components_key!r}].")

    vectors = np.asarray(adata.obsm[args.vector_key])
    positions = np.asarray(adata.obsm[args.position_key])
    components = np.asarray(adata.varm[args.components_key]).T

    if vectors.ndim != 2:
        raise ValueError("Reference vectors must be two-dimensional.")
    if positions.ndim != 2:
        raise ValueError("PCA positions must be two-dimensional.")
    if components.ndim != 2:
        raise ValueError("PCA components must be two-dimensional.")

    n_components = vectors.shape[1]

    if positions.shape[1] < n_components:
        raise ValueError(
            "PCA position dimensionality is smaller than reference vectors."
        )
    if components.shape[0] < n_components:
        raise ValueError(
            "PCA loading dimensionality is smaller than reference vectors."
        )

    positions = positions[:, :n_components]
    components = components[:n_components]

    if args.cell_id_column is None:
        cell_ids = np.asarray(adata.obs_names).astype(str)
    else:
        if args.cell_id_column not in adata.obs.columns:
            raise KeyError(
                f"Missing adata.obs[{args.cell_id_column!r}]."
            )
        cell_ids = (
            adata.obs[args.cell_id_column]
            .astype(str)
            .to_numpy()
        )

    groups = None
    if args.group_column is not None:
        if args.group_column not in adata.obs.columns:
            raise KeyError(
                f"Missing adata.obs[{args.group_column!r}]."
            )
        groups = (
            adata.obs[args.group_column]
            .fillna("")
            .astype(str)
            .to_numpy()
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
        cell_ids=cell_ids,
        vectors=vectors,
        pca_components=components,
        pca_mean=pca_mean,
        positions=positions,
        groups=groups,
        genes=np.asarray(adata.var_names).astype(str),
        expression_transform=args.expression_transform,
        library_size=args.library_size,
        projection_mode=args.projection_mode,
        velocity_epsilon=args.velocity_epsilon,
        alignment_mode=args.alignment_mode,
    )

    save_velocity_reference(args.output, reference)

    print("Velocity reference written")
    print(f"  name       : {reference.name}")
    print(f"  cells      : {len(reference.cell_ids)}")
    print(f"  genes      : {reference.n_genes}")
    print(f"  components : {reference.n_components}")
    print(f"  output     : {args.output.resolve()}")


if __name__ == "__main__":
    main()
