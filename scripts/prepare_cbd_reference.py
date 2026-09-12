#!/usr/bin/env python3

# std-lib imports
import argparse
from pathlib import Path

# 3 party imports
import anndata as ad

# package imports
from trajectoryflow.experiment.reference_generation import (
    SCIFATE2_PUBLISHED_CBD_TRANSITIONS,
    build_cbd_reference,
    fit_reference_pca,
    load_or_build_reference_adata,
)


def parse_transition(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Transitions must use SOURCE:TARGET, e.g. 'NMP:Mesoderm'."
        )
    source, target = (part.strip() for part in value.split(":", 1))
    if not source or not target:
        raise argparse.ArgumentTypeError("Both SOURCE and TARGET must be non-empty.")
    return source, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CBD reference directly from TrajectoryFlow's processed SCI-FATE2 "
            "store, or from an AnnData file."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Processed SCI-FATE2 directory or .h5ad file.",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default="cbd")
    parser.add_argument("--cluster-column", default="cell_type")
    parser.add_argument(
        "--transition",
        action="append",
        default=[],
        type=parse_transition,
        metavar="SOURCE:TARGET",
        help="CBD transition. Repeat for multiple transitions.",
    )
    parser.add_argument(
        "--published-scifate2-transitions",
        action="store_true",
        help="Use the seven cell_annotation edges from the authors' SCI-FATE2 maxi benchmark.",
    )
    parser.add_argument("--n-neighbors", type=int, default=100)
    parser.add_argument("--min-target-neighbors", type=int, default=1)

    parser.add_argument("--cache-adata", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-cells-per-timepoint", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--pca-batch-size", type=int, default=1024)
    parser.add_argument("--library-size", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input.is_dir():
        adata = load_or_build_reference_adata(
            data_root=args.input,
            cache_path=args.cache_adata,
            rebuild=args.rebuild_cache,
            max_cells_per_timepoint=args.max_cells_per_timepoint,
            n_components=args.n_components,
            pca_batch_size=args.pca_batch_size,
            library_size=args.library_size,
            seed=args.seed,
        )
    else:
        adata = ad.read_h5ad(args.input)
        if "X_pca" not in adata.obsm or "PCs" not in adata.varm:
            if "counts" not in adata.layers:
                adata.layers["counts"] = adata.X.copy()
            fit_reference_pca(
                adata,
                n_components=args.n_components,
                batch_size=args.pca_batch_size,
                library_size=args.library_size,
            )
        if "cell_id" not in adata.obs.columns:
            adata.obs["cell_id"] = adata.obs_names.astype(str)

    transitions = list(SCIFATE2_PUBLISHED_CBD_TRANSITIONS if args.published_scifate2_transitions else ()) + args.transition
    transitions = list(dict.fromkeys(transitions))
    if not transitions:
        raise SystemExit("Provide --transition SOURCE:TARGET or --published-scifate2-transitions.")

    reference = build_cbd_reference(
        adata,
        output=args.output,
        cluster_column=args.cluster_column,
        transitions=transitions,
        n_neighbors=args.n_neighbors,
        min_target_neighbors=args.min_target_neighbors,
        name=args.name,
    )

    print("CBD reference written")
    print(f"  transitions : {len(transitions)}")
    print(f"  rows        : {len(reference.cell_ids)}")
    print(f"  components  : {reference.n_components}")
    print(f"  output      : {args.output.resolve()}")


if __name__ == "__main__":
    main()
