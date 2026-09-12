#!/usr/bin/env python3

# std-lib imports
import argparse
from pathlib import Path

# 3 party imports

# package imports
from trajectoryflow.experiment.reference_generation import (
    SCIFATE2_PUBLISHED_CBD_TRANSITIONS,
    build_cbd_reference,
    build_crs_reference,
    load_or_build_reference_adata,
    principal_tree_reference_vectors,
    reference_from_vectors,
    run_scfates,
)
from trajectoryflow.experiment.evaluation import (
    save_velocity_reference,
)


def parse_transition(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Use SOURCE:TARGET, e.g. NMP:Mesoderm.")
    source, target = (part.strip() for part in value.split(":", 1))
    if not source or not target:
        raise argparse.ArgumentTypeError("Both SOURCE and TARGET must be non-empty.")
    return source, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TrajectoryFlow PTS, CRS and optional CBD references directly "
            "from the processed SCI-FATE2 store."
        )
    )
    parser.add_argument(
        "data_root",
        type=Path,
        nargs="?",
        default=Path("data/processed/scifate2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--cache-adata", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-cells-per-timepoint", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--pca-batch-size", type=int, default=1024)
    parser.add_argument("--library-size", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ppt-sigma", type=float, default=0.1)
    parser.add_argument("--ppt-lambda", type=float, default=1.0)
    parser.add_argument("--pseudotime-maps", type=int, default=1)

    parser.add_argument("--cellrank-neighbors", type=int, default=30)
    parser.add_argument(
        "--threshold-scheme",
        choices=("hard", "soft"),
        default="hard",
    )
    parser.add_argument("--frac-to-keep", type=float, default=0.3)

    parser.add_argument("--cluster-column", default="cell_type")
    parser.add_argument(
        "--transition",
        action="append",
        type=parse_transition,
        default=[],
        metavar="SOURCE:TARGET",
        help="Optional CBD transition. Repeat for multiple transitions.",
    )
    parser.add_argument(
        "--published-scifate2-transitions",
        action="store_true",
        help="Add the seven cell_annotation edges from the authors' SCI-FATE2 maxi benchmark.",
    )
    parser.add_argument("--cbd-neighbors", type=int, default=100)
    parser.add_argument("--min-target-neighbors", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data_root / "velocity_references"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_adata or output_dir / "reference_adata.h5ad"

    print("Preparing common reference representation...")
    adata = load_or_build_reference_adata(
        data_root=args.data_root,
        cache_path=cache_path,
        rebuild=args.rebuild_cache,
        max_cells_per_timepoint=args.max_cells_per_timepoint,
        n_components=args.n_components,
        pca_batch_size=args.pca_batch_size,
        library_size=args.library_size,
        seed=args.seed,
    )

    print("Running scFates tree + pseudotime...")
    root = run_scfates(
        adata,
        nodes=args.nodes,
        seed=args.seed,
        device=args.device,
        ppt_sigma=args.ppt_sigma,
        ppt_lambda=args.ppt_lambda,
        pseudotime_maps=args.pseudotime_maps,
    )

    pts_vectors = principal_tree_reference_vectors(adata, root=root)
    pts = reference_from_vectors(
        adata,
        name="pts",
        vectors=pts_vectors,
    )
    save_velocity_reference(output_dir / "pts.npz", pts)

    print("Running CellRank PseudotimeKernel...")
    crs = build_crs_reference(
        adata,
        output=output_dir / "crs.npz",
        n_neighbors=args.cellrank_neighbors,
        threshold_scheme=args.threshold_scheme,
        frac_to_keep=args.frac_to_keep,
    )

    transitions = list(SCIFATE2_PUBLISHED_CBD_TRANSITIONS if args.published_scifate2_transitions else ()) + args.transition
    transitions = list(dict.fromkeys(transitions))
    cbd = None
    if transitions:
        print("Building CBD reference...")
        cbd = build_cbd_reference(
            adata,
            output=output_dir / "cbd.npz",
            cluster_column=args.cluster_column,
            transitions=transitions,
            n_neighbors=args.cbd_neighbors,
            min_target_neighbors=args.min_target_neighbors,
        )

    # Persist the derived scFates analysis so it can be inspected/reused.
    adata.write_h5ad(cache_path, compression="gzip")

    print("\nDirection references ready")
    print(f"  cells         : {adata.n_obs}")
    print(f"  genes         : {adata.n_vars}")
    print(f"  PCA components: {pts.n_components}")
    print(f"  scFates root  : {root}")
    print(f"  PTS           : {(output_dir / 'pts.npz').resolve()}")
    print(f"  CRS           : {(output_dir / 'crs.npz').resolve()}")
    if cbd is not None:
        print(f"  CBD rows      : {len(cbd.cell_ids)}")
        print(f"  CBD           : {(output_dir / 'cbd.npz').resolve()}")
    print(f"  analysis cache: {cache_path.resolve()}")


if __name__ == "__main__":
    main()
