#!/usr/bin/env python3

# std-lib imports
import argparse
from pathlib import Path

# 3 party imports

# package imports
from trajectoryflow.data.scifate2 import FILES, prepare_scifate2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare SCI-FATE2 GSE236512 as a timepoint-split sparse training dataset.")
    parser.add_argument("--dataset", choices=FILES, default="estimate")
    parser.add_argument("--work-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/scifate2"))
    parser.add_argument("--h5ad-path", type=Path, default=None)
    parser.add_argument("--activation-layer", default="total")
    parser.add_argument("--new-layer", default="new_estimated")
    parser.add_argument("--timepoint-column", default="timepoint")
    parser.add_argument("--gene-id-column", default=None, help="Optional adata.var column used as gene ID; otherwise var_names are used.")
    parser.add_argument("--cell-annotations", type=Path, default=None, help="Optional CSV/TSV/Parquet/H5AD annotation source merged by cell ID. The authors' published celltyped_adata.h5ad can be used directly.")
    parser.add_argument("--annotation-cell-id-column", default="cell_id", help="Cell ID column in --cell-annotations; H5AD obs_names are used when this column is absent.")
    parser.add_argument("--annotation-cell-type-column", default=None, help="Column copied to canonical cell_type. If omitted, cell_annotation is used automatically when present.")
    parser.add_argument("--authors-cell-types", action="store_true", help="Reconstruct the published Maizels et al. cell annotations from the counting H5AD and merge them by cell ID.")
    parser.add_argument("--authors-counting-h5ad", type=Path, default=None, help="Optional local counting H5AD used for --authors-cell-types; downloaded from GEO when omitted and not cached.")
    parser.add_argument("--force-authors-cell-types", action="store_true", help="Recompute the authors' annotation cache even when a valid cache exists.")
    parser.add_argument("--authors-chunk-size", type=int, default=2048, help="Rows read at once while reconstructing authors' cell types. Lower this if RAM is tight; outputs are unchanged.")
    parser.add_argument("--min-cells", type=int, default=0, help="Absolute minimum number of cells in which a gene must be non-zero.")
    parser.add_argument("--min-gene-nonzero-fraction", type=float, default=0.0, help="Minimum fraction of all cells in which a gene must be non-zero.")
    parser.add_argument("--top-genes-by-detection", type=int, default=10_000, help="Keep the top N globally detected genes after filtering; 0 keeps all.")
    parser.add_argument("--clip-ratio", action="store_true", help="Clip computed NTR values to [0, 1].")
    parser.add_argument("--uncompressed-npz", action="store_true", help="Write larger but faster-loading uncompressed NPZ matrices.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--delete-h5ad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_scifate2(dataset=args.dataset, work_dir=args.work_dir, output_dir=args.output_dir, h5ad_path=args.h5ad_path, activation_layer=args.activation_layer, new_layer=args.new_layer, min_cells=args.min_cells, min_gene_nonzero_fraction=args.min_gene_nonzero_fraction, top_genes_by_detection=args.top_genes_by_detection, force_download=args.force_download, force_process=args.force_process, delete_h5ad=args.delete_h5ad, clip_ratio=args.clip_ratio, timepoint_column=args.timepoint_column, gene_id_column=args.gene_id_column, compressed_npz=not args.uncompressed_npz, cell_annotations=args.cell_annotations, annotation_cell_id_column=args.annotation_cell_id_column, annotation_cell_type_column=args.annotation_cell_type_column, authors_cell_types=args.authors_cell_types, authors_counting_h5ad=args.authors_counting_h5ad, force_authors_cell_types=args.force_authors_cell_types, authors_chunk_size=args.authors_chunk_size)


if __name__ == "__main__":
    main()
