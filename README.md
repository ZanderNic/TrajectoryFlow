# TrajectoryFlow

Learning cellular dynamics from SCI-FATE2 using flow matching and generative models to predict plausible future cell states from gene expression and newly synthesized RNA.

## Installation and data preparation

This project uses the publicly available sci-FATE2 dataset from GEO accession [GSE236512](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236512), which was introduced together with the Velvet framework in the paper [“Reconstructing developmental trajectories using latent dynamical systems and time-resolved transcriptomics”](https://doi.org/10.1016/j.cels.2024.04.004). The dataset contains time-resolved single-cell transcriptomics from differentiating mouse embryonic stem cells and provides processed `.h5ad` files with total RNA and newly synthesized RNA estimates.

To install the package locally, clone this repository and run:

```bash
pip install -e .
```

For development, use:

```bash
pip install -e ".[dev]"
```

if the optional development dependencies are defined in `pyproject.toml`.

The data preparation script downloads the processed `estimate` file from GEO if it is not already available locally. If the decompressed `.h5ad` file already exists, the script reuses it and does not download the dataset again. The script reads the `.h5ad` file in a memory-efficient way by processing the sparse matrix row by row instead of loading the complete expression matrix into memory.

The script extracts two sparse MatrixMarket matrices and two TSV metadata files:

```text
matrix.mtx
4sU.Binom.ntr.mtx
barcodes.tsv
features.tsv
```

`matrix.mtx` contains the expression matrix from the `total` layer. `4sU.Binom.ntr.mtx` contains the new-to-total RNA ratio computed from `new_estimated / total`. The `barcodes.tsv` file stores cell identifiers in the format `sample.timepoint.cell_barcode`, for example `MAI5081A385.4h.CGCTTGTTAT`. The `features.tsv` file stores gene information in a format similar to common single-cell MatrixMarket datasets, with columns for gene ID, gene name, feature type, region, and the original gene index.

A typical preprocessing command is:

```bash
python3 scripts/download_scifate2.py \
  --h5ad-path data/raw/GSE236512_processed_data_estimate.h5ad \
  --min-gene-nonzero-fraction 0.05 \
  --top-genes-by-detection 10000 \
  --force-process
```

This command keeps only genes that are detected in at least 5% of all cells and then keeps at most the 10,000 most frequently detected genes. This reduces the final dataset size while preserving genes with broad expression support across the dataset.

After successful processing, the prepared files are written to:

```text
data/processed/scifate2_mtx/
```

with the following structure:

```text
data/processed/scifate2_mtx/matrix.mtx
data/processed/scifate2_mtx/4sU.Binom.ntr.mtx
data/processed/scifate2_mtx/barcodes.tsv
data/processed/scifate2_mtx/features.tsv
```

The prepared dataset can then be loaded with the project loader:

```python
scifate_data = scifate_loader.load(
    expression_matrix_path="data/processed/scifate2_mtx/matrix.mtx",
    ntr_matrix_path="data/processed/scifate2_mtx/4sU.Binom.ntr.mtx",
    barcodes_path="data/processed/scifate2_mtx/barcodes.tsv",
    features_path="data/processed/scifate2_mtx/features.tsv",
)
```

The output matrices are written in `cells x genes` orientation by default. This matches the internal format expected by the loader and avoids an expensive matrix transpose during loading.

## Data preparation parameters

The most relevant parameters of `scripts/download_scifate2.py` are:

| Parameter                     |          Default | Description                                                                                                                                                      |
| ----------------------------- | ---------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`                   |       `estimate` | Selects which processed GEO file to use. For this project, `estimate` is the main file because it contains `total`, `new_estimated`, and `old_estimated` layers. |
| `--h5ad-path`                 |           `None` | Optional path to an already decompressed `.h5ad` file. If provided, the script uses this file instead of downloading again.                                      |
| `--activation-layer`          |          `total` | Layer used as the expression or activation matrix.                                                                                                               |
| `--new-layer`                 |  `new_estimated` | Layer used as newly synthesized RNA.                                                                                                                             |
| `--min-gene-nonzero-fraction` |            `0.0` | Minimum fraction of cells in which a gene must have a non-zero entry. For example, `0.05` means that a gene must be detected in at least 5% of all cells.        |
| `--min-cells`                 |              `0` | Absolute minimum number of cells in which a gene must have a non-zero entry. This can be used instead of, or together with, `--min-gene-nonzero-fraction`.       |
| `--top-genes-by-detection`    |          `10000` | Maximum number of genes to keep after filtering. The most frequently detected genes are retained. Use `0` to keep all genes passing the filters.                 |
| `--orientation`               | `cells-by-genes` | Matrix orientation of the exported `.mtx` files. The default is `cells-by-genes` because the loader expects `n_cells x n_genes`.                                 |
| `--force-process`             |         disabled | Recreates the output files even if they already exist.                                                                                                           |
| `--force-download`            |         disabled | Redownloads and overwrites the existing `.h5ad` file.                                                                                                            |
| `--delete-h5ad`               |         disabled | Deletes the decompressed `.h5ad` file after successful processing to save disk space.                                                                            |
| `--clip-ratio`                |         disabled | Clips the computed new-to-total RNA ratio to the interval `[0, 1]`.                                                                                              |

For machines with limited memory or disk space, a stronger filter can be used:

```bash
python3 scripts/download_scifate2.py \
  --h5ad-path data/raw/GSE236512_processed_data_estimate.h5ad \
  --min-gene-nonzero-fraction 0.01 \
  --top-genes-by-detection 5000 \
  --force-process
```

This creates a smaller dataset with at most 5,000 genes while still requiring each retained gene to be detected in at least 1% of cells.
