from trajectoryflow.data.scifate_dataset import ScifateDataLoader

loader = ScifateDataLoader()

scifate_data = loader.load(
    expression_matrix_path="data/processed/scifate2_mtx/matrix.mtx",
    ntr_matrix_path="data/processed/scifate2_mtx/4sU.Binom.ntr.mtx",
    barcodes_path="data/processed/scifate2_mtx/barcodes.tsv",
    features_path="data/processed/scifate2_mtx/features.tsv",
)

print("\n=== DATASET SHAPE ===")
print(scifate_data.expression_matrix.shape)

print("\n=== CELL INFO ===")
print(scifate_data.cell_info.head())

print("\n=== CELLS PER TIMEPOINT ===")
print(scifate_data.cell_info["time"].value_counts())

print("\n=== NUMBER OF GENES ===")
print(len(scifate_data.gene_info))
