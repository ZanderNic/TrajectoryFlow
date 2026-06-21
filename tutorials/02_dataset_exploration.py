from trajectoryflow.data.scifate_dataset import ScifateDataLoader

loader = ScifateDataLoader()

scifate_data = loader.load(
    expression_matrix_path="data/processed/scifate2_mtx/matrix.mtx",
    ntr_matrix_path="data/processed/scifate2_mtx/4sU.Binom.ntr.mtx",
    barcodes_path="data/processed/scifate2_mtx/barcodes.tsv",
    features_path="data/processed/scifate2_mtx/features.tsv",
)

print("=== DATASET INFO ===")

print("Expression matrix shape:")
print(scifate_data.expression_matrix.shape)

print("\nCell info columns:")
print(scifate_data.cell_info.columns)

print("\nFirst 5 rows:")
print(scifate_data.cell_info.head())

print("\nGene info columns:")
print(scifate_data.gene_info.columns)

print("\nFirst 5 genes:")
print(scifate_data.gene_info.head())
