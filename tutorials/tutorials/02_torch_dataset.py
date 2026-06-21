from trajectoryflow.data.scifate_dataset import ScifateDataLoader
from trajectoryflow.data.torch_scifate_dataset import TorchScifateDataset

# Load dataset
loader = ScifateDataLoader()

scifate_data = loader.load(
    expression_matrix_path="data/processed/scifate2_mtx/matrix.mtx",
    ntr_matrix_path="data/processed/scifate2_mtx/4sU.Binom.ntr.mtx",
    barcodes_path="data/processed/scifate2_mtx/barcodes.tsv",
    features_path="data/processed/scifate2_mtx/features.tsv",
)

# Create PyTorch dataset
torch_dataset = TorchScifateDataset(scifate_data)

print("Number of cells:", len(torch_dataset))
print("Shape of first cell:", torch_dataset[0].shape)
