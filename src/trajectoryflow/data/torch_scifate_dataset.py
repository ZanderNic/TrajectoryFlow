from torch.utils.data import Dataset
import torch


class TorchScifateDataset(Dataset):

    def __init__(self, scifate_dataset):
        self.scifate_dataset = scifate_dataset

        self.expression_matrix = (
            scifate_dataset.expression_matrix.to_np()
        )

    def __len__(self):
        return self.expression_matrix.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(
            self.expression_matrix[idx],
            dtype=torch.float32
        )

        return x
