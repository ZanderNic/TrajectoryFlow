from torch.utils.data import Dataset
import torch


class TorchScifateDataset(Dataset):

    def __init__(self, scifate_dataset):
        self.scifate_dataset = scifate_dataset

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None
