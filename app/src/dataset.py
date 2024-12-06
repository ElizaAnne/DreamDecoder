import torch
from torch.utils.data import Dataset

class DreamDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_data():
    # Placeholder: Load your dream bank data here
    data = torch.randint(0, 1000, (32, 50))  # Sample input data (batch_size, seq_len)
    targets = torch.randint(0, 1000, (32, 50))  # Sample target data
    return data, targets
