import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
