from torch.utils.data import Dataset
import torch
import os
import numpy as np

class coronary_dataset(Dataset):
    def __init__(self, root, ids, transforms) -> None:
        super().__init__()
        self.imgs3d = list(sorted(os.listdir(os.path.join(root,"segImgs"))))
        self.labels =list(sorted(os.listdir(os.path.join(root,"parsImgs"))))