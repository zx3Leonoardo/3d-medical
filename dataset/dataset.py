from torch.utils.data import Dataset
import SimpleITK as sitk
from filename import generate_filenames
import torch
import os
import numpy as np
from transforms import *

class coronary_dataset(Dataset):
    def __init__(self, args, phase) -> None:
        super().__init__()
        self.args = args
        self.transforms = Compose([
            RandomFlip_lr(0.5),
            RandomFlip_ud(0.5),
        ])
        self.filenames = generate_filenames(args, phase)
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path = self.filenames[index][0]
        label_path = self.filenames[index][1]
        img = sitk.ReadImage(img_path)
        label = sitk.ReadImage(label_path)
        img_array = sitk.GetArrayFromImage(img)
        label_array = sitk.GetArrayFromImage(label)

        #img_array /= self.args.norm_factor
        img_array = img_array.astype(np.float32)
        img_array = torch.FloatTensor(img_array).unsqueeze(0)
        label_array = torch.FloatTensor(label_array).unsqueeze(0)

        if self.transforms:
            img_array, label_array = self.transforms(img_array, label_array)
        return img_array, label_array