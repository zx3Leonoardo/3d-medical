from torch.utils.data import Dataset
import SimpleITK as sitk
import torch
import os
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix

from .filename import generate_filenames
from .transforms import *
from utils import common


class coronary_dataset(Dataset):
    # train/val
    def __init__(self, args, phase) -> None:
        super().__init__()
        self.args = args
        self.phase = phase
        if self.phase=='inference_ids':
            self.img_transforms = Compose([
                maxpool3d(2, 2),
                normTo1(),
            ])
            self.label_transforms = Compose([
                maxpool3d(2, 2),
            ])
        else:
            self.img_transforms = Compose([
                maxpool3d(2, 2),
                zCenterCrop(180),
                normTo1(),
            ])
            self.label_transforms = Compose([
                maxpool3d(2, 2),
                zCenterCrop(180),
            ])
        
        self.filenames = generate_filenames(self.args, self.phase)
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path, label_path = self.filenames[index][0], self.filenames[index][1]
        img, label = sitk.ReadImage(img_path), sitk.ReadImage(label_path)
        img_array, label_array = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(label)

        img_array = torch.FloatTensor(img_array).unsqueeze(0)
        label_array = torch.FloatTensor(label_array).unsqueeze(0)

        if self.label_transforms_inference:
            img_array = self.img_transforms_inference(img_array)
        if self.label_transforms_inference:
            label_array = self.label_transforms_inference(label_array)
        target_array = torch.squeeze(common.one_hot_3d(label_array, self.args.n_labels))
        
        return img_array, target_array

