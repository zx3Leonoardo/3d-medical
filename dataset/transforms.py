import random
from torch._C import float32
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import torch

# class Resize:
#     def __init__(self, scale) -> None:
#         self.scale = scale
    
#     def __call__(self, img, label):
#         img, label = img.unsqueeze(0), label.unsqueeze(0).float()
#         img = F.interpolate(img, scale_factor=(1, self.scale, self.scale), mode='trilinear', recompute_scale_factor=True)
#         label = F.interpolate(label, scale_factor=(1, self.scale, self.scale), mode='trilinear', recompute_scale_factor=True)
#         return img[0], label[0]
class RandomCrop:
    def __init__(self, slices):
        self.slices =  slices

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, mask):

        ss, es = self._get_range(mask.size(1), self.slices)
        
        # print(self.shape, img.shape, mask.shape)
        tmp_img = torch.zeros((img.size(0), self.slices, img.size(2), img.size(3)))
        tmp_mask = torch.zeros((mask.size(0), self.slices, mask.size(2), mask.size(3)))
        tmp_img[:,:es-ss] = img[:,ss:es]
        tmp_mask[:,:es-ss] = mask[:,ss:es]
        return tmp_img, tmp_mask

class Resize:
    def __init__(self, newSize) -> None:
        self.newSize = newSize

    def _resize(self, img, newSize, sampleMethod):
        sample = sitk.ResampleImageFilter()
        originSize = img.GetSize()
        originSpacing = img.GetSpacing()
        newSize = np.array(newSize, float)
        factor = originSize / newSize
        newSpacing = originSpacing*factor
        newSize = newSize.astype(np.int)
        sample.SetReferenceImage(img)
        sample.SetSize(newSize.tolist())
        sample.SetOutputSpacing(newSpacing.tolist())
        sample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        sample.SetInterpolator(sampleMethod)
        imgSampled = sample.Execute(img)
        return imgSampled
        
    def __call__(self, img, label):
        return self._resize(img, self.newSize, sitk.sitkLinear), self._resize(label, self.newSize, sitk.sitkNearestNeighbor)


class RandomFlip_lr:
    def __init__(self, prob) -> None:
        self.prob = prob

    def _flip(self, img, prob):
        if prob<=self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, label):
        prob = random.uniform(0, 1)
        return self._flip(img, self.prob), self._flip(label, self.prob)

class RandomFlip_ud:
    def __init__(self, prob) -> None:
        self.prob = prob
    
    def _flip(self, img, prob):
        if prob<=self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, label):
        prob = random.uniform(0, 1)
        return self._flip(img, self.prob), self._flip(label, self.prob)

class Compose:
    def __init__(self,transforms):
        self.transforms = transforms
    
    def __call__(self, img, label):
        for tr in self.transforms:
            img, label = tr(img, label)
        return img, label