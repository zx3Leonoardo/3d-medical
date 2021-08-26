import random
from numpy.lib.function_base import kaiser
from torch._C import dtype, float32
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import torch

class maxpool3d:
    # set kernel and stride to get img smaller in every dim
    def __init__(self, kernel, stride) -> None:
        self.kernel = kernel
        self.stride = stride
    
    def _mxp3d(self, img):
        img = torch.squeeze(img).numpy()
        n = np.int64((np.array(img.shape) - self.kernel) / self.stride) + 1
        imgn = np.empty([self.kernel*self.kernel*self.kernel] + n.tolist(), img.dtype)
        grid = np.indices((self.kernel, self.kernel, self.kernel))
        az = np.arange(n[0]) * self.kernel
        ay = np.arange(n[1]) * self.kernel
        ax = np.arange(n[2]) * self.kernel
        for i, c in enumerate(grid.reshape(3, -1).T):
            iz = (az + c[0])[..., None, None]
            iy = (ay + c[1])[None, ..., None]
            ix = (az + c[2])[None, None, ...]
            imgn[i] = img[iz, iy, ix]
        return torch.unsqueeze(torch.from_numpy(np.max(imgn, 0)), 0)
    
    def __call__(self, img):
        return self._mxp3d(img)

class normTo1:
    # normalize all input voxel to self.value 
    def __init__(self) -> None:
        self.value = 1
    
    def _nT1(self, img):
        img = img.numpy()
        img[img==2] = 1
        img = torch.from_numpy(img)
        return img

    def __call__(self, img):
        return self._nT1(img)

class zCenterCrop:
    # padding or crop input image dim z to zlength
    # pad on both side & crop on both side
    def __init__(self, zlength) -> None:
        self.zlength = zlength
    
    def _zCtrCrp(self, img):
        if self.zlength>=img.size()[1]:
            pd0 = (self.zlength - img.size()[1]) // 2
            return F.pad(img, (0, 0, 0, 0, pd0, pd0))
        else:
            st = (img.size()[1] - self.zlength) // 2
            return img[:, st:st+self.zlength, :]
    
    def __call__(self, img):
        return self._zCtrCrp(img)

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