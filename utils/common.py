from torch.nn.functional import one_hot
import torch
import numpy as np

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def mxpool3d(img, kernel, stride):
    img = torch.squeeze(img).numpy()
    n = np.int64((np.array(img.shape) - kernel) / stride) + 1
    imgn = np.empty([kernel*kernel*kernel] + n.tolist(), img.dtype)
    grid = np.indices((kernel, kernel, kernel))
    az = np.arange(n[0]) * kernel
    ay = np.arange(n[1]) * kernel
    ax = np.arange(n[2]) * kernel
    for i, c in enumerate(grid.reshape(3, -1).T):
        iz = (az + c[0])[..., None, None]
        iy = (ay + c[1])[None, ..., None]
        ix = (az + c[2])[None, None, ...]
        imgn[i] = img[iz, iy, ix]
    return np.max(imgn, 0)

def dice4pics(img1, img2):
    n = img1.size()[0]
    inter = (img1 * img2).reshape(n, -1)
    union = (img1 + img2).reshape(n, -1)
    dice = (2. * inter.sum(1) + 0.00001) / (union.sum(1) + 0.00001)
    return dice.unsqueeze(0)

def adjust_lr_v1(optim, epoch, args):
    """linear decay"""
    lr = args.lr * (0.1 ** (epoch // 20 ))
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def adjust_lr_v2(optim, lr):
    """constant lr"""
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def one_hot_3d(tensor, n_classes=20):
    n ,s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.type(torch.int64).view(n, 1, s, h, w), 1)
    return one_hot
