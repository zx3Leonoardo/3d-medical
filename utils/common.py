from torch.nn.functional import one_hot
import torch

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

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
