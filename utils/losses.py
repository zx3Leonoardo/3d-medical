import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.

        for i in range(pred.size()):
            dice += 2*(pred[:,i]*target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1)+
            target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1)+smooth)

        dice /= pred.size(1)
        return torch.clamp((1-dice).mean(), 0, 1)
