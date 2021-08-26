from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.

        for i in range(pred.size(1)):
            dice += 2*(pred[:,i]*target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1)+
            target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1)+smooth)

        dice /= pred.size(1)
        return torch.clamp((1-dice).mean(), 0, 1)

class logDiceloss(nn.Module):
    def __init__(self, smooth=0.0001, p=2, reduction='mean', gamma=0.3):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = torch.clamp(predict, self.smooth, 1-self.smooth)
        predict = predict.contiguous().view(predict.shape[1], -1)
        target = target.contiguous().view(target.shape[1], -1)
        num = torch.sum(torch.mul(predict, target), 1)
        den = torch.sum(predict.pow(self.p), 1) + torch.sum(target.pow(self.p), 1)
        dice = (2 * num + self.smooth) / (den + self.smooth)
        log_dice = -torch.log(dice)
        meanD = torch.mean(torch.pow(log_dice, self.gamma))
        return meanD