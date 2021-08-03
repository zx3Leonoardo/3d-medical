import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LossAverage(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)

class DiceAverage(object):
    def __init__(self, class_num) -> None:
        self.class_num = class_num
        self.reset()
    
    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_id in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_id, :, :, :] * targets[:, class_id, :, :, :])
            union = torch.sum(logits[:, class_id, :, :, :] + targets[:, class_id, :, :, :])
            dice = (2. * inter+1)/(union+1)
            dices.append(dice.item())
        return np.asarray(dices)