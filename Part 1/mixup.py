import numpy as np
import torch
import torch.nn as nn

def mixup(batch, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    data = lam * data + (1 - lam) * shuffled_data
    targets = (targets, shuffled_targets, lam)

    return data, targets


class MixUpCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = mixup(batch, self.alpha)
        return batch


class MixUpCriterion:
    def __init__(self, reduction):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)