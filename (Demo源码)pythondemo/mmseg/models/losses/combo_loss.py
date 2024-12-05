import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from .utils import weight_reduce_loss

ALPHA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss


@MODELS.register_module()
class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, pred, target, smooth=1, reduction='mean', avg_factor=None):
        # flatten prediction and target tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        pred = torch.clamp(pred, 1e-9, 1.0 - 1e-9)
        ce_loss = - (ALPHA * ((target * torch.log(pred)) + ((1 - ALPHA) * (1.0 - target) * torch.log(1.0 - pred))))
        weighted_ce = ce_loss.mean()

        combo_loss = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        loss = weight_reduce_loss(combo_loss, None, reduction, avg_factor)

        return loss
