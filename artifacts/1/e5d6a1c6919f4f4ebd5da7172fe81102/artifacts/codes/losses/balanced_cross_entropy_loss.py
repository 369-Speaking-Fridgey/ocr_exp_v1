import torch
import torch.nn as nn
import torch.nn.functional as F

class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, neg_ratio = 3.0, eps = 1e-5):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.neg_ratio = neg_ratio
        self.eps = eps
    
    def forward(self, pred, gt, mask, return_origin = False):
        """
        pred: (N, 1, H, W) - the pre
        gt: (N, 1, H, W) - the target
        mask: (N, H, W) - the mask indicates the positive regions
        """
        positive = (gt[:, 0, :, :] * ,mask).byte()
        negative = ((1 - gt[:, 0, :, :]) * mask).byte()
        pos_count = int(positive.float().sum())
        neg_count = min(int(negative.float().sum()), 
                        int(pos_count * self.neg_ratio))
        loss = F.binary_cross_entropy(pred, gt, reduction = 'none')[:, 0, :, :]
        pos_loss = loss * positive.float()
        neg_loss = loss * negative.float()
        neg_loss, _ = torch.topk(neg_loss.view(-1), neg_count)
        balance_loss = (pos_loss.sum() + neg_loss.sum()) / (pos_count + neg_count + self.eps)
        if return_origin:
            return balance_loss, loss
        return balance_loss