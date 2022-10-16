import torch
import torch.nn as nn

class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()
    
    def forward(self, pred, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum, dict(l1_loss = mask_sum)
        else:
            loss = torch.abs(pred[:, 0] - gt * mask).sum() / mask_sum
            return loss, dict(l1_loss = loss)
        
        
class BalanceL1Loss(nn.Module):
    def __init__(self, negative_ratio = 3.0):
        super(BalanceL1Loss, self).__init__()
        self.neg_ratio = negative_ratio
    
    def forward(self, pred, gt, mask):
        loss = torch.abs(pres[:, 0] - gt)
        pos = loss * mask
        neg = loss * (1-mask)
        pos_cnt = int(mask.sum())
        neg_cnt = min(int((1 - mask).sum()), int(pos_cnt * self.neg_ratio))
        neg_loss, _ = torch.topk(neg.view(-1), neg_cnt)
        neg_loss = neg_loss.sum() / neg_cnt
        pos_loss = pos_loss.sum() / pos_cnt
        return pos_loss + neg_loss, dict(l1_loss = pos_loss, neg_l1_loss = neg_loss)