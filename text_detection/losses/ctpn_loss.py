import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ctpn_utils.ctpn_data_utils import OHEM, RPN_TOTAL_NUM
""" RPN_REGR_Loss
- Smooth L1 Loss
"""
class RPN_REGR_Loss(nn.Module):
    def __init__(self, sigma = 9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
    
    def forward(self, pred_regr, gt_regr):
        try:
            cls = gt_regr[0, :, 0]
            regression = gt_regr[0, :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regression[regr_keep]
            regr_pred = pred_regr[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0/self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigam + torch.abs(1 - less_one) * (diff - 0.5/self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except:
            loss = torch.tensor(0.0)
        return loss

class RPN_CLS_Loss(nn.Module):
    def __init__(self):
        super(RPN_CLS_Loss, self).__init__()
        self.L_cls = nn.CrossEntropyLoss(reduction = 'none')
    
    def forward(self, pred_cls, gt_cls):
        if OHEM:
            cls_gt = gt_cls[0][0]
            num_pos = 0
            loss_pos_num = 0
            
            if len((gt_cls == 1).nonzero()) != 0:
                cls_pos = (cls_gt == 1).nonzero()[:, 0]
                gt_pos = cls_gt[cls_pos].long()
                cls_pred_pos = pred_cls[0][cls_pos]
                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))
                loss_pos_num = loss_pos.sum()
                num_pos = len(loss_pos)
                
            cls_neg = (cls_gt == 0).nonzero()[:, 0]
            gt_neg = cls_gt[cls_neg]
            cls_pred_neg = pred_cls[0][cls_neg]
            
            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
            loss_neg_topK, _ = torch.topk(loss_neg, min(len(loss_neg), RPN_TOTAL_NUM - num_pos))
            loss_cls = loss_pos_num + loss_neg_topK.sum()
            loss_cls /= RPN_TOTAL_NUM
            
            return loss_cls
        else:
            y_true = gt_cls[0][0]
            cls_keep = (y_true != -1).nonzero()[:, 0]
            cls_true = y_true[cls_keep].long()
            cls_pred = pred_cls[0][cls_keep]
            loss = F.nll_loss(F.log_softmax(cls_pred, dim = -1), cls_true)
            loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
            
            return loss
        
class CTPNLoss(nn.Module):
    def __init__(self):
        super(CTPNLoss, self).__init__()
        self.regr = RPN_REGR_Loss()
        self.cls = RPN_CLS_Loss()
        
    def forward(self, pred_cls, pred_regr, gt_cls, gt_regr):
        regr_loss = self.regr(pred_regr, gt_regr)
        cls_loss = self.cls(pred_cls, gt_cls)
        
        loss = regr_loss + cls_loss
        return loss
        
        