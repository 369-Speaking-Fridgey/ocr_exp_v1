import torch
import torch.nn as nn

def dice_loss(gt_score, pred_score):
    inter = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1. - (2 * inter / union)

def geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt) ## (세로 x 가로)
    area_pred = (d1_pred + d3_pred) + (d2_pred + d4_pred)
    w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred) 
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt) ## loss of rotation angle
    return iou_loss_map, angle_loss_map


class EASTLoss(nn.Module):
    def __init__(self, weight_angle = 10):
        super(EASTLoss, self).__init__()
        self.weight_angle = weight_angle
    
    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0
        
        classify_loss = dice_loss(gt_score, pred_score * (1-ignored_map))   ## 어느 영역이 text를 포함하고 있는 영역인지에 대한 점수를 계산한다.
        iou_loss_map, angle_loss_map = geo_loss(gt_geo, pred_geo)
        
        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
        
        ## The overal geometry loss is the weighted sum of the AABB loss and angle loss
        geo_loss = self.weight_angle * angle_loss + iou_loss
        
        return geo_loss + classify_loss