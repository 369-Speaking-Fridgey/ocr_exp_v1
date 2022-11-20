import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.ctpn_utils.ctpn_data_utils import gen_anchor, transform_bbox, clip_bbox, filter_bbox, nms, TextProposalConnectorOriented


def detect(image, model, display=True, prob_thresh=0.5):
    H, W = image.shape[:2]


def detect_while_training(batch, model, prob_thresh=0.5):
    image, cls, regr, full_boxes = batch
    B, C, H, W = image.shape
    # image, cls, regr = image.cuda(), cls.cuda(), regr.cuda()
    image = image.cuda()
    
    
    with torch.no_grad():
        pred_cls, pred_regr = model(image)
        pred_cls_prob = F.softmax(pred_cls, dim = -1).cpu().numpy()
        pred_regr = pred_regr.detach().cpu().numpy()
        anchor = gen_anchor((int(H / 16), int(W/16)), 16)
        bbox = transform_bbox(anchor, pred_regr)
        bbox = clip_bbox(bbox, [H, W])
        
        fg = np.where(pred_cls_prob[0, :, 1] > prob_thresh)[0]
        select_anchor = anchor[fg, :]
        select_score = pred_cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        keep_index = filter_bbox(select_anchor, 16)
        
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nms_box = np.hstack((select_anchor, select_score))
        keep = nms(nms_box, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]
        
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [H, W]) ## 마지막 text line은 점수를 의미함
    
    pred_box, pred_score = [], []
    gt_box = []
    for line in text:
        score = int(line[-1])
        line = [int(j) for j in line]
        x1, y1, x2, y2, x3, y3, x4, y4 = line[:8]
        min_x, max_x = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
        min_y, max_y = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
        pred_box.append([min_x, min_y, max_x, max_y])
        pred_score.append(score)
        
    full_boxes = full_boxes.numpy()
    for box in full_boxes[0]:
        gt_box.append(box)
    preds = [
        dict(
            boxes = torch.tensor(pred_box),
            scores = torch.tensor(pred_score),
            labels = torch.tensor([0 for _ in range(len(pred_box))])
        )
    ]
    
    targets = [
        dict(
            boxes = torch.tensor(gt_box),
            labels = torch.tensor([0 for _ in range(len(gt_box))])
        )
    ]
    
    metric = MeanAveragePrecision()
    metric.update(preds, targets)
    return {
        "map": metric['map'],
        "map_50": metric['map_50'],
        "map_75": metric['map_75']
    }
        
        