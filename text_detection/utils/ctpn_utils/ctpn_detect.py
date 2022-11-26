import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.ctpn_utils.ctpn_data_utils import gen_anchor, transform_bbox, clip_bbox, filter_bbox, nms, TextProposalConnectorOriented


def detect(image, model, display=True, prob_thresh=0.5):
    H, W = image.shape[:2]

def detect_all(eval_dataloader, model, prob_thresh = 0.3, iou = True):
    loop = tqdm(eval_dataloader)
    if iou == False:
        temp_dict = {
            "map": 0.0, "map_50": 0.0, "map_75": 0.0
        }
        for idx, batch in enumerate(loop):
            result = detect_while_training_precision(batch, model, prob_thresh)
            loop.set_postfix(result)
            for key, value in temp_dict.items():
                temp_dict[key] += result[key].item()
        # result = {key:value for (key, value) in ()}
        for key, value in temp_dict.items():
            temp_dict[key] = value.numpy()
            temp_dict[key] /= len(loop)
        return temp_dict
    else:
        temp_dict = {
            "iou": 0.0
        }
        for idx, batch in enumerate(loop):
            result = detect_while_training_iou(batch, model, prob_thresh)
            loop.set_postfix(result)
            for key, value in temp_dict.items():
                temp_dict[key] += result[key].item()
        # result = {key:value for (key, value) in ()}
        for key, value in temp_dict.items():
            temp_dict[key] /= len(loop)
        return temp_dict
    
def detect_while_training_iou(batch, model, prob_thresh = 0.3):
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
        
        fg = np.where(pred_cls_prob[0, :, 1] > prob_thresh)[0] ## score이 threshold 점수보다 높은 경우의 index
        select_anchor = bbox[fg, :]
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
        text, new_text = textConn.get_text_lines(select_anchor, select_score, [H, W]) ## 마지막 text line은 점수를 의미함
    
    pred_box, pred_score = [], []
    gt_box = []
    predicted = np.zeros_like(image.detach().cpu().squeeze(0).numpy())[0,:,:]
    answer = np.zeros_like(image.detach().cpu().squeeze(0).numpy())[0,:,:]
    """
    for line in new_text:
        line = [int(j) for j in line]
        predicted[line[1]:line[3], line[0]:line[2]] = 1
    """
    for line in text:
        score = int(line[-1])
        line = [int(j) for j in line]
        x1, y1, x2, y2, x3, y3, x4, y4 = line[:8]
        min_x, max_x = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
        min_y, max_y = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
        predicted[min_y:max_y,min_x:max_x] = 1
    # cv2.imwrite("./predicted.png", predicted)
    
    predicted = predicted == 1
    
    for box in full_boxes.numpy()[0]:
        min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        answer[min_y:max_y, min_x:max_x] = 1
    answer = answer == 1
    #cv2.imwrite("./answer.png", answer)
    intersection = (predicted&answer).sum((0,1))
    # print(intersection.shape)
    union = (predicted | answer).sum((0,1))
    # logger.info(f"INTERSECTION: {intersection")
    if union.sum() == 0.0 or union.sum() == 0:
        return {"iou": 0.0}
    else:
        iou = intersection / union
    # threshold = np.ceil(np.clip(20))
    
        return {"iou": iou}
        
        
        
def detect_while_training_precision(batch, model, prob_thresh=0.5):
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
        select_anchor = bbox[fg, :]
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
    result = metric.compute()
    return result
        
        