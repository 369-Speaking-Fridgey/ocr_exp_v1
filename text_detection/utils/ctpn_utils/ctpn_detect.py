import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.ctpn_utils.ctpn_data_utils import gen_anchor, transform_bbox, clip_bbox, filter_bbox, nms, TextProposalConnectorOriented
from loguru import logger
import torch
import numpy as np

from torch import Tensor
from typing import Tuple, List
from utils.ctpn_utils.anchor_data import generate_all_anchor_boxes
from utils.ctpn_utils.connector import TextProposalConnector

def clip_bboxes(bboxes: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Clip the bounding boxes within the image boundary.
    Args:
        bboxes (numpy.ndarray): The set of bounding boxes.
        image_size (int, tuple): The image's size.
    Returns:
        THe bounding boxes that are within the image boundaries.
    """

    height, width = image_size

    zero = 0.0
    w_diff = width - 1.0
    h_diff = height - 1.0

    # x1 >= 0 and x2 < width
    bboxes[:, 0::2] = np.maximum(np.minimum(bboxes[:, 0::2], w_diff), zero)
    # y1 >= 0 and y2 < height
    bboxes[:, 1::2] = np.maximum(np.minimum(bboxes[:, 1::2], h_diff), zero)

    return bboxes


def nms(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Compute the non max-suppression algorithm.
    Args:
        bboxes (numpy.ndarray): The bounding box coordinates.
        scores (numpy.ndarray): The scores for each bounding box coordinate.
        iou_threshold (float): The Jaccard overlap threshold.
    Returns:
        A list containing the best indices out of a set of overlapping bounding boxes.
    """

    # Grabbing the coordinates of the bounding boxes
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    areas = (xmax - xmin + 1) * (ymax - ymin + 1)

    # Sorting the scores in descending order
    score_indices = np.argsort(scores, kind="mergesort", axis=-1)[::-1]

    zero = 0.0

    candidates = []

    while score_indices.size > 0:
        # Picking the index of the highest IoU
        i = score_indices[0]

        candidates.append(i)

        # Finding the highest (xmin, ymin) coordinates
        xxmax = np.maximum(xmin[i], xmin[score_indices[1:]])
        yymax = np.maximum(ymin[i], ymin[score_indices[1:]])

        # Finding the smallest (xmax, ymax) coordinates
        xxmin = np.minimum(xmax[i], xmax[score_indices[1:]])
        yymin = np.minimum(ymax[i], ymax[score_indices[1:]])

        # compute the width and height of the bounding box
        w = np.maximum(zero, xxmin - xxmax)
        h = np.maximum(zero, yymin - yymax)

        area_of_overlap = w * h
        remaining_areas = areas[score_indices[1:]]
        area_of_union = areas[i] + remaining_areas - area_of_overlap

        # Computing the Intersection Over Union. That is:
        # dividing the area of overlap between the bounding boxes by the area of union
        IoU = area_of_overlap / area_of_union

        # Keeping only elements with an IoU <= thresh
        indices = np.where(IoU <= iou_threshold)[0]
        score_indices = score_indices[indices + 1]

    return candidates
def decode(predicted_bboxes: np.ndarray, anchor_boxes: np.ndarray) -> np.ndarray:
    """
    Decode the predicted bounding boxes.
    Args:
        predicted_bboxes (numpy array): The predicted set of bounding boxes.
        anchor_boxes (numpy array): The set of default boxes.
    Returns:
        The decoded bounding boxes that was predicted by the CTPN.
    """

    # The height of the anchor boxes.
    ha = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1.0

    # The center y-axis of the anchor boxes.
    Cya = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2.0

    # The center y-axis of the predicted boxes
    Vcy = predicted_bboxes[..., 0] * ha + Cya

    # The height of the predicted boxes
    Vhx = np.exp(predicted_bboxes[..., 1]) * ha

    x1 = anchor_boxes[:, 0]
    y1 = Vcy - Vhx / 2.0
    x2 = anchor_boxes[:, 2]
    y2 = Vcy + Vhx / 2.0

    bboxes = np.stack([x1, y1.squeeze(), x2, y2.squeeze()], axis=1)

    return bboxes

class TextDetector:
    def __init__(self):
        """
        Detect text in an image.
        
        Args:
            configs: The configuration file.
            
        """
        self.CONF_SCORE=0.9
        self.IOU_THRESH=0.3
        self.text_proposal_connector: object = TextProposalConnector()

    def __call__(self,
                 predictions: Tuple[Tensor, Tensor],
                 image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the text localization.
        
        Args:
            predictions (Tensor, tuple): The model's predictions.
            image_size (int, tuple): The image's size.
        Returns:
            A tuple containing the predicted bounding boxes and scores.
            
        """
        h, w = image_size

        predicted_bboxes, predicted_scores = predictions

        predicted_scores = torch.softmax(predicted_scores, dim=2)

        # Putting all to numpy array
        predicted_bboxes = predicted_bboxes.cpu().numpy()
        predicted_scores = predicted_scores.cpu().numpy()

        anchor_shift = 16

        # Estimate the size of feature map created by Convolutional neural network (VGG-16)
        feature_map_size = [int(np.ceil(h / anchor_shift)), int(np.ceil(w / anchor_shift))]

        # Generate all anchor boxes.
        anchor_boxes = generate_all_anchor_boxes(
            feature_map_size=feature_map_size,
            feat_stride=16,
            anchor_heights= [11, 15, 22, 32, 45, 65, 93, 133, 190, 273],
            anchor_shift=anchor_shift
        )

        # Decoding the model's predictions.
        decoded_bboxes = decode(predicted_bboxes=predicted_bboxes, anchor_boxes=anchor_boxes)

        # Keeping the predicted/decoded boxes inside the image.
        clipped_bboxes = clip_bboxes(bboxes=decoded_bboxes, image_size=image_size)
        # Taking only boxes and scores based on the text proposal minimum score.
        text_class = 1
        conf_scores = predicted_scores[0, :, text_class]  # 1: text
        conf_scores_mask = np.where(conf_scores > self.CONF_SCORE)[0]
        # print(len(conf_scores_mask))
        selected_bboxes = clipped_bboxes[conf_scores_mask, :]
        selected_scores = predicted_scores[0, conf_scores_mask, text_class]

        # Perform the non-max-suppression to eliminate unnecessary bounding boxes.
        candidates = nms(bboxes=selected_bboxes,
                         scores=selected_scores,
                         iou_threshold=self.IOU_THRESH)

        selected_bboxes, selected_scores = selected_bboxes[candidates], selected_scores[candidates]
        # Taking the text lines.
        text_lines, scores = self.text_proposal_connector.get_text_lines(text_proposals=selected_bboxes,
                                                                         scores=selected_scores,
                                                                         im_size=image_size)

        # detections: 0 = detected bboxes, detections: 1 = detected scores
        detections = (text_lines, scores)

        return text_lines, scores
 
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
            result = detect_while_training_iou(batch, model, prob_thresh, new=True)
            loop.set_postfix(result)
            for key, value in temp_dict.items():
                temp_dict[key] += result[key].item()
        # result = {key:value for (key, value) in ()}
        for key, value in temp_dict.items():
            temp_dict[key] /= len(loop)
        return temp_dict
    
def detect_while_training_iou(batch, model, prob_thresh = 0.3, new = True):
    image, _, full_boxes = batch
    B, C, H, W = image.shape
    # image, cls, regr = image.cuda(), cls.cuda(), regr.cuda()
    image = image.cuda()

    copy_image = torch.permute(image.squeeze(0), dims = (1,2,0)).detach().cpu().numpy().copy()

    detector = TextDetector()
    if new:
        with torch.no_grad():
            pred_cls, pred_regr = model(image)
            text, score = detector((pred_regr, pred_cls), (H, W))
    else:
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
    copy_image = np.array(copy_image)
    """
    for line in new_text:
        line = [int(j) for j in line]
        predicted[line[1]:line[3], line[0]:line[2]] = 1
    """
    for line in text:
        line = [int(j) for j in line]
        #logger.info(line)
        min_x, min_y, max_x, max_y = line[0], line[1], line[2], line[3]
        cv2.rectangle(copy_image, (min_x, min_y), (max_x, max_y), (0,255, 0), 2)
        predicted[min_y:max_y,min_x:max_x] = 1
        
    cv2.imwrite("/home/ubuntu/user/jihye.lee/ocr_exp_v1/predicted.png", copy_image)
    logger.info("PREDICTED")
    
    predicted = predicted == 1
    for box in full_boxes.squeeze(0).squeeze(0).numpy():
        #logger.info(box)
        min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        answer[min_y:max_y, min_x:max_x] = 1
        # print((min_x, min_y, max_x, max_y))
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
        
        