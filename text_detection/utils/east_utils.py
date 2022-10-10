import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import lanms
import math
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_model.east.east import EAST

def resize_img(image):
    ## 이미지의 각 변의 길이가 32로 나누어 떨어질 수 있도록 한다.
    w, h = image.size
    resize_w = w
    resize_h = h
    
    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    image = image.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h, ratio_w = resize_h / h, resize_w / w
    return image, ratio_h, ratio_w

def pil_to_tensor(image):
    ## input image should be a PIL image
    aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
    ])
    return aug(image)

def is_valid_poly(res, score_shape, scale):
    ## 올바른 polygon모양이 맞는지 확인하는 역할을 한다.
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False

def get_rotate_mat(theta):
    return np.array([[
        math.cos(theta), -math.sin(theta)], 
                     [math.sin(theta), math.cos(theta)]
    ])
    
def restore_polys(valid_pos, valid_geo, score_shape, scale = 4):
    polys, index = [], []
    valid_pos *= scale
    d = valid_geo[:4, :]
    angle = valid_geo[4, :]
    for i in range(valid_pos.shape[0]):
        x, y = valid_pos[i, 0], valid_pos[i, 1]
        y_min, y_max = y - d[0, i], y + d[1, i]
        x_min, x_max = x - d[2, i], x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])
        
        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordinates = np.concatenate((temp_x, temp_y), axis = 0)
        res = np.dot(rotate_mat, coordinates)
        res[0, :] += x
        res[1, :] += y
        
        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1,3]])
    return np.array(polys), index

def get_boxes(score, geo, score_thresh = 0.9, nms_thresh = 0.2):
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None
    
    boxes = np.zeros((polys_restored.shape[0], 9), dtype = np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes

def adjust_ratio(boxes, ratio_w, ratio_h):
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0,2,4,6]] /= ratio_w
    boxes[:, [1,3,5,7]] /= ratio_h
    return np.around(boxes)

def detect(image, model, device):
    image, ratio_h, ratio_w = resize_img(image)
    model.cuda()
    with torch.no_grad():
        image= pil_to_tensor(image)
        if len(image.shape) != 4:
            image = image.unsqueeze(0)
        image = image.cuda()
        score, geo = model(image)
    boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
    return adjust_ratio(boxes, ratio_h, ratio_w)


if __name__ == "__main__":
    model = EAST('vgg16')
    image = Image.open('/home/ubuntu/user/jihye.lee/data/CORD-1k-001/CORD/dev/image/receipt_00000.png').convert('RGB')

    boxes = detect(image, model, device = torch.cuda.device(0))
    print(len(boxes), len(boxes[0]))
    ## 총 281개의 bounding box를 감지하였고, 8개의 꼭짓점과 각각이 글자일 점수, 혹은 확률이 저장되어 있다.