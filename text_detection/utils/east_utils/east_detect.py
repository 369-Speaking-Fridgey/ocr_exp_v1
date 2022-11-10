import torch
from torchvision import transforms
from PIL import Image
import os
from utils.east_utils.geo_map_utils import get_rotate_mat
import numpy as np
import lanms


## (1) LOAD THE IMAGE (PIL -> TENSOR)
def load_image(image_path):
    image = Image.open(image_path)
    aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))    
    ])
    ## Batch의 크기가 1이면 차원의 수가 3개인데 그러면 모델에 입력이 불가능하다.    
    return aug(image).unsqueeze(0)

## (2) RESTORE POLYS FROM FEATURE MAPS IN GIVEN POSITIONS 
def restore_polys(position, geo_map, score_shape, scale = 4):
    polys, index = [], []
    position *= scale ## EAST 모델의 경우에는 output feature map의 크기가 처음에 input의 1/4배이다.
    d = geo_map[:4, :]
    angle = geo_map[4, :]
    
    for i in range(position.shape[0]):
        x = position[i, 0]
        y = position[i, 1]
        y_min, y_max = y - d[0, i], y + d[1, i]
        x_min, x_max = x - d[2, i], x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])
        
        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordinates = np.concatenate((temp_x, temp_y), axis = 0)
        res = np.dot(rotate_mat, coordinates)
        res[0, :] += x
        res[1, :] += y
        
        if valid_poly_check(res, score_shape, scale):
            index.append(i)
            polys.append([
                res[0,0], res[1,0], res[0,1], res[1,1], \
                    res[0,2], res[1,2], res[0,3], res[1,3]
            ])
            
    return np.array(polys), index
        
## (3) CHECK IF THE POLYGON IS VALID
def valid_poly_check(result, score_shape, scale):
    """ Check if the poly is in the image scope
    result: restored poly in the original image
    scale: feature map * scale = original image
    """
    cnt = 0
    for i in range(result.shape[1]):
        if result[0, i] < 0 or result[0, i] >= score_shape[1] * scale or \
            result[1, i] < 0 or result[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False        

## (4) GET THE BOUNDING BOX FROM THE FEATURE MAP
def get_boxes(score, geo, score_thresh, nms_thresh):
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh) ## text에 해당하는 점수가 일정 threshold 이상인 경우에
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]
    restored_poly, index = restore_polys(valid_pos, valid_geo, score.shape,  scale = 4)
    
    if restored_poly.size == 0:
        return None
    
    boxes = np.zeros((restored_poly.shape[0], 9), dtype = np.float32)
    boxes[:, :8] = restored_poly ## 앞의 8개는 (x1, y1, x2, y2, x3, y3, x4, y4)의 정보를 포함한다.
    boxes[:, 8] = score[xy_text[:, 0], xy_text[:, 1]]
    
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes
    
## (5) DETECT
def detect(image_path, model, device, score_thresh = 0.9, nms_thresh = 0.2):
    image = Image.open(image_path)
    image, ratio_h, ratio_w = resize_image(image)
    with torch.no_grad():
        score, geo = model(load_image(image_path).to(device))
    
    box = get_boxes(score, geo, score_thresh, nms_thresh)
    