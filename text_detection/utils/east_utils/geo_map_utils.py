## contains the functions needed for generating the dataset
## 이 함수들은 모두 EAST 모델의 데이터셋을 만들기 위해서 필요하다.
## 근데 당분간은 GEO map는 사용하지 않을 계획이기 때문에...
"""
- bounding box, 혹은 4개의 (x,y)로 이루어진 좌표가 주어진다면 이를 사용해서 angle + Bounding Box로 바꾸기 위한 파일이다.
- 여기 있는 함수들은 주로 dataset을 만들기 위해서 필요로 한다. 
- 즉, EAST 모델이 grount truth로서 학습시키기 위해서 필요한 것들을 만들어 준다
"""
# 원래는 QUARD만 사용하려 했었는데 PVANET + RBOX가 성능이 제일 좋았음이 증명되었기에 RBOX를 사용할 예정이다.
import math
import json
import random
import json
import numpy as np
from shapely.geometry import Polygon
import requests
from io import BytesIO
import cv2
import os
from PIL import Image


def shrink_poly(vertices, coef = 0.3):
    '''
    Shrink the text region
    vertices: vertices of text region <numpy.ndarray, (8, )>
    coef: shrink ratio
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x4, y4), cal_distance(x3, y3, x2, y2))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1,r2,r3,r4]
    
    ## 세로가 더 짧은지 가로가 더 짧은지 비교를 해 주어야 한다.
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > cal_distance(x2, y2, x3, y3) + cal_distance(x4, y4, x3, y3):
        
    
def cal_distance(x1, y1, x2, y2):
    '''Calculate the Euclidian Distance (=L2 Distance)'''
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist

def get_rotate_mat(theta):
    ```positive theta value means rotate clockwise```
    return np.array([
        [math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]
    ])
    

def add_rotation(img, max_theta):
    theta = random.randint(0, max_theta)
    minus = random.randint(1, 3)
    theta = theta * -1 if minus == 1 else theta
    H, W, C = img.shape
    cX, cY = W // 2, H // 2
    M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
    rotated = cv2.warpAffine(crop_img, M, (W, H)) ## 이건 cv2로 이미지를 읽을 때에 사용하게 된다
    
    return rotated, theta
    
    
## 우선은 AI HUB의 데이터를 사용한다고 생각을 해도 나쁘지 않을 것이다. 적어도 text detection & recognition에 관해서는 말이다.
##################EAST DATASET UTILS######################################
##========(1) Extract Vertices==========##
def extract_vertices(file_url, symbol = False):
    '''
    [left_x, top_y, right_x, bot_y]
    -> [left_x, top_y, right_x, top_y, right_x, bot_y, left_x, bot_y] 
    == [x1, y1, x2, y2, x3, y3, x4, y4]
    '''
    
    # 만약에 symbol = False로 둔다면 
    ```Extract the vertices from the json & txt files```
    response = requests.get(file_url)
    ## get file url of the image
    with open(BytesIO(response.content), 'r') as f:
        json_file = json.load(f)
    
    vertices = []
    labels = []
    for item in json_file['text']['word']:
        bbox = item['wordbox'] ## bbox가 4개의 좌표로 주어지는데, 그 이유는 휘어지거나 point of view가 평행하지 않은 경우가 없기 때문이다.
        
        value = item['value']
        if len(bbox) == 4: ## AIHUB의 데이터에서의 bbox의 경우에는 수평이 잘 맞기 때문에 4개의 좌표의 정보만 저장이 되어 있는 경우가 있다.
            ## 이런 경우에는 8point로 이루어진 vertice의 원래 성격에 맞게 변형을 해 주어야 한다.
            left_x, top_y, right_x, bot_y = bbox
            new_bbox = [
                left_x, top_y, right_x, top_y, right_x, bot_y, left_x, bot_y
            ]
            vertices.append(new_bbox)
        else:
            vertices.append(bbox)
        labels.append(1)
    return np.array(vertices), np.array(labels)
            

    
##========(2) Adjust Height=============##
def adjust_height(image, vertices, ratio = 0.2):
    '''
    adjust the height of image to augment the data
    '''
    ratio_h = 1 + ratio * (np.random.rand()* 2 - 1) 
    old_h = image.height
    new_h = int(np.round(ratio_h * old_h))
    image = image.resize((img.width, new_h), Image.BILINEAR)
    
    new_vertices = vertices.copy()
    bbox = False
    if len(vertices[0]) == 4:
        bbox = True
    if vertices.size > 0:
        if bbox:
            new_vertices[:, :] = vertices[:, :] * (new_h / old_h)
        else:
            new_vertices[:, [1,3,5,6]] = vertices[:, [1,3,5,6]]] * (new_h / old_h)
    return image, new_vertices
    
##========(3) Rotate Image==============##
def rotate_vertices(vertice, theta, anchor = None):
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v-anchor)
    
    return  (res + anchor).T.reshape(-1)

def rotate_img(image, vertices, angle_range = 10):
    '''
    image: PIL Image
    vertices: vertices of text regions <numpy.ndarray, (n, 8)> -> but AIHub data uses (n, 4) text regions
    angle_range: rotate range [-angle_range, angle_range]
    '''
    center_x, center_y = (image.width - 1) / 2, (image.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    image = image.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
        
    return image, new_vertices
##========(4) Crop Image================##

def crosses_text(start_loc, length, vertices):
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False

def crop_img(image, vertices, labels, length):
    """
    - batch size를 크게 유지해야 하는데 그렇게 하기 위해서는 이미지를 적당한 크기의 patch로 crop해 줄 수 있어야 한다.
    
    """
    H, W = image.height, image.width
    if H >= W and W < length:
        image = image.resize((length, int(H * length / W)), Image.BILINEAR) ## Width를 length로 바꾸어줌
    elif H < W and H < length:
        image = image.resize((int(W * length / H), length), Image.BILINEAR) ## Height를 length로 바꾸어줌
    
    ratio_w = image.width / W
    ratio_h = image.height / H
    
    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h
    
    ## FIND RANDOM POSITION TO START CROPPING
    remain_h = inage.height - length
    remain_w = image.width - length
    flag = True
    cnt = 0
    
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = crosses_text([start_w, start_h], length, new_vertices[labels==1. :])  ## DONE
        
    box = (start_w, start_h, start_w + length, start_h + length)
    region = image.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices
    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    
    return region, new_vertices
    
    
    
    
        
##========(5) Make ground truth map=====##
def get_boundary(vertices):
    '''
    get the tight boundary that is rectangle shaped around the given vertices
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    
    return x_min, x_max, y_min, y_max

def cal_error(vertices):
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    
    return err

def find_min_rect_angle(vertices):
    '''
    find the best angle to rotate poly and obtain min rectangle
    vertices: shape:(8,)
    output: the best angle (radian measure)
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)
        
    sorted_area_index = sorted(list(range(len(area_list))), key = lambda k: area_list[k])
    min_error = float('inf')
    best_index, rank_num = -1, 10
    ## Find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi) ## DONE
        temp_error = cal_error(rotated) ## DONE
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
            
    return angle_list[best_index] / 180 * math.pi

def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''
    get rotated locations of all pixels
    anchor_x: fixed x position
    anchor_y: fixed y position
    rotated_x: rotated x positions | shape = (length, length)
    rotated_y: rotated y positions | shape = (length, length)
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, y.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + np.array([[anchor_x], [anchor_y]])
    
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    
    return rotated_x, rotated_y

def make_geo_score(img, vertices, labels, scale, length):
    """
    - Function to make the geometry and score map ground truth
    image: PIL Image
    vertices: vertices of text regions <numpy.ndarray, (n, 8)>
    labels: 1-> valid 0 -> invalid
    scale: feature map / image
    length: image length
    """
    score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    
    index = np.arange(0, length, int(1/scale))
    ## 원래 bounding box에서의 중심점과 
    index_x, index_y = np.meshgrid(index, index)
    ignored_polys = []
    polys = []
    
    for i, vertice in enumerate(vertices):
        if labels[i] == 0: ## valid한 text를 포함하는 text box가 아닌 경우에
            ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
            continue

        
        ## poly는 ((x1, y1), (x2, y2), (x3, y3), (x4, y4))의 형태를 띄어야 한다.
        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32) ## scaled and shrinked
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)
        
        theta = find_min_rect_angle(vertice) ## DONE
        rotate_mat = get_rotate_mat(theta) ## DONE
        
        rotated_vertices = rotate_vertices(vertice, theta, anchor = None) ## DONE
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices) ## DONE
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length) ## DONE
        
        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += temp_mask * theta
    
    cv2.fillPoly(ignored_map, ignored_polus, 1)
    cv2.fillPoly(score_map, polys, 1)
    
    return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo_map).permute(2, 0, 1), torch.Tensor(ignored_map).permute(2, 0, 1)
        
        
            
    
    

