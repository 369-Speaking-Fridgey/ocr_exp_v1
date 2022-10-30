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

def cal_distance(x1, y1, x2, y2):
    ```Calculate the Euclidian Distance (=L2 Distance)```
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
    rotated = cv2.warpAffine(crop_img, M, (W, H))
    
    return rotated, theta
    
    
## 우선은 AI HUB의 데이터를 사용한다고 생각을 해도 나쁘지 않을 것이다. 적어도 text detection & recognition에 관해서는 말이다.
##################EAST DATASET UTILS######################################
##========(1) Extract Vertices==========##
def extract_vertices(file_url):
    ```Extract the vertices from the json & txt files```
    response = requests.get(file_url)
    with open(BytesIO(response.content), 'r') as f:
        json_file = json.load(f)
    
    vertices = []
    for item in json_file['text']['word']:
        bbox = item['wordbox'] ## bbox가 4개의 좌표로 주어지는데, 그 이유는 휘어지거나 point of view가 평행하지 않은 경우가 없기 때문이다.
        
        value = item['value']
        if value.isalpha() == True:
            vertices.append(bbox)
        else:
            
            
            
        
        
    
##========(2) Adjust Height=============##
def adjust_height(image, vertices):
    pass
##========(3) Rotate Image==============##
def rotate_img(image, vertices):
    pass
##========(4) Crop Image================##
def crop_img(image, vertices):
    pass
##========(5) Make ground truth map=====##
def make_geo_score():
    """
    - Function to make the geometry and score map ground truth
    """
    pass

