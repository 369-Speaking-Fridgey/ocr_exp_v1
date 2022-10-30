## contains the functions needed for generating the dataset
## 이 함수들은 모두 EAST 모델의 데이터셋을 만들기 위해서 필요하다.
## 근데 당분간은 GEO map는 사용하지 않을 계획이기 때문에...
# 원래는 QUARD만 사용하려 했었는데 PVANET + RBOX가 성능이 제일 좋았음이 증명되었기에 RBOX를 사용할 예정이다.
import math
import json
import numpy as np
from shapely.geometry import Polygon
import cv2
import os
from PIL import Image

def cal_distance(x1, y1, x2, y2):
    '''Calculate the Euclidian Distance (=L2 Distance)'''
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist

def get_rotate_mat(theta):
    return np.array([
        [math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]
    ])
##################EAST DATASET UTILS######################################
##========(1) Extract Vertices==========##
def extract_vertices(file_path):
    pass
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

