## contains the functions needed for generating the dataset
# 원래는 QUARD만 사용하려 했었는데 PVANET + RBOX가 성능이 제일 좋았음이 증명되었기에 RBOX를 사용할 예정이다.
import numpy as np
import math
import json
from shapely.geometry import Polygon
import cv2
from PIL import Image

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

