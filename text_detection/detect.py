import torch
from torchvision import transforms
from PIL import Image
import os
from utils.east_utils.geo_map_utils import get_rotate_mat
import numpy as np
import lanms


## (1) LOAD THE IMAGE
def load_image(image):
    