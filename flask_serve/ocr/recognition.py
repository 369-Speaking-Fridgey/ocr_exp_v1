import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(BASE))
from text_recognition.rec_model.hangulnet import HangulNet

class Recognition(object):
    def __init__(self):
        super(Recognition, self).__init__()