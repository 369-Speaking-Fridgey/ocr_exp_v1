import torch
import torch.nn as nn
import torch.nn.functional as F

class CTPNLoss(nn.Module):
    def __init__(self):
        super(CTPNLoss, self).__init__()
    def forward(self, input, target):
        pass