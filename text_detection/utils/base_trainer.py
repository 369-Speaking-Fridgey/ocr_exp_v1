import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseTrainer:
    def __init__(self, 
                 data_cfg, 
                 model_cfg,
                 mlops_cfg,
                 train_cfg):
        