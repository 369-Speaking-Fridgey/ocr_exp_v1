import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, 
                 data_cfg, 
                 model_cfg,
                 mlops_cfg,
                 train_cfg):
        super(BaseTrainer, self).__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.mlops_cfg = mlops_cfg
        self.train_cfg = train_cfg
        
    def setup(self):
        pass
    def run(self, train_dataloader, test_dataloader, ):
        pass
    def evaluate(self, eval_dataloader);
        pass
    def run_one_epoch(self, epoch, ):
        pass
    def save(self, is_last = False):
        pass