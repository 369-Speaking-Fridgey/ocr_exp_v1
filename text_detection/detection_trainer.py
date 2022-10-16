import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from tools.base_trainer import BaseTrainer
from loguru import logger

## DETECTION TRAINER ##

class Trainer(BaseTrainer):
    def build(self):
        ## build the model, optimizer, schduler, loss functions, etc ..
        pass
    def run(self, train_dataloader, eval_dataloader):
        logger.info("===> connected to detection trainer ===>")
        return
        