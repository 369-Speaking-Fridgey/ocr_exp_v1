import torch
import torch.nn as nn
import os, sys

from detect_model.__init__ import load_model ## 모델 이름과 model_configuration을 입력으로 넣어줌
from losses.__init__ import load_loss, calculate_loss ## 손싷함수의 이름과 각각의 가중치를 위해서 train_configuration을 입력으로 넣어줌
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
        