import torch
import torch.nn as nn
from tqdm import tqdm
import os, sys
import mlflow
import detect_model as DetectModel ## 모델 이름과 model_configuration을 입력으로 넣어줌
import losses as DetectLoss ## 손싷함수의 이름과 각각의 가중치를 위해서 train_configuration을 입력으로 넣어줌
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from tools.base_trainer import BaseTrainer
from loguru import logger
ARTIFACT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
## DETECTION TRAINER ##

optimizer_registry = {
    "ADAM": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

## 만약에 Adam optimizer을 사용한다면 learning rate에 momentum이 적용이 되어서 굳이 
# learning rate scheduler이 필요하지 않을 수도 있다.
scheduler_registry = {
    "MULTISTEP": torch.optim.lr_scheduler.MultiStepLR,
    "LINEAR": torch.optim.lr_scheduler.LinearLR,
    "COSINE": torch.optim.lr_scheduler.CosineAnnealingLR
}

class Trainer(BaseTrainer):
    def build(self):
        self.current_metric = {}
        ## build the model, optimizer, schduler, loss functions, etc ..
        mlflow.log_artifacts(os.path.join(ARTIFACT_DIR, 'text_detection'), artifact_path = "code")
        self.model = DetectModel.load_model(self.model_cfg['name'], self.model_cfg).cuda()
        self.criterion = DetectLoss.load_loss(self.train_cfg) ## 모델별로 지정된 손실 함수를 불로오기 위해서 사용
        
        self.optimizer = optimizer_registry[self.train_cfg['optimizer']['name'].upper()](
            params = self.model.parameters(), lr = self.train_cfg['optimizer']['lr']
        )
        ## 당분간은 scheduler은 사용하지 않기로 하자
        


    def run(self, train_dataloader, eval_dataloader):
        logger.info("===> connected to detection trainer ===>")
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.losses = []
        self.model.train()
        for epoch in range(self.total_epochs):
            train_loop = tqmd(self.train_dataloader)
            for idx, batch in enumerate(train_loop):
                img, gt_score = batch
                pred_score, pred_geo = self.model(img) 
                ## (B, 1, W, H) (B, 5, W, H)
                
                loss = self.criterion[0](pred_score, bbox)
                
                
                
        return
    def start_first_epoch(self, current_epoch):
        pass
    def save(self,  last = False):
        '''
        Function for saving the model weights if best model or if it is the last epoch
        '''
        pass
    def evaluate(self, root_path= './results'):
        self.model.eval()
        with torch.no_grad():
            loop = tqdm(self.eval_dataloader)
            for idx, batch in enumerate(loop):
                img, bbox = batch
        pass
        