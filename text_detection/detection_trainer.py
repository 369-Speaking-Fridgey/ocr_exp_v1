import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import os, sys
import datetime
import numpy as np
import mlflow
import random
import detect_model as DetectModel ## 모델 이름과 model_configuration을 입력으로 넣어줌
import losses as DetectLoss ## 손싷함수의 이름과 각각의 가중치를 위해서 train_configuration을 입력으로 넣어줌
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from tools.base_trainer import BaseTrainer
from loguru import logger
from torchmetrics.functional import precision_recall
from torchmetrics.functional import f1_score as F1
from utils.east_utils import east_detect
from utils.ctpn_utils import ctpn_detect

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
now = datetime.datetime.now()
TODAY = str(now.strftime('%Y-%m-%d %H:%M:%S'))
class Trainer(BaseTrainer):
    def build(self):
        self.experiment_number = TODAY
        self.current_metric = {}
        logger.info(
            f"ARTIFACT DIR: {ARTIFACT_DIR}"
        )
        self.eval_epoch = self.train_cfg['eval_epoch']
        self.total_epochs = self.train_cfg['epoch'] + self.train_cfg['eval_epoch']
        ## build the model, optimizer, schduler, loss functions, etc ..
        mlflow.log_artifacts(os.path.join(ARTIFACT_DIR, 'text_detection'), artifact_path = "code")
        self.model = DetectModel.load_model(self.model_cfg['model_name'], self.model_cfg).cuda()
        
        ## (1) LOAD THE PRETRAINED MODEL WEIGHTS
        if self.model_cfg['pretrained_model'] != '':
            pretrained = torch.load(self.model_cfg['pretrained_model'])
            org = self.model.state_dict()
            new = {key:value for key, value in pretrained.items() if key in org and \
                            value.shape == pretrianed[key].shape}
            org.update(new)
            self.model.load_state_dict(org)
            
        self.criterion, self.lamda = DetectLoss.load_loss(self.train_cfg) ## 모델별로 지정된 손실 함수를 불로오기 위해서 사용
        
        self.optimizer = optimizer_registry[self.train_cfg['optimizer'].upper()](
            params = self.model.parameters(), lr = self.train_cfg['learning_rate']
        )
        ## 당분간은 scheduler은 사용하지 않기로 하자
        
        self.best_metric_dict = {}
        self.current_metric_dict = {}

    def run(self, train_dataloader, eval_dataloader):
        self.build()
        logger.info("===> connected to detection trainer ===>")
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.losses = {}
        self.model.train()
        for epoch in range(self.total_epochs):
            epoch_loss = 0.0
            train_loop = tqdm(self.train_dataloader)
            for idx, batch in enumerate(train_loop):
                
                if self.model_cfg['model_name'].upper() == 'EAST':
                    img, gt_score, gt_geo, gt_ignore = batch
                    
                    img, gt_score, gt_geo, gt_ignore = img.cuda(), gt_score.cuda(), gt_geo.cuda(), gt_ignore.cuda()
                    pred_score, pred_geo = self.model(img) 
                    ## (B, 1, W, H) (B, 5, W, H)
                    loss = self.criterion[0](gt_score, pred_score, gt_geo, pred_geo, gt_ignore)
                elif self.model_cfg['model_name'].upper() == 'CTPN':
                    pass
                img = img.detach().cpu().numpy()
                # print(img.shape)
                for b in range(img.shape[0]):
                    save_img = (((img[b,:,:,:].transpose(1, 2, 0) * 0.5) + 0.5) * 255).astype(np.uint8)
                    save_img = np.mean(save_img, axis = 2) 
                    cv2.imwrite(os.path.join('/home/ubuntu/user/jihye.lee/ocr_exp_v1/text_detection/results', f"{b}.png"),save_img)
                    break
                epoch_loss += loss.item()
                if loss.item() == 0:
                    continue
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                train_loop.set_postfix({
                    "Loss": loss.item(),
                    "Epoch": epoch,
                    "SCORE L1 Loss": F.l1_loss(input = gt_score, target = pred_score).item(),
                    "GEO L1 Loss": F.l1_loss(input = gt_geo, target = pred_geo).item()
                })
                
                
            self.losses[epoch] = epoch_loss
            if (epoch+1) % self.eval_epoch == 0:
                self.validate()
                self.save(last = False) ## 만약에 metric값, 즉 정확도가 향상이 되었다면 
                
        self.validate()
        self.save(last = True)            
        return
    
    
    def start_first_epoch(self, current_epoch):
        pass
    
    def save(self,  last = False):
        '''
        Function for saving the model weights if best model or if it is the last epoch
        '''
        os.makedirs(os.path.join(self.eval_cfg['weight'], self.experiment_number), exist_ok = True)
        if last:
            new_path = os.path.join(self.eval_cfg['weight'], self.experiment_number, 'last.pt')
            torch.save(self.model.state_dict(), new_path)
        else:
            new_path = os.path.join(self.eval_cfg['weight'], self.experiment_number, 'best.pt')
            torch.save(self.model.state_dict(), new_path)
            client = mlflow.tracking.MlflowClient()
            client.log_artifact()

    

    def evaluate(self, result_path = './samples'):
        ## 실제 bounding box를 detect한 다음에 
        os.makedirs('./samples', exist_ok = True)
        
    def validate(self, root_path= './results'):
        ## evaluate 단계에서는 전체 이미지를 crop이나 height adjust없이 넣어준다.
        self.model.eval()
        idxs = random.sample(range(1, len(self.eval_dataloader)), 5)
        with torch.no_grad():
            loop = tqdm(self.eval_dataloader)
            pr_score, pr_geo, f1_score, f1_geo = 0.0, 0.0, 0.0, 0.0
            for idx, batch in enumerate(loop):
                if self.model_cfg['model_name'].upper() == 'EAST':
                    img, gt_score, gt_geo, gt_ignore = batch
                    #img, vertices = batch
                    #img, vertices = img.cuda(), vertices.cuda()
                    img, gt_score, gt_geo, gt_ignore = img.cuda(), gt_score.cuda(), gt_geo.cuda(), gt_ignore.cuda()
                    pred_score, pred_geo = self.model(img)
                    if idx in idxs:
                        east_detect.detect_sample(pred_score, pred_geo, img)
                
                    pr_score += precision_recall(preds = pred_score, target = gt_score, average = 'micro')
                    pr_geo += precision_recall(preds = pred_geo, target = gt_geo, average = 'micro')
                    f1_score += F1(preds = pred_score, target = gt_score, average = 'micro')
                    f1_geo += F1(preds =  pred_geo, target = gt_geo, average = 'micro')
                
                    self.current_metric_dict = {
                        "Score Precision Recall": pr_score / len(loop),
                        "Geo Precision Recall": pr_geo / len(loop),
                        "Score F1": f1_score / len(loop),
                        "Geo F1": f1_geo / len(loop)
                        }   
                    loop.set_postfix(self.current_metric_dict)
                
                elif self.model_cfg['model_name'].upper() == 'CTPN':
                    pass

        