import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os, sys
import mlflow
ARTIFACT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ARTIFACT_DIR)
## mlflow를 연결하기 위해서 원래대로라면 registered uri를 지정해 주었어야 했는데 그러지 못했음

class ManagedMLFlow:
    def __init__(self, experiment_name, run_name, user_name, tracking_uri):
        super(ManagedMLFlow, self).__init__()
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        try:
            self.experiment_id = mlflow.create_experiment(name = self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(name = self.experiment_name).experiment_id

        self.run = mlflow.start_run(
            run_name = self.run_name,
            experiment_id = self.experiment_id
        )
        
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
        
        self.exp_setup()
        
    def exp_setup(self):
        self.mlops = ManagedMLFlow(
            experiment_name = self.mlops_cfg['experiment_name'],
            run_name = self.mlops_cfg['run_name'] + f" {self.model_cfg['model_name']}",
            user_name = self.mlops_cfg['user_name'],
            tracking_uri = self.mlops_cfg['tracking_uri']
        ) ## setup the mlflow 
        mlflow.log_params(self.train_cfg)
        mlflow.log_params(self.model_cfg)
        mlflow.log_params(self.data_cfg)
        #mlflow.log_params({'train_cfg': self.train_cfg,
        #                   'model_cfg': self.model_cfg,
        #                   'data_cfg': self.data_cfg})
        mlflow.log_artifacts(os.path.join(ARTIFACT_DIR, 'text_detection'), artifact_path="codes")

    
    def build(self):
        pass

    def run(self, train_dataloader, eval_dataloader):
        self.build()
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.total_epochs = self.train_cfg['epochs']
        self.eval_epochs = self.train_cfg['eval_epoch']
        pass
    def evaluate(self, eval_dataloader):
        pass
    def run_one_epoch(self, epoch, ):
        pass
    def save(self, is_last = False):
        pass
    
    
