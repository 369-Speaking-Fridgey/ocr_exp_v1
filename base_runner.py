import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf, DictConfig
import os, sys
from loguru import logger
from torch.utils.data import DataLoader
import argparse
import importlib

mode_registery = {
    0: ('text_detection', 'east_detect_config.yml', 
        'text_detection.detection_dataset', 'text_detection.detection_trainer'),
    1: ('text_recognition', 'recog_config.yml'),
    2: ('key_info_extraction', 'kie_config.yml')
}

## loss function도 여러개를 다른 lambda ratio로 사용하게 될 수 있으니 둘다 list의 형태로 받아야 한다
class TrainerEntry():
    def __init__(self):
        self.cfg = None
    
    def __call__(self, cfg:DictConfig) -> None:
        model_cfg = OmegaConf.to_container(cfg['model_configuration'])
        data_cfg = OmegaConf.to_container(cfg['data_configuration'])
        eval_cfg = OmegaConf.to_container(cfg['eval_configuration'])
        train_cfg = OmegaConf.to_container(cfg['train_configuration'])
        mlops_cfg = OmegaConf.to_container(cfg['mlops_configuration'])
        
        ## (1) LOAD THE TRAINER
        trainer = importlib.import_module(mode_registery[model_cfg['mode']][3]).Trainer(
            data_cfg, model_cfg, mlops_cfg, train_cfg
        )
        
        ## (2) LOAD THE DATASET & DATALOADER
        dataset = importlib.import_module(mode_registery[model_cfg['mode']][2])
        train_dataset_base = dataset.DATASET(model_cfg['model_name'], data_cfg, mode = 'train')
        train_dataset = train_dataset_base.get()
        train_dataloader = DataLoader(train_dataset, batch_size = data_cfg['batch_size'], shuffle = True)
        
        eval_dataset_base = dataset.DATASET(model_cfg['model_name'], data_cfg, mode = 'eval')
        eval_dataset = eval_dataset_base.get()
        eval_dataloader = DataLoader(eval_dataset, batch_size = 1, shuffle = False)
        
        test_dataset_base = dataset.DATASET(model_cfg['model_name'], data_cfg, mode = 'test')
        test_dataset = test_dataset_base.get()
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
        ret = trainer.run(
            train_dataloader, eval_dataloader
        )
        trainer.mlops.end_run()
        
        return ret



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=int, default = 0, help="Which module needed for training")
    parser.add_argument('--name', type=str, default = "ctpn", help="Which model to train")
    args = vars(parser.parse_args())
    trainer_entry = TrainerEntry()
    config_name = mode_registery[args['mode']][1]
    if args['name'] == 'ctpn':
        config_name = config_name.replace('east', 'ctpn')
    logger.info(f"CONFIG NAME: {config_name}")
    pipe = hydra.main(
        config_path = 'config', config_name = config_name#  mode_registery[args['mode']][1]
    )(trainer_entry.__call__)
    pipe()
        
        

        