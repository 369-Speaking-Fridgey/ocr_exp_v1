import torch
import torch.nn as nn

class DetectTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.epochs = 0
        
        