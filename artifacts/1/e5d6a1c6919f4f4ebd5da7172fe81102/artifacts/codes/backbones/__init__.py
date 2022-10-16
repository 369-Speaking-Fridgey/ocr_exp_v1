import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import vgg
import resnet
from pva import PVANet
import torch.nn as nn
def make_layers(branch_name, pretrained = True):

    if 'vgg' in branch_name: ## vgg계열의 모델인 경우
        func = getattr(vgg, branch_name)
        model = func(pretrained)
        return model
    elif 'resnet' in branch_name: ## resnet계열의 모델인 경우
        func = getattr(resnet, branch_name)
        model = func(pretrained)
        layers = nn.ModuleDict()
        for name, m in model.named_children():
            if name == 'avgpool':
                break
            else:
                layers[name] = m
        model = layers
        return model
    elif 'pva' in branch_name:
        model = PVANet()
        return model


def get_channels(branch_name):
    if 'vgg' in branch_name:
        return [128, 256, 512, 512]
    elif 'resnet' in branch_name:
        return [256, 512, 1024, 2048]
    elif 'pva' in branch_name:
        return [64, 128, 256, 384]
    
        
if __name__ == "__main__":
    make_layers('vgg19_bn')