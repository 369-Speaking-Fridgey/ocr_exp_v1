from torchvision import models
import torch
import torch.nn as nn

def resnet50(pretrained = True):
    if pretrained == False:
        model = models.resnet50()
    else:
        model = models.resnet50(weights = models.ResNet50_Weights)
    return model 

def resnet101(pretrained = True):
    if pretrained == False:
        model = models.resnet101()
    else:
        model = models.resnet101(weights = models.ResNet101_Weghts)
    return model

def resnet152(pretrained = True):
    if pretrained == False:
        model = models.resnet152()
    else:
        model = models.resnet152(weights = models.ResNet152_Weights)
    return model

