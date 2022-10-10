import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os, sys
import math
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE)
from backbones import make_layers, get_channels

class double_conv(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in + ch_mid, ch_mid, kernel_size = 1),
            nn.BatchNorm2d(ch_mid), nn.GELU(),
            nn.Conv2d(ch_mid, ch_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(ch_out), nn.GELU()
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    

"""
- Backbone으로 어떤 모델을 사용하는지에 따라서 결과적으로 CRAFT의 decoder이 사용하게 될 channel 수가 달라질 것이다.
- VGG
- ResNet
- PVANet & PVANetx2
"""
##===========================Extractor Layer=================================##
class extractor(nn.Module):
    def __init__(self, branch_name, pretrained):
        super(extractor, self).__init__()
        self.branch_name = branch_name
        self.features = make_layers(self.branch_name, pretrained)

    def forward(self, x):
        out = []
        if 'vgg' in self.branch_name:
            for m in self.features.features:
                x = m(x)
                if isinstance(m, nn.MaxPool2d):
                    out.append(x)
            return out[1:]
        elif 'resnet' in self.branch_name:
            for key, value in self.features.items():
                x = value(x)
                if 'layer' in key:
                    out.append(x)
        return out
    
##===========================Merge Layer=====================================##
class merge(nn.Module):
    def __init__(self, branch_name):
        super(merge, self).__init__()
        self.branch_name = branch_name
        self.out_dims = [128, 64, 32, 32]
        self.in_dims = get_channels(branch_name)
        
        ## FIRST BLOCK ##
        self.conv1 = nn.Conv2d(self.in_dims[-1] + self.in_dims[-2], self.out_dims[0], kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(self.out_dims[0])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_dims[0], self.out_dims[0], kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.out_dims[0])
        self.relu2 = nn.ReLU()
        
        ## SECOND BLOCK ##
        self.conv3 = nn.Conv2d(self.in_dims[-3] + self.out_dims[0], self.out_dims[1], kernel_size = 1)
        self.bn3 = nn.BatchNorm2d(self.out_dims[1])
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(self.out_dims[1], self.out_dims[1], kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(self.out_dims[1])
        self.relu4 = nn.ReLU()
        
        ## THIRD BLOCK ##
        self.conv5 = nn.Conv2d(self.in_dims[-4] + self.out_dims[1], self.out_dims[2], kernel_size = 1)
        self.bn5 = nn.BatchNorm2d(self.out_dims[2])
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(self.out_dims[2], self.out_dims[2], kernel_size = 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(self.out_dims[2])
        self.relu6 = nn.ReLU()
        
        ## LAST BLOCK ##
        self.conv7 = nn.Conv2d(self.out_dims[2], self.out_dims[3], kernel_size = 3, padding = 1)
        self.bn7 = nn.BatchNorm2d(self.out_dims[3])
        self.relu7 = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        ## unpool -> concat -> 1x1 -> 3x3
        y = F.interpolate(x[3], scale_factor = 2, mode = 'bilinear', align_corners = True)
        y = torch.cat((y, x[2]), 1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))
        
        y = F.interpolate(y, scale_factor = 2, mode = 'bilinear', align_corners = True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))
        
        y = F.interpolate(y, scale_factor = 2, mode = 'bilinear', align_corners = True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))
        
        y = self.relu7(self.bn7(self.conv7(y)))
        
        return y
        
##===========================Output Layer====================================##
class output(nn.Module):
    def __init__(self, scope = 512):
        super(output, self).__init__()
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = scope
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo = torch.cat((loc, angle), 1)
        return score, geo

##========================EAST Model==========================================##
class EAST(nn.Module):
    def __init__(self,branch_name, output_scope = 512, pretrained_bbone = True, freeze_bbone = False):
        super(EAST, self).__init__()
        self.extractor = extractor(branch_name, pretrained = pretrained_bbone)
        self.merge = merge(branch_name)
        self.output = output(scope = output_scope)
    def forward(self, x):
        return self.output(self.merge(self.extractor(x)))
    
    
if __name__ == "__main__":
    # layer = make_layers('vgg19')
    # print(layer)
    model = EAST(branch_name = 'resnet152')
    x = torch.randn(1, 3, 512, 512)
    score, geo = model(x)
    print(score.shape, geo.shape)        
        
        