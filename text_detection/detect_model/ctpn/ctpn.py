import torch
import torch.nn as nn
from torchvision import models



class BasicConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride,
                 bn = True, bias = True, relu = True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, padding = (kernel_size-1)//2,
                              stride = stride, bias = bias)
        self.bn = nn.BatchNorm2d(ch_out, eps = 1e-5, momentum = 0.01, affine = True) if bn else None
        self.relu = nn.ReLU(inplace = True) if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class CTPN(nn.Module):
    def __init__(self, **kwargs):
        super(CTPN, self).__init__()
        vgg = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
        self.base_model = nn.Sequential(*list(vgg.features)[:-1])
        self.rpn = BasicConv(512,512, kernel_size = 3, stride = 1, bn = False)
        self.brnn = nn.GRU(512, 128, bidirectional = True, batch_first = True)
        self.lstm_fc = BasicConv(256, 512, kernel_size = 1, stride = 1, relu = True, bn = False)
        
        self.vertical_cord = BasicConv(512, 10 * 4, kernel_size = 1, stride = 1, relu = False, bn = False)
        self.score = BasicConv(512, 10 * 2, kernel_size = 1, stride = 1, relu = False, bn = False)
        self.side_refinement = BasicConv(512, 10, kernel_size = 1, stride = 1, relu = False, bn = False)
        
    def forward(self, x):
        ## (B, C, H, W)
        x = self.base_model(x)
        x = self.rpn(x) ## (B, 512, H', W') -> 이렇게 vgg16의 feature 추출 layer에서의 output을 사용한다.

        x1 = x.permute(0, 2, 3, 1).contiguous() ## (B, C, H, W) -> (B, H, W, C)
        B, H, W, C = x1.size()
        x1 = x1.view(B * H, W, C) ## (B*H, W, C)
        x2, _ = self.brnn(x1)
        x3 = x2.view(x.size(0), x.size(2), x.size(3), 256) ## (B, H', W', 256)
        
        x3 = x3.permute(0, 3, 1, 2).contiguous() ## (B, 256, H', W')
        x3 = self.lstm_fc(x3) ## (B, 512, H', W')
        x = x3
         
        vertical_pred = self.vertical_cord(x) ## (B, 40, H', W')
        score = self.score(x) ## (B, 20, H', W')
        side_refinement = self.side_refinement(x) ## (B, 10, H', W')
        """
        - score: text/nontext score
        - vertical_pred: vertical coordinates
        - side_refinement: side-refinement offset
        """
        return score, vertical_pred, side_refinement