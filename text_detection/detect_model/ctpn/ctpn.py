import torch
import torch.nn as nn
from torchvision import models
from loguru import logger


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
        try:
            vgg = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
        except:
            vgg = models.vgg16(pretrained = True)
        vgg.features[0].weight.requires_grad = False
        vgg.features[0].bias.requires_grad = False
        vgg.features[2].weight.requires_grad = False
        vgg.features[2].bias.requires_grad = False
            
        self.base_layers = nn.Sequential(*list(vgg.features)[:-1])
        
        
            
        self.rpn = BasicConv(512,512, kernel_size = 3, stride = 1, bn = False) ## Conv - ReLU
        self.brnn = nn.GRU(512, 128, bidirectional = True, batch_first = True) ## bidirectional=True로 했기 때문에 D=2라서 output shape가 2 * H_out이다.
        self.lstm_fc = BasicConv(256, 512, kernel_size = 1, stride = 1, relu = True, bn = False)
        
        self.rpn_class = BasicConv(512, 10 * 2, kernel_size = 1, stride = 1, relu = False, bn = False)
        self.rpn_regress = BasicConv(512, 10 * 2, kernel_size = 1, stride = 1, relu = False, bn = False)
        #self.vertical_cord = BasicConv(512, 10 * 4, kernel_size = 1, stride = 1, relu = False, bn = False)
        #self.score = BasicConv(512, 10 * 2, kernel_size = 1, stride = 1, relu = False, bn = False)
        #self.side_refinement = BasicConv(512, 10, kernel_size = 1, stride = 1, relu = False, bn = False)
        
    def forward(self, x):
        ## (B, C, H, W)
        x = self.base_layers(x)
        x = self.rpn(x) ## (B, 512, H', W') -> 이렇게 vgg16의 feature 추출 layer에서의 output을 사용한다.

        x1 = x.permute(0, 2, 3, 1).contiguous() ## (B, C, H, W) -> (B, H, W, C)
        B, H, W, C = x1.size()
        x1 = x1.view(B * H, W, C) ## (B*H, W, C) (sequence size, batch size, input size)
        x2, _ = self.brnn(x1) ## (B*H, W, 128 * 2)
        x3 = x2.view(x.size(0), x.size(2), x.size(3), 256) ## (B, H', W', 256)
        
        x3 = x3.permute(0, 3, 1, 2).contiguous() ## (B, 256, H', W')
        x3 = self.lstm_fc(x3) ## (B, 512, H', W')
        x = x3
        
        cls = self.rpn_class(x) ## (B, 20, H', W')
        regression = self.rpn_regress(x) ## (B, 20, H', W')
        cls = cls.permute(0, 2, 3, 1).contiguous() ## (B, H', W', 20)
        regression = regression.permute(0, 2, 3, 1).contiguous() ## (B, H', W', 20)
        
        cls = cls.contiguous().view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        regression = regression.contiguous().view(regression.size(0), regression.size(1) * regression.size(2) * 10, 2)
        """
        - score: text/nontext score
        - vertical_pred: vertical coordinates
        - side_refinement: side-refinement offset
        """
        return cls, regression
    
    
if __name__ == "__main__":
    device = torch.device('cuda')
    model = CTPN().to(device)
    WEIGHT = '/home/ubuntu/user/jihye.lee/ocr_exp_v1/text_detection/weight/ctpn.pth'
    model_weight = model.state_dict()
    pretrained_weight = torch.load(WEIGHT)['model_state_dict']
    available = {key:value for (key, value) in pretrained_weight.items() if key in model_weight and \
                    value.shape == model_weight[key].shape}
    model_weight.update(available)
    model.load_state_dict(model_weight)
    #model.load_state_dict(torch.load(WEIGHT)['model_state_dict'])
    
    sample = torch.rand((2, 3, 512, 512)).to(device)
    cls, regression = model(sample)
    print(cls.shape, regression.shape)
