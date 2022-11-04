import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, padding = ((kernel_size - 1)//2))
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_1, ch_2, ch_3):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            ConvLayer(ch_in, ch_1, kernel_size = 1),
            ConvLayer(ch_1, ch_2, kernel_size = 3),
            ConvLayer(ch_2, ch_3, kernel_size = 1)
        ) 
    def forward(self, x):
        return self.layer(x)
    
class InceptionBlock(nn.Module):
    def __init__(self, ch_in, out_1_1, out_2_1, out_2_2, out_3_1, out_3_2, out_3_3, ch_out):
        super(InceptionBlock, self).__init__()
        self.conv1 = ConvLayer(ch_in, out_1_1, kernel_size = 1)
        self.conv2 = nn.Sequential(
            ConvLayer(ch_in, out_2_1, 3),
            ConvLayer(out_2_1, out_2_2, 1)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(ch_in, out_3_1, 3),
            ConvLayer(out_3_1, out_3_2, 3),
            ConvLayer(out_3_2, out_3_3, 1)
        )
        self.tail = ConvLayer(out_1_1 + out_2_2 + out_3_3, ch_out, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1, x2, x3), 1)
        return self.tail(x)
    
class PVANet(nn.Module):
    def __init__(self, 
                 ch_in = 3,
                 channels = [16, 64, 128, 256, 384],
                 
                 ):
        super(PVANet, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(ch_in, channels[0], kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(channels[0]), nn.ReLU(),
        )
        self.layer1 = ConvBlock(channels[0], 24, 24, channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer2 = ConvBlock(channels[1], 48, 48, channels[2])
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer3 = InceptionBlock(channels[2], 64, 48, 128, 24, 48, 48, channels[3])
        self.pool4 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer4 = InceptionBlock(channels[3], 64, 96, 192, 32, 64, 64, channels[4])
        self.pool5 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    
    def forward(self, x):
        out = []
        x = self.head(x)
        x = self.pool2(self.layer1(x))
        out.append(x)
        x = self.pool3(self.layer2(x))
        out.append(x)
        x = self.pool4(self.layer3(x))
        out.append(x)
        x = self.pool5(self.layer4(x))
        out.append(x)
        return out    
    
    
class PVA(nn.Module):
    def __init__(self, ch_in = 3, channels = [16, 64, 128, 256, 384],
                 class_n = 1000):
        super(PVA, self).__init__()
        self.features = PVANet(ch_in, channels)
        self.avgpool = nn.AdaptivaAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, class_n)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    



if __name__ == "__main__":
    import os, sys
    from torchvision.datasets import ImageNet
    from torch.utils.data import DataLoader
    from torchmetrics import Accuracy
    from loguru import logger
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    
    train_dataset = ImageNet(root = '', split = 'train')
    test_dataset = ImageNet(root = '', split = 'val')
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    
    EPOCH = 1000
    EVAL_EPOCH = 50
    BEST_ACC = 0.0
    WEIGHT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weight')
    model = PVA().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    criterion = nn.CrossEntropyLoss(reduction = 'mean')
    metric = Accuracy()
    
    for epoch in range(EPOCH):
        model.train()
        train_loop = tqdm(train_dataloader)
        for idx, batch in enumerate(train_loop):
            image, label = batch
            image, label = image.cuda(), label.cuda()
            predict = model(image)
            loss = criterion(input = predict, target = label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loop.set_postfix({"LOSS" : loss.item()})
            
        if (epoch +1) % EVAL_EPOCH == 0:
            model.eval()
            run_acc = 0.0
            with torch.no_grad():
                eval_loop = tqdm(test_dataloader)
                
                for idx, batch in enumerate(eval_loop):
                    image, label = batch
                    image, label = image.cuda(), label.cuda()
                    predict = model(image)
                    acc = metric(predict, label)
                    run_acc += acc
                    eval_loop.set_postfix({"ACCURACY" : acc})
                    
            if run_acc > BEST_ACC:
                BEST_ACC = run_acc
                torch.save(model.state_dict(), os.path.join(WEIGHT_PATH, 'best_pva.pth'))
                logger.info("SAVED BEST MODEL..")
    
    torch.save(model.state_dict(), os.path.join(WEIGHT_PATH, 'last_pva.pth'))
            
                
                
            
        
        
    
    