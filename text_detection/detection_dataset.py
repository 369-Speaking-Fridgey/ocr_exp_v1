## score map should be made 
import os, sys
from PIL import Image
import torchvision
from utils.ctpn_utils import ctpn_data_utils as ctpn_utils ## 아마도 CTPN 모덿은 사용하지 않을 확률이 높을 것
from utils.east_utils import east_utils as east_utils
from torch.utils.data import Dataset, DataLoader

def json_to_txt(json_path):
    pass
    
## DETECTION DATASET ##
class Dataset(Dataset):
    def __init__(self, data_cfg, mode = 'train'):
        super(Dataset, self).__init__()
        self.mode = mode
        self.data_cfg = data_cfg
        self.mean, self.std = data_cfg['mean'], data_cfg['std']
        ## 데이터의 형태가 다 동일할수는 없지만 기본적으로 원하는 형태로 어떤 데이터던 바꿔줄 수 있어야 한다.
        self.img_files = [os.path.join(self.data_cfg['img_path'], img_file) for img_file in sorted(os.listdir(self.data_cfg['img_path']))]
        self.label_files = [os.path.join(self.data_cfg['label_path'], label_file) for label_file in sorted(os.listdir(self.data_cfg['label_path']))]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        vertices, labels = east_utils.extract_vertices(self.label_files[idx]) ## json 형태로 저장되어있는 label을 모두 txt 파일로 바꾸어야 한다.
        img = Image.open(self.img_files[idx])
        
        if self.mode.upper() == 'TRAIN':
            img, vertices = east_utils.adjust_height(img, vertices)
            img, vertices = east_utils.rotate_img(img, vertices)
            img, vertices = east_utils.crop_img(img, vertices, labels, length = self.data_cfg['crop_length'])
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean = [self.mean for _ in range(3)], std = [self.std for _ in range(3)])
                ]
            )
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean = [self.mean for _ in range(3)], std = [self.std for _ in range(3)])
            ])
            
        score_map, geo_map, ignored_map = east_utils.make_geo_score(
                                        img, vertices, labels, \
                                    self.data_cfg['scale'], self.data_cfg['crop_length'])
        
        return transform(img), score_map, geo_map, ignored_map
        
            