## score map should be made 
import os, sys
import io
import zipfile

from PIL import Image
import torchvision
from utils.ctpn_utils import ctpn_data_utils as ctpn_utils ## 아마도 CTPN 모덿은 사용하지 않을 확률이 높을 것
from utils.east_utils import east_utils as east_utils
from torch.utils.data import Dataset, DataLoader

def json_to_txt(json_path):
    pass

class DATATSET:
    def __init__(self, model_name, data_cfg, mode):
        super(DATASET, self).__init__()
        if model_name.upper() == 'EAST':
            return EASTDataset(data_cfg, mode)
        elif model_name.upper() == 'CTPN':
            return CTPNDataset(data_cfg, mode)


class BASEDataset(Dataset):
    def __init__(self, data_cfg, mode = 'train'):
        super(BASEDataset, self).__init___()
        self.mode = mode
        self.data_cfg = data_cfg
        self.mean, self.std = data_cfg['mean'], data_cfg['std']
        ## 데이터의 형태가 다 동일할수는 없지만 기본적으로 원하는 형태로 어떤 데이터던 바꿔줄 수 있어야 한다.
        # self.img_files = [os.path.join(self.data_cfg['img_path'], img_file) for img_file in sorted(os.listdir(self.data_cfg['img_path']))]
        # self.label_files = [os.path.join(self.data_cfg['label_path'], label_file) for label_file in sorted(os.listdir(self.data_cfg['label_path']))]
        self.img_zip_path = data_cfg['img_path']
        self.label_zip_path = data_cfg['label_path']
        self.img_archive = zipfile.ZipFile(self.img_zip_path, 'r')
        self.img_files = sorted(self.img_archive.namelist())
        self.label_archive = zipfile.ZipFile(self.label_zip_path, 'r')
        self.label_files = sorted(self.label_archive.namelist())
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        pass
    
## DETECTION DATASET FOR THE EAST MODEL ##
class EASTDataset(BASEDataset):
    def __init__(self, data_cfg, mode = 'train'):
        super(EASTDataset, self).__init__()
        
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        label_data = self.label_archive.read(self.label_files[idx]).decode('utf-8')
        vertices, labels = east_utils.extract_vertices(label_data) ## json 형태로 저장되어있는 label을 모두 txt 파일로 바꾸어야 한다.
        # img = Image.open(self.img_files[idx])
        img_data = self.img_archive.read(self.img_files[idx])
        img_io - io.BytesIO(img_data)
        img = Image.open(img_io)
        
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
        
## DETECTION DATASET FOR THE CTPN MODEL ##
class CTPNDataset(BASEDataset):
    def __init__(self, data_cfg, mode = 'train'):
        super(CTPNDataset, self).__init__()
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        pass
    