## score map should be made 
import os, sys
import io
import zipfile
import torch
import numpy as np
import cv2
from PIL import Image
from loguru import logger
import random
import torchvision
from utils.ctpn_utils import ctpn_data_utils as ctpn_utils ## 아마도 CTPN 모덿은 사용하지 않을 확률이 높을 것
from utils.east_utils import east_utils as east_utils
from utils.east_utils import east_detect as east_detect
from utils.east_utils import geo_map_utils as geo_map_utils

from torch.utils.data import Dataset, DataLoader

def json_to_txt(json_path):
    pass

class DATASET:
    def __init__(self, model_name, data_cfg, mode):
        super(DATASET, self).__init__()
        self.model_name = model_name
        self.data_cfg = data_cfg
        self.mode = mode
    def get(self):
        if self.model_name.upper() == 'EAST':
            return EASTDataset(self.data_cfg, self.mode)
        elif self.model_name.upper() == 'CTPN':
            return CTPNDataset(self.data_cfg, self.mode)


class BASEDataset(Dataset):
    def __init__(self, data_cfg, mode = 'train'):
        self.mode = mode
        self.data_cfg = data_cfg
        self.mean, self.std = data_cfg['mean'], data_cfg['std']
        ## 데이터의 형태가 다 동일할수는 없지만 기본적으로 원하는 형태로 어떤 데이터던 바꿔줄 수 있어야 한다.
        # self.img_files = [os.path.join(self.data_cfg['img_path'], img_file) for img_file in sorted(os.listdir(self.data_cfg['img_path']))]
        # self.label_files = [os.path.join(self.data_cfg['label_path'], label_file) for label_file in sorted(os.listdir(self.data_cfg['label_path']))]
        img_zip_path = data_cfg['img_path'] ## list type
        img_zip_path = list(map(lambda x: os.path.join('/home/ubuntu/user/jihye.lee/data/detection_aihub', x), img_zip_path))
        self.img_zip_path = {
            key: value for (key, value) in zip([int(i) for i in range(len(img_zip_path))], img_zip_path)
        }
        self.label_zip_path = data_cfg['label_path']
        self.img_archive = {}
        for key, value in self.img_zip_path.items():
            self.img_archive[key] = zipfile.ZipFile(value, 'r')
        # self.img_archive = zipfile.ZipFile(self.img_zip_path, 'r')
        self.img_files = {}
        for key, value in self.img_archive.items():
            self.img_files[key] = value.namelist()
        # self.img_files = sorted(self.img_archive.namelist())
        self.label_archive = zipfile.ZipFile(self.label_zip_path, 'r')
        self.label_files = sorted(self.label_archive.namelist())
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        pass
    
## DETECTION DATASET FOR THE EAST MODEL ##
class EASTDataset(BASEDataset):
    def __init__(self, data_cfg, mode):
        BASEDataset.__init__(self, data_cfg, mode)
        
        
    def __len__(self):
        if self.mode.upper() == 'TRAIN':
            return 1000
        elif self.mode.upper() == 'TEST':
            return 10
        else:
            return 100
    
    def __getitem__(self, idx):
        # label_data = self.label_archive.read(self.label_files[idx]).decode('utf-8')
        
        # img = Image.open(self.img_files[idx])
        sucess = False
        while sucess == False:
            folder_no = random.randint(0, 12)
            file_no = random.randint(0, len(self.img_files[folder_no])-1)
            try:
                img_data = self.img_archive[folder_no].read(self.img_files[folder_no][file_no])
                img_io = io.BytesIO(img_data)
                img = Image.open(img_io)
                sucess = True
            except:
                continue
        # img = np.expand_dims(img, axis = -1)
        # img = np.concatenate((img, img, img),axis = -1).astype(np.uint8)
        # img = Image.fromarray(img)
        
        text_file_name = self.img_files[folder_no][file_no].replace('image', 'box').replace('jpg', 'txt')
        label_data = self.label_archive.read(text_file_name).decode('utf-8')
        
        vertices, labels = geo_map_utils.extract_vertices_from_txt(label_data) ## json 형태로 저장되어있는 label을 모두 txt 파일로 바꾸어야 한다.
        
        if self.mode.upper() == 'TRAIN':
            img, vertices = geo_map_utils.adjust_height(img, vertices)
            ## 생각해 보니 영수증을 올바르게 촬영을 한다는 가정에서 볼 때에 굳이 image rotate를 할 필요는 없을 것 같다.
            # img, vertices = geo_map_utils.rotate_img(img, vertices)
            img, vertices = geo_map_utils.crop_img(img, vertices, labels, length = self.data_cfg['crop_length'])
            # cv2.imwrite('/home/ubuntu/user/jihye.lee/ocr_exp_v1/text_detection/results/sample.png', np.array(img))
            transform = torchvision.transforms.Compose(
                [
                    # torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean = [self.mean for _ in range(3)], std = [self.std for _ in range(3)])
                ]
            )
            length = self.data_cfg['crop_length']
        else:
            def make_divisional(image):
                H, W = image.size
                ## H,W = image.size -> PIL Image.size를 사용하면 높이, 너비만 알려줌
                adjust_h = H if H % 32 == 0 else (H // 32) * 32
                adjust_w = W if W % 32 == 0 else (W // 32) * 32
                # new_image = image.resize((adjust_h, adjust_w), Image.BILINEAR)
                ratio_h = adjust_h / H ## < 1.0
                ratio_w = adjust_w / W

                return image, ratio_h, ratio_w, adjust_h, adjust_w
            # img, vertices = geo_map_utils.adjust_height(img, vertices)
            # length = min(img.width, img.height) ## 이미지의 크기가 32로 나누어 떨어져야 한다.
            # length = 1024
            # img, ratio_h, ratio_w = east_detect.resize_image(img)
            # img, vertices = geo_map_utils.crop_img(img, vertices, labels, length)
            image, ratio_h, ratio_w, newH, newW = make_divisional(img)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size = (newH, newW), interpolation = torchvision.transforms.InterpolationMode.BILINEAR),
                torchvision.transforms.Normalize(mean = [self.mean for _ in range(3)], std = [self.std for _ in range(3)])
            ])
            return transform(image), torch.Tensor(vertices), {"ratio_h": ratio_h, "ratio_w": ratio_w}

        
        score_map, geo_map, ignored_map = geo_map_utils.make_geo_score(
                                        img, vertices, labels, \
                                    self.data_cfg['scale'], length) # self.data_cfg['crop_length'])
        
        return transform(img), score_map, geo_map, ignored_map
        
## DETECTION DATASET FOR THE CTPN MODEL ##
class CTPNDataset(BASEDataset):
    def __init__(self, data_cfg, mode = 'train'):
        BASEDataset.__init__(self, data_cfg, mode)
        
    def __len__(self):
        if self.mode.upper() == 'TRAIN':
            return 1000#  int(len(self.img_files) * 0.8)
        elif self.mode.upper() == 'TEST':
            return 10
        else:
            return 100

    def get_gtbox(self, txt_file, rescale_factor = 1.0):
        #with open(txt_file, 'r') as f:
            #txt_data = f.readlines()
        txt_data = txt_file.split('\n')
        gt_boxes = []
        full_boxes= []
        for data in txt_data:
            if data == '':
                continue
            # print(data)
            x1, y1, x2, y2, x3, y3, x4, y4 = data.split(' ')[:8]
            x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
            xmin, xmax = min(x1, x2, x3, x4), max(x1, x2,x3, x4)
            ymin, ymax = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
            full_boxes.append((xmin, ymin, xmax, ymax))
            if rescale_factor > 1.0:
                xmin = int(xmin / rescale_factor)
                xmax = int(xmax / rescale_factor)
                ymin = int(ymin / rescale_factor)
                ymax = int(ymax / rescale_factor)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16 * i -0.5
                gt_boxes.append((prev, ymin, next, ymax))
                prev = next
            gt_boxes.append((prev, ymin, xmax, ymax))
            
        return np.array(gt_boxes), np.array(full_boxes)
            
    def __getitem__(self, idx):
        sucess = False
        while sucess == False:
            folder_no = random.randint(0, 12)
            file_no = random.randint(0, len(self.img_files[folder_no])-1)
            try:
                img_data = self.img_archive[folder_no].read(self.img_files[folder_no][file_no])
                img_io = io.BytesIO(img_data)
                img = Image.open(img_io)
                sucess = True
            except:
                continue
        # img = np.expand_dims(img, axis = -1)
        # img = np.concatenate((img, img, img),axis = -1).astype(np.uint8)
       #  img = Image.fromarray(img) ## read the image in a PIL Image form
        
        H, W = img.size
        #logger.info(f"{(H, W)}")
        rescale_factor = min(H, W) / 1000
        if rescale_factor > 1.0:
            H = int(H / rescale_factor)
            W = int(W / rescale_factor)
            img = img.resize((H, W), Image.BILINEAR)
            
        text_file_name = self.img_files[folder_no][file_no].replace('image', 'box').replace('jpg', 'txt')
        label_data = self.label_archive.read(text_file_name).decode('utf-8') ## read the text label data
        
        vertices, full_boxes = self.get_gtbox(label_data)
        
        img = np.array(img)
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            new1 = W - vertices[:, 2] - 1
            new2 = W - vertices[:, 0] - 1
            vertices[:, 0] = new1
            vertices[:, 1] = new2
        
        [cls, regr] = ctpn_utils.cal_rpn((H, W), (int(H / 16), int(W / 16)), 16, vertices)
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr]) ## Bounding Box targets
        cls = np.expand_dims(cls, axis = 0) ## Labels
        
        mean_img = img - ctpn_utils.IMAGE_MEAN
        # mean_img = torchvision.transforms.ToTensor()(mean_img)
        mean_img = mean_img.transpose([2, 0, 1])
        mean_img = torch.from_numpy(mean_img).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()
        
        if self.mode.upper() == 'TRAIN':
            return mean_img, cls, regr
        else:
            return mean_img, cls, regr, torch.from_numpy(full_boxes).float()
        
        
if __name__ == "__main__":
    DATA_CFG = dict(
        label_path="/home/ubuntu/user/jihye.lee/data/detection_aihub/box_data.zip",
        img_path= "/home/ubuntu/user/jihye.lee/data/detection_aihub/image_data-20221112T142753Z-001.zip",
        batch_size=4,
        scale= 0.25,
        crop_length= 800,
        mean= 0.5,
        std=0.5
    )
    dataset =DATASET(model_name = 'ctpn', data_cfg = DATA_CFG, mode = 'train')
    ctpn_dataset = dataset.get()
    dataloader = torch.utils.data.DataLoader(ctpn_dataset, batch_size = 3, shuffle = False)
    for idx, batch in enumerate(dataloader):
        img, cls, rgr = batch
        logger.info(f"IMAGE: {img.shape} CLS: {cls.shape} RGR: {rgr.shape}")
        print(np.unique(cls.numpy()))
        break
    
        
        
        
    