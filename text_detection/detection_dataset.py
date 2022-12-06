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
from torchvision import transforms as transforms
from utils.ctpn_utils import ctpn_data_utils as ctpn_utils ## 아마도 CTPN 모덿은 사용하지 않을 확률이 높을 것
from utils.east_utils import east_utils as east_utils
from utils.east_utils import east_detect as east_detect
from utils.east_utils import geo_map_utils as geo_map_utils
from utils.ctpn_utils import anchor_data
from torch.utils.data import Dataset, DataLoader
def to_xy_min_max(bboxes):
    """
    Convert one bounding box whose form is: [x1, y1, x2, y2, x3, y3, x4, y4]
    into a box of form (xmin, ymin, xmax, ymax)
    Args:
        bboxes (numpy.ndarray): A numpy array containing the bounding box 8-coordinates.
    Returns:
        A list containing the bounding box 4-coordinates.
    """

    if len(bboxes) != 8:
        raise NotImplementedError("The bounding box coordinates must a length of 8!")

    Xs = bboxes[0::2]
    Ys = bboxes[1::2]

    xmin = int(round(np.min(Xs, 0)))
    ymin = int(round(np.min(Ys, 0)))
    xmax = int(round(np.max(Xs, 0)))
    ymax = int(round(np.max(Ys, 0)))

    final_boxes = [xmin, ymin, xmax, ymax]

    return final_boxes


def order_point_clockwise(bboxes: np.ndarray) -> np.ndarray:
    """
    Order in clockwise the bounding box coordinates.
    Args:
        bboxes (numpy.ndarray): A numpy array containing the bounding box coordinates. Shape: [4, 2].
    Returns:
        An ordered clockwise bounding box.
    """
    if bboxes.ndim != 2 and bboxes.shape != (4, 2):
        raise ValueError("The bounding box coordinates are not in the correct shape!"
                         "It must be an numpy array of 2D whose shape is (4, 2).")

    # sort the points based on their x-coordinates
    xSorted = bboxes[np.argsort(bboxes[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates, so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (top_left, bottom_left) = leftMost

    # now, sort the right-most coordinates according to their
    # y-coordinates, so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (top_right, bottom_right) = rightMost

    # return the coordinates in this following order: top-left, top-right, bottom-right, and bottom-left
    return np.array([top_left, top_right, bottom_right, bottom_left])
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
            return NewCTPNDataset(self.data_cfg, self.mode)


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
            if self.mode.upper() == 'TRAIN':
                folder_no = random.randint(0, 12)
                file_no = random.randint(0, len(self.img_files[folder_no])-1)
            else:
                folder_no = 0
                file_no = idx
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

class NewCTPNDataset(BASEDataset):
    def __init__(self, data_cfg, mode = 'train'):
        BASEDataset.__init__(self, data_cfg, mode)
    def __len__(self):
        if self.mode.upper() == 'TRAIN':
            return 3000
        else:return 100
    
    
    def parse_gtbox(self, txt_file):
        txt_data = txt_file.split('\n')
        bboxes = []
        all_box = []
        for line in txt_data:
            if line == '':
                continue
            else:
                new_line = [float(i) for i in line.split(' ')[:8]]
                X = new_line[::2]
                Y = new_line[1::2]
                
                bbox = order_point_clockwise(np.array(list(map(np.float32, new_line[:8]))).reshape((4, 2)))
                #bbox = np.array([min(X), min(Y), max(X), max(Y)])
                if cv2.arcLength(bbox, True) > 0:
                    bbox = np.array(to_xy_min_max(bbox.flatten()))
                    bboxes.append(bbox)
                    all_box.append([min(X), min(Y), max(X), max(Y)])
        return np.array(bboxes, dtype = np.float32), np.array(all_box, dtype =np.float32)
    
    def split_bbox(self, image, gt_bboxes):
        # Now we split bounding box coordinates according to the anchor shift value.
        new_gt_bboxes = []

        for i, bbox in enumerate(gt_bboxes):
            xmin, ymin, xmax, ymax = bbox

            bbox_ids = np.arange(int(np.floor(1.0 * xmin / 16)),
                                 int(np.ceil(1.0 * xmax / 16)))

            new_bboxes = np.zeros(shape=(len(bbox_ids), 4))

            new_bboxes[:, 0] = bbox_ids * 16

            new_bboxes[:, 1] = ymin

            new_bboxes[:, 2] = (bbox_ids + 1.0) * 16

            new_bboxes[:, 3] = ymax

            new_gt_bboxes.append(new_bboxes)

        # Bounding boxes must be within the image size.
        new_gt_bboxes = np.concatenate(new_gt_bboxes, axis=0)

        return image, new_gt_bboxes
    
    def __getitem__(self,idx ):
        sucess = False
        while sucess == False:
            if self.mode.upper() == 'TRAIN':
                folder_no = random.randint(0, len(self.img_files)-1)
                file_no = random.randint(0, len(self.img_files[folder_no])-1)
            else:
                folder_no = 0
                file_no = idx
            
            try:
                img_data = self.img_archive[folder_no].read(self.img_files[folder_no][file_no])
                img_io = io.BytesIO(img_data)
                img = Image.open(img_io)
                #img = np.array(img)
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                sucess = True
            except:
                continue
        #H, W, C = img.shape
        W, H = img.size
        text_file_name = self.img_files[folder_no][file_no].split('/')[-1].replace('jpg', 'txt')
        label_data = self.label_archive.read(text_file_name).decode('utf-8') ## read the text label data
        gt_bbox, all_box = self.parse_gtbox(label_data)
        # reshape_w, reshape_h = ctpn_utils.IMAGE_SIZE[1], ctpn_utils.IMAGE_SIZE[0]
        if H > W:
            reshape_h = 2048
            reshape_w = int((2048 / H)* W)
            reshape_w = (reshape_w // 16) * 16
            
        else:
            reshape_w = 2048
            reshape_h = int((2048 / W) * H)
            reshape_h = (reshape_h // 16) * 16
            
        rescale_w = reshape_w/W
        rescale_h = reshape_h/H
        scale = [[rescale_w, rescale_h, rescale_w, rescale_h]]
        all_box *= scale
        gt_bbox *= scale
        img = transforms.Compose([
            transforms.Resize(size = (reshape_h, reshape_w))
        ])(img)
        # logger.info(f"{reshape_w}, {reshape_h}")
        img, gt_bbox = self.split_bbox(img, gt_bbox)
        copy_img = np.array(img.copy())
        for box in all_box:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(copy_img, (x1, y1), (x2, y2),(255,0,0), 2)
        #cv2.imwrite('/home/ubuntu/user/jihye.lee/ocr_exp_v1/test.png', copy_img)
        #logger.info("WROTE")
        img = np.array(img) - ctpn_utils.IMAGE_MEAN
        #img = Image.fromarray(img)
        #data_transform = transforms.Compose([
         #   transforms.ToTensor(),
            # transforms.Normalize(mean = ctpn_utils.NEW_IMAGE_MEAN, std = ctpn_utils.NEW_IMAGE_STD)
        #])
        # image  = data_transform(img)
        img = img.transpose([2, 0, 1])
        image = torch.from_numpy(img).float()
        all_box = transforms.ToTensor()(all_box)
        gt_bbox = transforms.ToTensor()(gt_bbox)
        # target_trasnsform에서 고정된 width와 정해준 height를 갖는 anchor들을 생성해 준다.
        # 결과적으로 모델의 GRU layer의 output의 dimension의 개수가 anchor의 개수이고 regression output의 0axis값은
        # 중심좌표의 y축 위치, 그리고 1 axis값은 anchor의 높이를 의미하게 될 것이다.
        target_transform = anchor_data.TargetTransform()(
            gt_boxes = gt_bbox.clone(),
            image_size = image.shape[1:3],
            return_anchor_boxes=False
        )
        return image,target_transform[:2], all_box
########################################################################################################
## DETECTION DATASET FOR THE CTPN MODEL ##
class CTPNDataset(BASEDataset):
    def __init__(self, data_cfg, mode = 'train'):
        BASEDataset.__init__(self, data_cfg, mode)
        
    def __len__(self):
        if self.mode.upper() == 'TRAIN':
            return 3000#  int(len(self.img_files) * 0.8)
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
            
            if rescale_factor > 1.0:
                xmin = int(xmin / rescale_factor)
                xmax = int(xmax / rescale_factor)
                ymin = int(ymin / rescale_factor)
                ymax = int(ymax / rescale_factor)
            full_boxes.append((xmin, ymin, xmax, ymax))
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
            if self.mode.upper() == 'TRAIN':
                folder_no = random.randint(0, len(self.img_files)-1)
                file_no = random.randint(0, len(self.img_files[folder_no])-1)
            else:
                folder_no = 0
                file_no = idx
            
            try:
                img_data = self.img_archive[folder_no].read(self.img_files[folder_no][file_no])
                img_io = io.BytesIO(img_data)
                img = Image.open(img_io)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                sucess = True
            except:
                continue
        # img = np.expand_dims(img, axis = -1)
        # img = np.concatenate((img, img, img),axis = -1).astype(np.uint8)
       #  img = Image.fromarray(img) ## read the image in a PIL Image form
        #numpy_img = np.array(img)
        #Color(numpy_img, cv2.COLOR_RGB2BGR)
        # W,H = img.size
        H, W, C = img.shape
        text_file_name = self.img_files[folder_no][file_no].split('/')[-1].replace('jpg', 'txt')
        label_data = self.label_archive.read(text_file_name).decode('utf-8') ## read the text label data
        
        # img, label_data = self.random_crop(img, label_data) ## x axis에 대해서 crop을 해서 scale도 그렇고 예측해야 하는 text line의 개수를 receipt에 있는 개수와 비슷하게 맞춰주려 했다.

        # H, W, C = img.shape
        # logger.info(f"{(H, W)}")
        # rescale_factor = max(H, W) / 1000
        
        rescale_factor = max(H, W) / 1000
        """image rescaling을 하는데 rescale factor을 ground truth generation에서
        보내주지 않았기 때문에 문제가 생겼었다.
        """
        if rescale_factor > 1.0:
            H = int(H / rescale_factor)
            W = int(W / rescale_factor)
            # img = img.resize((W,H), Image.BILINEAR)
            img = cv2.resize(img, (W, H))
            # img = cv2.resize(img, (W, H))
        # text_file_name = self.img_files[folder_no][file_no].replace('image', 'box').replace('jpg', 'txt')
        


        
        vertices, full_boxes = self.get_gtbox(label_data, rescale_factor)
        gt_bboxes = self.parse_gtbox(label_data)
        
        [cls, regr] = ctpn_utils.cal_rpn((H, W), (int(H / 16), int(W / 16)), 16, vertices)
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr]) ## Bounding Box targets
        cls = np.expand_dims(cls, axis = 0) ## Labels
        
        mean_img = img - ctpn_utils.IMAGE_MEAN ## 이미지의 정규화를 할 때에 
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
        img_path= ["image_data-20221112T142753Z-001.zip"],
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
    
        
        
        
    