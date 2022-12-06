import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from loguru import logger
import os, sys
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)
sys.path.append(os.path.dirname(BASE))
from text_detection.utils.ctpn_utils import ctpn_detect, ctpn_data_utils
from text_detection.detect_model.ctpn.ctpn import CTPN
from torchvision import transforms as transforms
from ocr.preprocess import preprocess

class DetectCFG:
    PRETRAINED_WEIGHT='/home/ubuntu/user/jihye.lee/ocr_exp_v1/text_detection/results/2022-12-01 15:32:38/best.pt'
    # PRETRAINED_WEIGHT='/home/ubuntu/user/jihye.lee/ocr_exp_v1/text_detection/weight/ctpn.pth'
    LINE_MIN_SCORE=0.7
    TEXT_PROPOSALS_MIN_SCORE=0.9
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MAX_HORIZONTAL_GAP=70
    MIN_V_OVERLAPS=0.7
    MIN_SIZE_SIM=0.7
    ANCHOR_SHIFT=16
    IMAGE_SIZE= [2048, 1536] #[1024, 768]
    # IMAGE_SIZE=[2080, 1024] ## 세로의 길이를 더 길게 설정을 해 주어야 한다.
    IMAGE_MEAN=[123.68, 116.779, 103.939]
    #IMAGE_STD=[0.20037157, 0.18366718, 0.19631825]
    #IMAGE_MEAN = [0.90890862, 0.91631571, 0.90724233]


class TextDetector(object):
    def __init__(self):
        super(TextDetector, self).__init__()
        self.model = CTPN().cuda()
        pretrained_weight = torch.load(DetectCFG.PRETRAINED_WEIGHT)
        self.new_H = DetectCFG.IMAGE_SIZE[0]
        self.new_W = DetectCFG.IMAGE_SIZE[1]
        if 'model_state_dict' in pretrained_weight:
            pretrained_weight = pretrained_weight['model_state_dict']
        self.model.load_state_dict(pretrained_weight)
        self.text_proposal_connector = ctpn_detect.TextDetector()
        
        # self.text_proposal_connector = ctpn_data_utils.TextProposalConnectorOriented()
    
    def run(self, image):
        """ Args
        image: PIL Image (근데 원래 알고 있는대로 이미지를 cv2 형태로 읽으려면 np.array로 바꾸어주는것이 맞다.)
        Outputs
        new_text: list형태인데 모든 감지된 bounding box의 (minx, miny, maxX, maxY)의 값을 갖는다.ㄴ
        """
        #image = preprocess(image)
        copy_image = image.copy()
        H, W, C = np.array(image).shape ## 원본 이미지
        max_len = max(H, W);min_len = min(H, W)
        if H < W and H < 1000:
            NEW_MAX = 1024
        else:
            NEW_MAX = 2048
        
        """ WHAT IS IMPORTANT
        - 중요한건 입력 이미지의 가로와 세로의 비율을 계속 유지해 주어야 한다는 것이다.
        - 게다가 CTPN 모델의 경우 anchor의 가로의 길이가 16이지만 세로의 길이는 11 ~ 283까지 다양하기 때문에
        이미지를 어느 정도는 키워 주어야 한다. 
        - 글씨 detect이기 때문에 감지 영역이 작고,이미지의 원본의 scale이 너무 작다면 anchor안에 인식하고 싶은 개별적인 부분이 아닌
        문단 단위의 느낌으로 감지가 되기 때문이다.
        """
        if H > W:
            self.new_H = NEW_MAX
            self.new_W = int((NEW_MAX / H) * W)
            self.new_W = (self.new_W // 16) * 16
            
        else:
            self.new_W = NEW_MAX
            self.new_H = int((NEW_MAX / W) * H)
            self.new_H = (self.new_H // 16) * 16 
        
        copy_image = image.copy()
        logger.info(f"{H}, {W}")

        """
        rescale_factor = 2048 / max(H, W)
        if rescale_factor > 1.0:
            self.new_H = int(H * rescale_factor)
            self.new_W = int(W * rescale_factor)
            #image = cv2.resize(image, (new_w, new_h))
        """
        
        image = cv2.resize(image, (self.new_W, self.new_H))
        image = preprocess(image)
    
        rescale_w = W/self.new_W
        rescale_h = H/self.new_H
        ## (0) 나중에 text box를 원본 이미지의 ratio에 맞춰 주어야 하니까 비율을 미리 계산해 둔다.
        # rescale_w = new_w / DetectCFG.IMAGE_SIZE[1] 
        # rescale_h = H/ DetectCFG.IMAGE_SIZE[0] 
        scale = np.array([[rescale_w, rescale_h, rescale_w, rescale_h]])
        # scale = np.array([[rescale_factor, rescale_factor, rescale_factor, rescale_factor]])
        ## (1) Get the output of the model
        cls, regr = self.get_prediction(image)
        ## (2) Get the Detection Box based on the output
        selected_anchor, selected_score = self.get_detection(cls, regr, self.new_W,self.new_H)
        ## (3) Get the Text Lines
        # text, new_text = self.text_proposal_connector.get_text_lines(selected_anchor, selected_score, (self.new_H, self.new_W))
        text, scores = self.text_proposal_connector((regr.detach(), cls.detach()), (self.new_H, self.new_W))
        ## (4) Rescale the Text Lines
        #x_diff = (max_len-W)//2
        #y_diff = (max_len-H)//2
        new_text = []
        for t in text:
            """
            t = [int(i) for i in t][:8]
            x1, y1, x2, y2, x3, y3, x4, y4 = t
            X = [x1,x2,x3,x4];Y = [y1,y2,y3,y4]
            """
            t = [int(i) for i in t][:4]
            x1,y1,x2,y2 =  t
            X = [x1,x2];Y = [y1,y2]
            new_text.append([min(X), min(Y), max(X), max(Y)])
        new_text *= scale
        ## (5) Draw the Bounding Box
        copy_img, box_dict = self.draw(copy_image, new_text)
        return copy_img, box_dict


    def draw(self, image, new_text):
        copy_img = image.copy()
        box_dict = {}
        box_dict['box'] = new_text
        for idx, box in enumerate(new_text):
            logger.info(box)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(copy_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
        cv2.imwrite('./result.png',copy_img)

        return copy_img, box_dict


    def get_detection(self, cls, regr, new_w, new_h):
        """ Args
        Returns: A tuple containing the predicted bounding boxes and score
        """
       #  H, W = DetectCFG.IMAGE_SIZE[0], DetectCFG.IMAGE_SIZE[1]
        H = new_h
        W = new_w
        ## (1) Change all to numpy array
        pred_cls = F.softmax(cls, dim = -1).detach().cpu().numpy()
        pred_regr = regr.detach().cpu().numpy()
        anchor_shift = DetectCFG.ANCHOR_SHIFT
        #logger.info(f"CLS : {pred_cls.shape} REGR : {pred_regr.shape}")
        feature_map_size = (int(H / anchor_shift), int(W / anchor_shift))
        ## (2) Generate all the anchor boxes
        anchor = ctpn_data_utils.gen_anchor(feature_map_size, anchor_shift)
        bbox = ctpn_data_utils.transform_bbox(anchor, pred_regr)
        bbox = ctpn_data_utils.clip_bbox(bbox, [H, W])
        
        fg = np.where(pred_cls[0,:,1] > DetectCFG.TEXT_PROPOSALS_MIN_SCORE)[0]
        
        
        select_anchor = bbox[fg, :]
        select_score = pred_cls[0, fg, 1].astype(np.uint32)
        keep_index = ctpn_data_utils.filter_bbox(select_anchor, 16)
        
        ## (3) Take only the bbox and score based on the text proposal minimum score
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nms_box = np.hstack((select_anchor, select_score))
        keep = ctpn_data_utils.nms(nms_box, DetectCFG.TEXT_PROPOSALS_NMS_THRESH)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        return select_anchor, select_score


    def get_prediction(self, image):
        # image = np.resize(image,(DetectCFG.IMAGE_SIZE[0], DetectCFG.IMAGE_SIZE[1],3))
        #image = np.array(image)
        #H, W = DetectCFG.IMAGE_SIZE[0], DetectCFG.IMAGE_SIZE[1]
        #image=cv2.resize(image, (W, H))
        # logger.info(image.shape)
        image = image - DetectCFG.IMAGE_MEAN
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).float().unsqueeze(0).cuda()
        logger.info(image.shape)
        
        #aug = transforms.Compose([
        #    transforms.Resize(DetectCFG.IMAGE_SIZE),
        #    transforms.ToTensor(),
            #transforms.Normalize(mean=DetectCFG.IMAGE_MEAN, std=DetectCFG.IMAGE_STD)
        #])
        # tensor_image = aug(image).unsqueeze(0).cuda() ## (1, C, H, W)
        self.model.eval()
        cls, regr = self.model(image)
        return cls, regr
        

if __name__ == "__main__":
    from PIL import Image
    image = Image.open('/home/ubuntu/user/jihye.lee/ocr_exp_v1/마켓컬리_주문.jpg')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    logger.info(image.shape)
    detector = TextDetector()
    detector.run(image)