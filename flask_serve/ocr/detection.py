import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from loguru import logger
import os, sys
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(BASE))
from text_detection.utils.ctpn_utils import ctpn_detect, ctpn_data_utils
from text_detection.detect_model.ctpn.ctpn import CTPN
from torchvision import transforms as transforms

class DetectCFG:
    PRETRAINED_WEIGHT='/home/ubuntu/user/jihye.lee/ocr_exp_v1/text_detection/weight/ctpn.pth'
    LINE_MIN_SCORE=0.7
    TEXT_PROPOSALS_MIN_SCORE=0.5
    TEXT_PROPOSALS_NMS_THRESH=0.5
    MAX_HORIZONTAL_GAP=20
    MIN_V_OVERLAPS=0.7
    MIN_SIZE_SIM=0.7
    ANCHOR_SHIFT=16
    #IMAGE_SIZE=[1024, 2048]
    IMAGE_MEAN=[123.68, 116.779, 103.939]
    #IMAGE_STD=[0.20037157, 0.18366718, 0.19631825]
    #IMAGE_MEAN = [0.90890862, 0.91631571, 0.90724233]


class TextDetector(object):
    def __init__(self):
        super(TextDetector, self).__init__()
        self.model = CTPN().cuda()
        pretrained_weight = torch.load(DetectCFG.PRETRAINED_WEIGHT)
        if 'model_state_dict' in pretrained_weight:
            pretrained_weight = pretrained_weight['model_state_dict']
        self.model.load_state_dict(pretrained_weight)
        self.text_proposal_connector = ctpn_data_utils.TextProposalConnectorOriented()
    
    def run(self, image):
        """ Args
        image: PIL Image (근데 원래 알고 있는대로 이미지를 cv2 형태로 읽으려면 np.array로 바꾸어주는것이 맞다.)
        Outputs
        new_text: list형태인데 모든 감지된 bounding box의 (minx, miny, maxX, maxY)의 값을 갖는다.ㄴ
        """
        H, W, C = np.array(image).shape ## 원본 이미지
        copy_image = image.copy()
        logger.info(f"{H}, {W}")
        rescale_factor = max(H, W) / 1000
        if rescale_factor > 1.0:
            new_h = int(H / rescale_factor)
            new_w = int(W / rescale_factor)
            image = cv2.resize(image, (new_w, new_h))
        else:
            new_h = H
            new_w = W
        ## (0) 나중에 text box를 원본 이미지의 ratio에 맞춰 주어야 하니까 비율을 미리 계산해 둔다.
        # rescale_w = new_w / DetectCFG.IMAGE_SIZE[1] 
        # rescale_h = H/ DetectCFG.IMAGE_SIZE[0] 
        scale = np.array([[rescale_factor, rescale_factor, rescale_factor, rescale_factor]])
        ## (1) Get the output of the model
        cls, regr = self.get_prediction(image)
        ## (2) Get the Detection Box based on the output
        selected_anchor, selected_score = self.get_detection(cls, regr, new_w, new_h)
        ## (3) Get the Text Lines
        text, new_text = self.text_proposal_connector.get_text_lines(selected_anchor, selected_score, (new_h, new_w))
        
        ## (4) Rescale the Text Lines
        new_text = []
        for t in text:
            t = [int(i) for i in t][:8]
            x1, y1, x2, y2, x3, y3, x4, y4 = t
            X = [x1,x2,x3,x4];Y = [y1,y2,y3,y4]
            new_text.append([min(X), min(Y), max(X), max(Y)])
        if rescale_factor > 1.0:
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
            cv2.rectangle(copy_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
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