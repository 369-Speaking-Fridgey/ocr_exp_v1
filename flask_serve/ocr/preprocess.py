""" preprocess.py
(1) Crop Image
(2) To Morphology
(3) Convert Color to grayscale image
(4) Get the Sobel Gradient to crop only the receipt part
"""
import cv2
import numpy as np
## Step1: Read image in Gray scale
def step1(img_path):
  img = cv2.imread(img_path)
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img, gray_img


## Step2: Calculate the threshold for each image
def step2(gray_img):
  blurred = cv2.GaussianBlur(gray_img, ksize = (3, 3), sigmaX = 7)
  ## gradient (to get the gradient, we must remove the noise first)
  gradX = cv2.Sobel(blurred,  ddepth=cv2.CV_32F, dx=1, dy=0)
  gradY = cv2.Sobel(blurred, ddepth = cv2.CV_32F, dx = 0, dy = 1)
  gradient = cv2.subtract(gradX, gradY)
  gradient = cv2.convertScaleAbs(gradient)
  # thresh_and_blur
  blurred = cv2.GaussianBlur(gradient, (3, 3), 7)
  (_, thresh) = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
  return thresh

## Step3: Need to set the ellipse size at first and do morphological thing.
def step3(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,# cv2.MORPH_ELLIPSE, 
                                       (int(thresh.shape[1]/5), int(thresh.shape[0]/5)))
    morpho_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    ## Opening 과정을 통해서 작은 객체나 돌기 제거 등을 한다.
    morpho_image = cv2.erode(morpho_image, None, iterations=1)
    morpho_image = cv2.dilate(morpho_image, None, iterations=1)

    return morpho_image

## Step4: Based on the morpho_image, get the bounding box of the recipe part
def step4(morpho_image, original_image):
  H, W, C = original_image.shape
  contours, hierarchy = cv2.findContours(morpho_image.copy(),
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
  c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
  area = cv2.minAreaRect(c) ## ((x1, y1), (x2, y2), angle)
  box = cv2.boxPoints(area) ## 왼쪽 아래 좌표에서부터 시계방향으로 8box point를 반환
  box = np.int0(box)
  ## 그냥 그리면 원본 이미지에 그려지니까 반드시 copy를 해서 그려야 한다.
  draw_box = cv2.drawContours(original_image.copy(), [box], 0, (0, 0, 255), 2)
  # x1, y1, x2, y2, x3, y3, x4, y4 = box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]
  X = [p[0] for p in box]
  Y = [p[1] for p in box]
  x1 = max(min(X), 0);x2 = min(max(X), W)
  y1 = max(min(Y), 0);y2 = min(max(Y), H)
  height = y2-y1
  width = x2-x1
  croped = original_image[y1:y1+height, x1:x1+width]

  return croped, x1, y1, draw_box

def preprocess_for_recognition(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = step2(gray_image)
    morpho_image = step3(thresh)
    croped, x1,y1,box = step4(morpho_image, image)
    new_H, new_W, _ = croped.shape
    H, W, C = image.shape
    org = np.ones((H, W)) * gray_image[0,0]
    org[y1:y1 + new_H, x1:x1 + new_W] = gray_image[y1:y1 + new_H, x1:x1 + new_W] #croped
    cv2.imwrite('/home/ubuntu/user/jihye.lee/ocr_exp_v1/flask_serve/img_test_res/box.png', box)
    cv2.imwrite('/home/ubuntu/user/jihye.lee/ocr_exp_v1/flask_serve/img_test_res/org.png',org)
    return org

def preprocess_for_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = step2(gray_image)
    morpho_image = step3(thresh)
    croped, x1,y1,_ = step4(morpho_image, image)
    new_H, new_W, _ = croped.shape
    H, W, C = image.shape
    org = np.ones((H, W, C)) * 255
    org[y1:y1 + new_H, x1:x1 + new_W, :] = croped
    return org