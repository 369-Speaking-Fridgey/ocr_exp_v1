FOLDER = '/home/ubuntu/user/jihye.lee/data/new_box_data.zip'
import zipfile
import io, os
from loguru import logger
import numpy as np
from PIL import Image
import cv2
mean = [0.0,0.0,0.0]
std = [0.0,0.0,0.0]
data_archive = zipfile.ZipFile(FOLDER, 'r')
folders = os.listdir('/home/ubuntu/user/jihye.lee/data/detection_aihub')
cnt = 0
for folder in folders:
    if folder == 'box_data.zip':
        continue
    else:
        image_archive = zipfile.ZipFile(os.path.join('/home/ubuntu/user/jihye.lee/data/detection_aihub', folder), 'r')

        image_files = image_archive.namelist()
#logger.info(f"FILE NO = {len(image_files)}")
        for f in image_files:
            image_file = image_archive.read(image_files[0])
            image = Image.open(io.BytesIO(image_file))
            image = np.array(image)
            for i in range(3):
                mean[i] += image[:,:,i].mean()
                std[i] += image[:,:,i].std()
            cnt += 1
mean /= cnt
std /= cnt
logger.info(f"MEAN: {mean} STD: {std}")
"""
fname = image_files[0].replace('jpg', 'txt').split('/')[-1]
data = data_archive.read(fname).decode('utf-8')
info = data.split('\n')
for i in info:
    if i == '':
        continue
    else:
        i = [int(j) for j in i.split(' ')[:8]]
        x1, y1, x2, y2, x3, y3, x4, y4 = i
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(image, (x2, y2), (x3, y3), (0, 0, 255), 2)
        cv2.arrowedLine(image, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv2.line(image, (x4, y4),(x1, y1), (0, 0, 255), 2)
cv2.imwrite( './test.jpg',image)
"""