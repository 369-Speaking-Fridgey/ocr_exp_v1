FOLDER = '/home/ubuntu/user/jihye.lee/data/detection_aihub'
import zipfile
import io, os
from loguru import logger

FILES = os.listdir(FOLDER)
for fname in FILES:
    if fname != 'box_data.zip':
        image_zip = os.path.join(FOLDER, fname)
        image_archive = zipfile.ZipFile(image_zip, 'r')
        image_files = image_archive.namelist()
        logger.info(f"FILE NO = {len(image_files)}")
        for f in image_files:
            try:
                img = image_archive.read(f)
            except:
                logger.info(f"UNABLE TO READ {f}")