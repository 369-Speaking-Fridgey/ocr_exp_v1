from google.cloud import storage
import requests
import os
from io import BytesIO
from PIL import Image
from logurtu import logger

BASE_ITEM_PATH = 'https://storage.googleapis.com/ocr_data_v1/data/'

folder_registry = {
    "AIHUB": {
        "IMAGE": "printed_img/form",
        "LABEL": "printed_label/form"
        },
    "CORD": {
        "IMAGE": "train/image",
        "LABEL": "train/json"
    },
    "SROIE": {
        "IMAGE": "train/img",
        "LABEL": "train/box"
    }
}

label_registry = {
    "AIHUB": "printed_label",
    "CORD-1K-001": "json",
    "CORD-1K-002": "json",
    "SROIE2019": "box"
}

class ManagedGCP:
    def __init__(self, project_name,
                    bucket_name, ):
        super(ManagedGCP, self).__init__()
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.client = storage.Client(project = self.project_name)
        self.bucket = storage.Bucket(client = self.client, bucket = self.bucket_name)
        logger.info(
            f"PROJECT - {self.project_name} BUCKET = {self.bucket_name} LOADED..."
        )
        
    def get_image_urls(self, dataset_name, prefix = 'jpg'):
        ## CORD데이터셋은 prefix가 'png'여야 한다.
        urls = []
        for idx, item in enumerate(self.client.list_blobs(
            bucket_or_name = self.bucket,
            prefix = prefix
        )):
            public_url = item.public_url
            if dataset_name in public_url:
                urls.append(public_url)
        if len(urls) == 0:
            for idx, item in enumeratge(self.client.list_blobs(bucket_or_name = self.bucket, prefix = 'png')):
                if dataset_name in item.public_url:
                    urls.append(item.public_url)
                    
        return urls
    
    def get_label_urls(self, dataset_name, image_urls):
        """
        - 만약에 SROIE2019 dataset을 사용하는 상황이었다면 .txt 파일을 prefix로 사용해야 했을 수 있다.
        - 그냥 마음 편하게 image url들을 받아서 json 파일로 바꾸어 주는 것도 좋을 것 같다.
        """
        urls = []
        for img_url in image_urls:
            file_name = os.path.splitext(img_url)[0]
            left = url.replace(BASE_ITEM_PATH, "").split('/')
            
            json_url = os.path.join(BASE_ITEM_PATH,left[0],label_registry[dataset_name.upper()]) + \
                '/'.join(left[2:-1]) + '/' + file_name + '.json'
            urls.append(json_url)
        return urls

    
            
            
        
        