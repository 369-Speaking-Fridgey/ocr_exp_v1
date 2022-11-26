from flask import Flask, request, render_template
import os, sys
from PIL import Image
import cv2
import numpy as np
import io
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from ocr.detection import TextDetector
from ocr.recognition import Recognition
ROOT=os.path.dirname(os.path.abspath(__file__))
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('demo.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['file']
        img_bytes = image.read()
        image = Image.open(io.BytesIO(img_bytes))
        """ CAUTION
        - 여기서는 우선은 그냥 image를 성공적으로 받았다는 표현만 하지만 사실을 이미지를 
        """
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detector = TextDetector()
        box_image, box_dict = detector.run(image)
        cv2.imwrite(os.path.join(ROOT,'static', 'org.jpg'), image)
        cv2.imwrite(os.path.join(ROOT, 'static', 'box.jpg'), box_image)
        return render_template('predict.html', 
                                box_img='/static/box.jpg', 
                                original_img='/static/org.jpg')
        

if __name__ == "__main__()":
    app.run(debug = True)