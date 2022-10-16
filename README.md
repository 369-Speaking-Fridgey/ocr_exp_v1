# ocr_exp_v1
Reference: https://rrc.cvc.uab.es/?ch=13&com=introduction
### Experimental repository for Korean OCR 
#### Paddle Paddle OCR (References)
1. Text Detection
2. Detection Box Rectification
3. Text Recognition

#### Text Detection
1. CRAFT: Character-Region Awareness For Text Detection
2. EAST: Efficient and Accurate Scene Text Detection

#### Text Recognition
1. Decoupled Attention Network
2. Backbone
3. Neck
4. Head

```
## all the configs folder contains the configuration files to differentiate the training settings
key_info_extraction: Pre-trained BERT, 혹은 RoBERTa 모델로 classification을 하려 한다.
    |__ model
        |__ transformer: Loads the pretrained
        |__ classifier
    |__ data
    |__ configs
    |__ tools
        |_ train.py
        |_ program.py
        |_ test.py

recipt_ocr: recipe 이미지에 있는 모든 문자는 텍스트로 인식해서 출력할 수 있게 한다.
    |__ configs
    |__ STR_1: based on the Clova AI Paper
        |__ transformation: for image refignment
        |__ feature extraction
        |__ Sequence Modeling
        |__ Prediction
    |__ STR_2: based on the semantic reasoning network
text_localization: CRAFT, EAST, DBNet등과 같은 모델을 사용해서 text 영역의 bounding box 찾기
    |__ DBNet: Differentiable Binarization
```


#### for tracking the model with mlflow..
1. go to your main project folder `/ocr_exp_v1`
2. run the command below and browse to `http://127.0.0.1:5000/#/experiments/1`
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root $(pwd)/artifacts
```