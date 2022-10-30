1. https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/iet-cvi.2019.0916
2. https://arxiv.org/pdf/1911.08947.pdf
3. https://www.ijcai.org/proceedings/2020/0072.pdf

#### SETUP
- nms and bbox utils are written in cython, so you must build the library first


#### TODO LIST
- [X] Overal Geometry Loss
- [X] Pretrained Backbones (VGG, RESNET)
- [X] EAST Model Implementation
- [X] Score and geo map utils Code
- [X] Finish the `base_runner.py`
- [X] Finish conecting `mlflow` & `hydra`
- [ ] Code for the geo and score map generation
- [ ] Dataset Code
- [ ] Train Runner Code
- [ ] Evaluation Code
- [ ] Pre-Training PVANet with IMAGENET Dataset 
- [ ] Connect MLFlow & ONNX for model deployment and experiments