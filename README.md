# Evaluation
-------------------------
## 0. Requirement
- absl-py
- numpy
- pandas
- nltk
## 1. bbox evaluation
**Quick start**
- python mAP.py -gt ./path_to_ground_truth -pred ./path_to_prediction
### 1.1 mAP
![map 사진](./imgs/map_total.jpg)
## 2. ocr evaluation
**Quick start**
- python ocr_eval.py -gt ./path_to_ground_truth -pred ./path_to_prediction -wem
### 2.1 1-NED
### 2.2 WEM