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

#### mAP 계산 주의사항
- score 기준 정렬후 높은 score 먼저 bbox 비교 실행
- iou가 가장높은 박스를 동시에 예측한경우 스코어가 가장 높은 예측만 tp (repeated match)
- iou 계산 시 양끝좌표 고려하여 + 1

## 2. ocr evaluation
**Quick start**
- python ocr_eval.py -gt ./path_to_ground_truth -pred ./path_to_prediction -wem
### 2.1 1-NED
- Normalized Edit Distance
- 실제 단어와 예측 단어의 편집 거리를 측정한 뒤 긴 단어 길이로 정규화 후 측정
>- 편집 거리 : 삽입, 수정, 삭제를 통해 다른 단어로 변환하는데 필요한 최소 연산
>- NED : 편집거리 / 긴 단어의 길이
>- 1-NED 의 경우 1에 가까울 수록 좋은 성능
### 2.2 WEM
- Word based Exactly Matching
- 실제 단어와 예측 단어를 비교하여 일치여부를 판단하여 평가
- 단어가 일치하는 경우에만 참으로 판별 (1-ned에 비해 유사도 같은 사항을 반영하지 못함)
>- 한글자라도 다른 경우 False
>- WEM : True count / (False count + True count)
>- WEM의 경우 1에 가까울 수록 좋은 성능

