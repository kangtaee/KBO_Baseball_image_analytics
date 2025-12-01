# ⚾ KBO Baseball Image Analytics (YOLOv8 & FastAPI)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00FFFF?logo=yolo&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Server-009688?logo=fastapi&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Frontend-Bootstrap5-7952B3?logo=bootstrap&logoColor=white)

## 1. 프로젝트 개요 (Overview)
본 프로젝트는 **KBO 리그 야구 경기 하이라이트 영상**에서 투수, 타자, 심판 등 **7개 주요 포지션**을 자동으로 식별하는 딥러닝 객체 탐지(Object Detection) 모델을 개발하고, 이를 쉽게 활용할 수 있는 **웹 대시보드**를 제공합니다.

`YOLOv8m` 모델을 기반으로 학습하여 높은 정확도를 확보하였으며, `FastAPI`를 통해 사용자가 직접 이미지나 동영상을 업로드하고 분석 결과를 시각적으로 확인할 수 있습니다.

---

## 2. 데이터셋 및 클래스 (Dataset & Classes)
KBO 경기 영상에서 추출한 이미지를 사용하여 7개의 클래스로 라벨링하였습니다.

### 📌 클래스 분포 및 데이터 (Labels & Instances)
학습 데이터의 클래스 분포와 객체 크기/위치 분포는 아래와 같습니다.

![Label Distribution](runs/style/labels.jpg)

| ID | Class Name | 설명 |
|:---:|:---|:---|
| 0 | **judge** | 심판 (주심 및 루심) |
| 1 | **batter** | 타자 |
| 2 | **catcher** | 포수 |
| 3 | **pitcher** | 투수 |
| 4 | **infielder** | 내야수 (1, 2, 3루수, 유격수) |
| 5 | **outfielder** | 외야수 (좌, 중, 우익수) |
| 6 | **runner** | 주자 |

---

## 3. 모델 학습 (Model Training)
* **Model:** YOLOv8m (Medium)
* **Epochs:** 50
* **Batch Size:** 16
* **Image Size:** 640

### 🖼️ 학습 데이터 예시 (Training Batches)
모델이 학습 과정에서 실제로 입력받은 데이터 배치(Mosaic Augmentation 적용) 예시입니다.
![Train Batch](runs/style/train_batch0.jpg)

### 📈 학습 결과 그래프 (Training Results)
학습 진행에 따른 Loss 감소와 성능 지표(Precision, Recall, mAP) 변화 추이입니다.
![Results](runs/style/results.png)

---

## 4. 성능 평가 (Evaluation)
검증 데이터셋(Validation Set)에 대한 정량적 평가 결과입니다.

### 📊 혼동 행렬 (Confusion Matrix)
클래스별 예측 정확도를 시각화한 결과입니다. 특징이 뚜렷한 **투수, 타자, 포수** 클래스에서 특히 높은 정확도를 보입니다.
![Confusion Matrix](runs/style/confusion_matrix.png)

### 📉 PR Curve & F1 Curve
모델의 신뢰도(Confidence)에 따른 Precision-Recall 및 F1 Score 곡선입니다.
<p align="center">
  <img src="runs/style/BoxPR_curve.png" width="48%" />
  <img src="runs/style/BoxF1_curve.png" width="48%" />
</p>

---

## 5. 예측 결과 시각화 (Inference Examples)
학습된 모델을 사용하여 실제 경기 장면을 추론한 결과입니다. 다중 객체(심판, 타자, 포수, 투수 등)가 혼재된 상황에서도 안정적으로 탐지하는 것을 확인할 수 있습니다.

### ✅ Validation Batch 예측 결과
![Val Batch 0](runs/style/val_batch0_pred.jpg)
![Val Batch 1](runs/style/val_batch1_pred.jpg)
![Val Batch 2](runs/style/val_batch2_pred.jpg)

---

## 6. 설치 및 실행 (Installation & Usage)

### 1️⃣ 환경 설정
```bash
# 필수 패키지 설치
pip install ultralytics fastapi uvicorn jinja2 python-multipart
