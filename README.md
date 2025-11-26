# Intro to AI: Machine Learning Basic Projects

실제 데이터를 활용하여 **Classification, Regression, Clustering** 세 가지 태스크를 수행하였으며 PyTorch와 NumPy를 활용해 모델을 구현하고 하이퍼파라미터 튜닝에 따른 성능 변화를 분석했다.

## Project Overview

| Task | Dataset | Model / Algorithm | Key Library |
|------|---------|-------------------|-------------|
| **Classification** | Breast Cancer Wisconsin (Diagnostic) | MLP (Multi-Layer Perceptron) | PyTorch |
| **Regression** | Medical Cost Personal Datasets | MLP Regression | PyTorch |
| **Clustering** | Body Fat Prediction Dataset | K-Means (Implemented from scratch) | NumPy |

# 1. Classification: 유방암 진단 (Breast Cancer Diagnosis)

### Goal
유방암 데이터셋의 30가지 특징(Features)을 기반으로 종양을 악성(Malignant)과 양성(Benign)으로 분류한다.

### Implementation Details
* **Data Split:** Train / Validation / Test (6:2:2)
* **Model Architecture:**
    * Input(30) -> Hidden(80) -> Hidden(80) -> Hidden(80) -> Output(2)
    * Activation: ReLU
* **Key Hypothesis & Experiment:**
    * 초기 예시 코드(Small Model, Short Training) 대비 모델의 깊이(Depth)와 너비(Width)를 늘리고 학습(Epoch)을 오래 시키면 성능이 향상될 것이라 가정함
    * 실험 결과, Validation Accuracy가 안정적으로 90% 이상을 유지함을 확인함

### Results
* **Test Accuracy:** **96.49%**
* **Confusion Matrix:** 악성(M)을 양성(B)으로 잘못 예측하는(False Negative) 비율을 최소화하여 의료 데이터로서의 신뢰성을 확보함

# 2. Regression: 개인 의료비 예측 (Medical Cost Prediction)

### Goal
나이, 성별, BMI, 자녀 수, 흡연 여부 등의 정보를 바탕으로 개인의 연간 의료비(Charges)를 예측한다.

### Implementation Details
* **Preprocessing:** One-hot Encoding (Region), Label Encoding (Sex, Smoker)
* **Model Architecture:**
    * Deep MLP (Depth: 7 layers, Width: 110 units)
* **Key Hypothesis & Experiment:**
    * Learning Rate를 0.005로 상향하고 Batch Size를 110으로 증가시켜 학습 속도와 수렴 안정성을 동시에 확보하고자 함
    * 학습 과정에서 Loss의 진동(Vibration)이 관찰되었으나, 전체적으로 Loss가 우하향하며 수렴함

### Results
* **Test RMSE:** 약 **4,712 USD**
* **Insight:** 데이터 스케일링(Scaling) 부재로 인해 초기 Loss가 컸으나, 모델 용량(Capacity)을 늘려 유의미한 예측 성능을 달성함

# 3. Clustering: 체형 군집화 (Body Fat Clustering)

### Goal
체지방률(BodyFat)과 몸무게(Weight) 데이터를 기반으로 사람들의 체형을 비지도 학습을 통해 군집화한다.

### Implementation Details
* **Algorithm:** K-Means Clustering (Scikit-learn 없이 NumPy로 직접 구현)
* **Process:**
    1. **Initialization:** K=3개의 중심점(Centroids) 랜덤 설정
    2. **E-Step:** 각 데이터 포인트를 가장 가까운 중심점에 할당
    3. **M-Step:** 군집의 평균 위치로 중심점 업데이트
    4. 수렴할 때까지 반복

### Results
* **Visual Analysis:** 산점도(Scatter Plot)를 통해 반복(Iteration) 횟수에 따라 중심점이 데이터의 밀집 지역으로 이동하는 과정을 시각화함
* **Clusters:**
    * Cluster 0: 저체중 / 저체지방
    * Cluster 1: 고체중 / 고체지방 (고위험군)
    * Cluster 2: 평균 체중 / 평균 체지방
