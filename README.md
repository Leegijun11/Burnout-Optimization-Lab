# Developer Burnout Prediction using Deep Learning

개발자의 업무 환경 및 생활 습관 데이터를 바탕으로 **번아웃 지수(Low, Medium, High)**를 예측하는 딥러닝 프로젝트입니다. 단순히 모델을 만드는 것에 그치지 않고, 다양한 전처리 실험을 통해 성능을 최적화하는 과정을 기록했습니다.

## 1. 프로젝트 개요
* **목표**: 개발자의 번아웃 수준을 예측하여 조기 경고 시스템의 기초 모델 구축
* **데이터셋**: 7,000건의 개발자 활동 데이터 (업무 시간, 수면, 커밋 수, 카페인 섭취 등)
* **기술 스택**: Python, PyTorch, Pandas, Scikit-learn, Matplotlib, Seaborn

## 2. 실험 및 최적화 과정 (Step-by-Step)
최적의 모델을 찾기 위해 다음과 같은 순차적 실험을 진행했습니다.

### [Exp 1] Feature Selection
- 번아웃과 상관계수가 낮은 피처를 포함했을 때와 제외했을 때의 성능 비교
- 불필요한 노이즈 제거를 통한 모델 효율성 확인

### [Exp 2] Normalization Comparison
- `StandardScaler` vs `MinMaxScaler` 성능 비교
- 데이터 특성상 0~1 사이로 범위를 제한하는 **MinMaxScaler**가 학습 안정성에 기여함을 확인

### [Exp 3] Missing Value Handling
- 결측치를 **평균값으로 대체(Mean)**하는 방식과 **행 제거(Drop)** 방식 비교
- 데이터 손실을 최소화하는 평균값 대체 방식 채택

### [Development] Model Enhancement (Feature Engineering & SMOTE)
- **파생 변수 도입**: `work_sleep_ratio`(수면 대비 업무 비율) 등 4종의 파생 변수 생성
- **클래스 불균형 해소**: `SMOTE`를 적용하여 데이터 불균형 문제 해결

## 3. 실험 결과 및 한계점 (Key Findings)
다양한 기법을 동원하여 성능 개선을 시도한 결과는 다음과 같습니다.

* **최종 성능**: Test Accuracy **75% ~ 80%** 수렴
* **분석**:
    1. 파생 변수와 SMOTE를 적용했음에도 불구하고 정확도가 80% 벽을 넘지 못하는 현상 확인.
    2. **데이터의 선형적 한계**: 혼동 행렬(Confusion Matrix) 분석 결과, `Medium` 클래스가 `Low`와 `High` 사이에서 모호하게 중첩되어 있음.
    3. 이는 모델 구조의 문제라기보다, 현재 제공된 수치형 피처만으로는 클래스 간 경계를 완벽히 분리하기 어려운 **데이터셋 자체의 한계**로 판단됨.

## 4. 파일 구성
- `data/`: 분석에 사용된 데이터셋
- `notebooks/`: 단계별 전처리 및 개선 실험 스크립트 (`experiment1_feature.py`, `experiment2_normalize.py`, `experiment3_missing.py`)
- `final_model.py`: 최적의 하이퍼파라미터가 적용된 최종 예측 모델
- `develop_model.py`: 최적의 하이퍼파라미터와 SMOTE, 파생 변수를 적용한 최적화 시도 모델
