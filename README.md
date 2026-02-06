## 로컬 실행 방법

```bash
git clone https://github.com/kjo5191/lostark-price-app.git
cd lostark-price-app

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run 시세_예측_대시보드.py
```

---
## 📦 LOSTARK-PRICE-APP 프로젝트 구조 설명

이 프로젝트는 로스트아크 거래소 아이템 시세를
여러 머신러닝/시계열 모델로 예측하고,
그 결과를 Streamlit 대시보드로 시각화하는 웹 애플리케이션이다.

전체 흐름은 다음과 같다.
1) DB에서 원시 시세/공지 데이터를 불러온다
2) 전처리 및 Feature Engineering을 수행한다
3) 여러 모델을 학습하거나 저장된 모델을 불러온다
4) 미래 시세를 예측하고 앙상블 결과를 만든다
5) 결과를 대시보드와 투자 시뮬레이션 페이지에 시각화한다


---
### 1. 프로젝트 최상단 파일

시세_예측_대시보드.py
- Streamlit 메인 엔트리 포인트
- 사용자 입력(아이템, 기간, 검증 모델)을 받는다
- 데이터 로드 → 전처리 → feature 생성 → 모델 호출을 총괄한다
- 3대장 앙상블(LightGBM + XGBoost + NeuralProphet) 예측을 기본으로 표시한다
- 검증용 모델(RF/LGBM/XGB/LSTM)을 선택해 성능 비교 그래프를 제공한다
- 미래 예측 그래프를 가장 상단에 배치하고,
  검증 그래프는 expander로 접을 수 있게 구성되어 있다
- 투자 시뮬레이션 페이지로 이동하는 링크를 제공한다


data_loader.py
- 데이터베이스에서 원본 데이터를 불러오는 역할
- 시세 로그 + 아이템 메타 정보를 결합한 데이터프레임을 생성한다
- GPT 기반 공지 점수 데이터를 함께 로드한다
- 모든 분석의 출발점이 되는 데이터 공급자 역할을 한다


preprocess.py
- 시계열 전처리 전담 모듈
- 10분 단위 데이터를 30분 봉으로 리샘플링한다
- GPT 공지 점수를 시계열 구간에 매핑한다
- 롤링 통계 기반으로 이상치를 제거하고 보정한다
- “모델이 학습하기 좋은 시계열”을 만드는 역할을 한다


features.py
- Feature Engineering 전담 모듈
- 아이템 키워드/등급으로 분석 대상 아이템을 선택한다
- lag feature, RSI, Bollinger Band, 시간 변수 등을 생성한다
- 모델에 입력될 df_ml과 feature 컬럼 리스트를 반환한다
- 모든 모델이 공통으로 사용하는 입력 데이터 구조를 만든다


backtest.py
- 예측 결과를 활용한 투자 시뮬레이션 로직
- “이 예측을 기준으로 실제로 투자했다면?”을 가정해 성과를 계산한다
- Streamlit의 투자 시뮬레이션 페이지에서 사용된다


notice.py
- 공지사항 데이터 처리 관련 모듈
- 공지 텍스트를 분석하거나 GPT 점수화 로직이 포함된 파일
- 결과는 DB에 저장되고, 이후 data_loader를 통해 다시 불러온다


models_old.py
- 과거 실험/구현 모델을 보관한 아카이브 파일
- 현재 서비스 로직에서는 사용하지 않는다
- 이전 시도나 참고용 코드로 유지되고 있다


requirements.txt
- 프로젝트 실행에 필요한 Python 패키지 목록
- 새로운 환경에서 `pip install -r requirements.txt`로 세팅한다


.env
- DB 접속 정보, API 키 등 민감한 환경 변수 파일
- Git에는 올라가지 않도록 .gitignore 처리되어 있다


.gitignore
- 가상환경, 캐시, 학습된 모델 파일 등
  Git에 포함되지 않아야 할 항목을 관리한다


README.md
- 프로젝트 개요 및 실행 방법을 설명하는 문서
- 구조 설명과 사용 방법을 정리해두기 위한 파일

---
### 2. models/ 폴더 (머신러닝 계층)

models/base.py
- 모든 가격 예측 모델이 따라야 할 공통 인터페이스 정의
- train / predict_test / predict_future 메서드를 강제한다
- 덕분에 Streamlit 쪽에서는 모델 종류를 몰라도 동일한 방식으로 호출할 수 있다


models/factory.py
- 문자열 키(rf, lgbm, xgb, lstm, np)를
  실제 모델 클래스와 매핑하는 팩토리 역할
- 새로운 모델을 추가해도 이 파일만 수정하면 된다


models/io.py
- 모델 저장/로드 전담 모듈
- 학습된 모델을 joblib으로 파일에 저장한다
- trained_models/{model_key}/{model_key}_item_{item_id}.pkl 형태로 관리한다
- 모델이 이미 있으면 불러오고, 없으면 새로 학습해서 저장한다
- Streamlit에서는 load_or_train_model()만 호출하면 된다


models/random_forest_model.py
- RandomForestRegressor 기반 모델
- 빠르고 안정적인 트리 기반 베이스라인 모델
- auto-regressive 방식으로 미래 예측을 수행한다


models/lightgbm_model.py
- LightGBM 기반 Gradient Boosting 트리 모델
- 예측 성능과 속도의 균형이 좋아 앙상블의 핵심 모델이다
- 미래 예측 로직은 RandomForest와 유사하다


models/xgboost_model.py
- XGBoost 기반 Gradient Boosting 트리 모델
- LightGBM과 다른 부스팅 특성을 활용하기 위해 앙상블에 포함된다


models/lstm_model.py
- Keras 기반 LSTM 시계열 모델
- 시퀀스 패턴 학습에 특화되어 있다
- 현재는 검증(백테스트) 용도로만 사용하고
  미래 예측(predict_future)은 의도적으로 비활성화되어 있다


models/neuralprophet_model.py
- NeuralProphet 기반 시계열 모델
- 트렌드, 시즌성, 공지(GPT 점수) 효과를 함께 모델링한다
- 앙상블에서 보조적인 시계열 관점(가중치 1.0)을 담당한다

---
### 3. trained_models/ 폴더

trained_models/
- 학습이 완료된 모델이 저장되는 디렉터리
- 모델 종류와 아이템 ID 기준으로 파일이 분리된다

예시:
trained_models/
  lgbm/lgbm_item_12345.pkl
  xgb/xgb_item_12345.pkl
  np/np_item_12345.pkl

- 한 번 학습된 모델은 다시 학습하지 않고 재사용된다
- 덕분에 대시보드 응답 속도를 크게 줄일 수 있다

---
### 4. pages/ 폴더

pages/투자_시뮬레이션.py

- 시세 예측 결과를 실제 투자 전략에 적용했을 때의 성과를 가상으로 검증하는 서브 페이지다

- 메인 페이지에서 학습·예측한 결과를 session_state로 받아 재사용할 수 있으며, 이를 기반으로 백테스트를 수행한다

- 입력: 예측 가격(y_pred), 실제 가격(y_test), 투자 파라미터
- 처리: 비율 기반 매수/매도 전략 시뮬레이션
- 출력: 최종 자산, 수익률(ROI), 거래 기록, 투자 판단 요약

- 메인 대시보드가 “미래 시세 예측”에 집중한다면,
이 페이지는 “그 예측으로 실제로 수익을 낼 수 있었는지”를 검증하는 역할을 한다

---
### 5. 기타 디렉터리

venv/
- 프로젝트 전용 Python 가상환경

__pycache__/
- Python 실행 시 자동 생성되는 캐시 디렉터리

lightning_logs/
- NeuralProphet(PyTorch Lightning) 학습 로그
- 현재 서비스에서는 직접 사용하지 않는다

---
### 6. 학습 및 예측 흐름 요약

1) 사용자가 대시보드에서 아이템을 선택한다
2) DB에서 시세/공지 데이터를 불러온다
3) 전처리 및 feature engineering을 수행한다
4) LightGBM / XGBoost / NeuralProphet 모델을 각각 학습 또는 로드한다
5) 3개 모델의 미래 예측값을 가중 평균하여 앙상블 예측을 만든다
6) 앙상블 예측을 대표 결과로 가장 먼저 시각화한다
7) 선택한 검증 모델의 테스트 구간 성능을 추가로 비교한다

이 구조 덕분에
- 모델 추가/교체가 쉽고
- 실험 코드와 서비스 코드가 분리되어 있으며
- 데이터 → 전처리 → 모델 → 시각화 흐름이 명확하게 유지된다

---

## 데모 데이터 생성 방법 (Local / Offline 실행용)

이 프로젝트는 기본적으로 AWS RDS에 저장된 실시간 시세 데이터를 사용합니다.  
하지만 개인 포트폴리오 용도나 오프라인 환경에서도 실행할 수 있도록,
DB 데이터를 CSV 형태로 백업해 로컬 데이터로 사용하는 방식을 지원합니다.

### 데모 데이터 생성 (DB → CSV)

DB 연결이 가능한 환경에서 아래 스크립트를 실행합니다.

```bash
python export_demo_data.py
```