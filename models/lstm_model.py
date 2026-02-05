# models/lstm_model.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from .base import BasePriceModel


class LSTMPriceModel(BasePriceModel):
	def __init__(self, window_size: int = 48):
		# 하이퍼파라미터
		self.window_size = window_size

		# 데이터/모델 관련 상태
		self.model = None
		self.df = None
		self.features = None

		self.scaler_X = MinMaxScaler()
		self.scaler_y = MinMaxScaler()

		self.X_seq = None
		self.y_seq_scaled = None
		self.y_seq_real = None
		self.date_seq = None

		# 평가 정보
		self.split_idx = None		# 원본 df 기준 split index
		self.y_test = None
		self.y_pred = None
		self.rmse = None
		self.r2 = None

		# 미래 예측용 부가 정보
		self.price_feat_index = None	# features 내에서 'price' 위치
		self.freq = None				# 시계열 간격 (예: 10분)
		self.last_date = None			# 마지막 시점


	def train(self, df: pd.DataFrame, features: list[str]):
		"""
		df: feature + price + date 를 포함한 전체 데이터프레임
		features: LSTM에 사용할 feature 컬럼 목록
		"""
		self.df = df
		self.features = features

		# 0. LSTM 전용 feature 세트 구성
		#    - 외부 features에는 price가 없어도 됨
		#    - LSTM 내부 입력에는 price를 추가해서 사용
		if "price" in features:
			lstm_features = features
			self.price_feat_index = features.index("price")
		else:
			lstm_features = features + ["price"]
			self.price_feat_index = len(lstm_features) - 1	

		# 1. raw 값 추출
		# X_raw = df[features].values			# (N, F)
		X_raw = df[lstm_features].values			# (N, F_lstm)
		y_raw = df["price"].values			# (N,)
		dates = df["date"].values			# 시각화용

		# 2. 스케일링
		X_scaled = self.scaler_X.fit_transform(X_raw)					# (N, F)
		y_scaled = self.scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()	# (N,)

		# 3. 시퀀스(Window) 생성
		window_size = self.window_size

		X_seq = []
		y_seq_scaled = []
		y_seq_real = []
		date_seq = []

		for i in range(window_size, len(df)):
			# 과거 window_size 개 시점
			X_seq.append(X_scaled[i - window_size : i, :])		# (window_size, F)
			y_seq_scaled.append(y_scaled[i])					# scaled target
			y_seq_real.append(y_raw[i])							# 원래 price
			date_seq.append(dates[i])

		X_seq = np.array(X_seq)				# (N_seq, window, F)
		y_seq_scaled = np.array(y_seq_scaled)
		y_seq_real = np.array(y_seq_real)
		date_seq = np.array(date_seq)

		self.X_seq = X_seq
		self.y_seq_scaled = y_seq_scaled
		self.y_seq_real = y_seq_real
		self.date_seq = date_seq

		# 4. Train/Test 분리 (시계열: 앞 80% / 뒤 20%)
		n_seq = len(X_seq)
		split_idx_seq = int(n_seq * 0.8)		# 시퀀스 기준 split

		X_train_seq = X_seq[:split_idx_seq]
		X_test_seq = X_seq[split_idx_seq:]

		y_train_scaled = y_seq_scaled[:split_idx_seq]
		y_test_scaled = y_seq_scaled[split_idx_seq:]

		# 평가 & 시각화용 real 값
		y_test_real = y_seq_real[split_idx_seq:]
		test_dates_seq = date_seq[split_idx_seq:]

		# 원본 df 기준 split index (window offset 고려)
		# self.split_idx = window_size + split_idx_seq
		self.split_idx = len(df) - len(y_test_real)


		# 5. LSTM 모델 정의
		timesteps = X_train_seq.shape[1]
		feature_dim = X_train_seq.shape[2]

		model = Sequential(
			[
				LSTM(64, input_shape=(timesteps, feature_dim)),
				Dense(32, activation="relu"),
				Dense(1),
			]
		)

		model.compile(optimizer="adam", loss="mse")

		early_stopping = EarlyStopping(
			monitor="val_loss",
			patience=5,
			restore_best_weights=True,
		)

		# 6. 학습
		history = model.fit(
			X_train_seq,
			y_train_scaled,
			validation_split=0.1,
			epochs=50,
			batch_size=64,
			callbacks=[early_stopping],
			shuffle=False,
			verbose=0,		# Streamlit 로그 과한 출력 방지
		)

		self.model = model

		# 7. 테스트 구간 예측 및 역스케일링
		y_pred_scaled = model.predict(X_test_seq).flatten()
		y_pred_lstm = (
			self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
			.flatten()
		)

		# 8. 성능 지표 (원래 단위 기준)
		rmse_lstm = np.sqrt(mean_squared_error(y_test_real, y_pred_lstm))
		r2_lstm = r2_score(y_test_real, y_pred_lstm)

		self.y_test = pd.Series(y_test_real)
		self.y_pred = pd.Series(y_pred_lstm)
		self.rmse = rmse_lstm
		self.r2 = r2_lstm

		# ---- 여기부터 추가: 시계열 간격(frequency) 및 마지막 시점 저장 ----
		dates_dt = pd.to_datetime(df["date"])
		freq = dates_dt.diff().median()
		# 데이터가 너무 적거나 이상하면 fallback
		if pd.isna(freq):
			freq = pd.Timedelta(minutes=10)

		self.freq = freq
		self.last_date = dates_dt.iloc[-1]		


	def predict_test(self):
		"""
		검증 구간 예측 결과 반환
		(RandomForestPriceModel / LightGBMPriceModel 과 인터페이스 동일)
		"""
		return (
			self.y_test,
			self.y_pred,
			self.split_idx,
			self.rmse,
			self.r2,
		)


	def predict_future(self, steps: int):
		"""
		공식적으로는 LSTM을 '백테스트/시계열 패턴 분석용'으로만 사용하고,
		미래 예측은 트리 기반 모델(RF/LightGBM)에 맡기기로 했음.

		※ 아래에 _predict_future_naive()로 실험용 구현은 남겨두었지만,
		   실제 앱에서는 사용하지 않고, 이 메서드는 항상 NotImplementedError를 던진다.
		"""
		raise NotImplementedError(
			"LSTM 모델은 현재 미래 예측 기능(predict_future)이 비활성화되어 있습니다.\n"
			"백테스트 및 과거 구간 예측 평가용으로만 사용하세요."
		)


	def predict_future_naive(self, steps: int):
		"""
		[실험용] LSTM naive roll-out 기반 미래 예측 구현.

		- 실제 앱에서는 사용하지 않는다.
		- 추후 연구/실험용으로 사용할 수 있도록 남겨둔 코드.

		※ 문제점
		  - price만 업데이트하고 기타 feature(RSI, Bollinger 등)는 고정이라
		    멀티스텝 예측에서 예측선이 평균값 근처로 수렴/붕괴하는 현상이 발생.
		  - 제대로 쓰려면 step마다 feature 재계산이 필요함.

		return: DataFrame(date, price)
		"""
		# 0. 사전 체크
		if self.model is None or self.X_seq is None:
			raise RuntimeError("먼저 train()을 호출해야 합니다.")

		if self.price_feat_index is None:
			# price가 feature에 포함되어 있지 않으면, 현재 로직으로는 미래 예측 불가
			raise NotImplementedError(
				"'price' 컬럼이 features 목록에 없어 LSTM 미래 예측을 수행할 수 없습니다."
			)

		if self.freq is None or self.last_date is None:
			raise RuntimeError("시계열 간격(freq) 또는 last_date 정보가 없습니다. train()이 제대로 수행됐는지 확인하세요.")

		# 1. 마지막 시퀀스를 복사해서 시작점으로 사용
		#    shape: (window_size, feature_dim)
		last_seq = self.X_seq[-1].copy()
		window_size, feature_dim = last_seq.shape

		future_scaled = []
		future_real = []
		future_dates = []

		current_date = self.last_date

		for step in range(steps):
			# 2-1. 현재 window로 다음 시점 예측 (scaled space)
			input_batch = last_seq.reshape(1, window_size, feature_dim)
            # model.predict 결과: shape (1, 1)
			next_scaled = self.model.predict(input_batch, verbose=0).flatten()[0]

			# 2-2. 역스케일링해서 실제 가격 단위로 변환
			next_real = (
				self.scaler_y.inverse_transform(np.array([[next_scaled]]))
				.flatten()[0]
			)

			# 2-3. 날짜 갱신
			current_date = current_date + self.freq

			future_scaled.append(next_scaled)
			future_real.append(next_real)
			future_dates.append(current_date)

			# 2-4. window roll:
			#      - 앞의 것 한 칸씩 땡기고
			#      - 맨 뒤 row 를 "최근 상태"로 채운 후, price 차원만 업데이트
			new_seq = np.roll(last_seq, shift=-1, axis=0)  # 위로 한 칸씩 당김
			new_seq[-1] = new_seq[-2]                       # 마지막 row 를 직전 row로 복사
			new_seq[-1, self.price_feat_index] = next_scaled  # price 차원을 새 예측값으로 교체

			last_seq = new_seq

		# 3. 결과 DataFrame 생성 (대시보드에서 기대하는 형태)
		future_df = pd.DataFrame({
			"date": future_dates,
			"price": future_real,
		})

		return future_df

