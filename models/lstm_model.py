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

	def train(self, df: pd.DataFrame, features: list[str]):
		"""
		df: feature + price + date 를 포함한 전체 데이터프레임
		features: LSTM에 사용할 feature 컬럼 목록
		"""
		self.df = df
		self.features = features

		# 1. raw 값 추출
		X_raw = df[features].values			# (N, F)
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
		LSTM의 미래 예측은 현재 구조(df_ml 기반 feature 생성)에
		딱 맞게 설계하려면 별도의 시뮬레이션 로직이 필요해서,
		일단은 명시적으로 미구현 처리해둠.
		"""
		raise NotImplementedError("LSTM은 아직 미래 예측 기능(predict_future)을 지원하지 않습니다.")
