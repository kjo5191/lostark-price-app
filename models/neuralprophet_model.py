# models/neuralprophet_model.py

import numpy as np
import pandas as pd

from neuralprophet import NeuralProphet, save, load

from .base import BasePriceModel


class NeuralProphetPriceModel(BasePriceModel):
	def __init__(
		self,
		forecast_horizon: int = 144,  # 3일(30분단위) = 3 * 24 * 2 = 144
		n_lags: int = 240,           # 과거 5일(30분단위) = 5 * 24 * 2 = 240
	):
		self.forecast_horizon = forecast_horizon
		self.n_lags = n_lags

		self.model: NeuralProphet | None = None
		self.df_np: pd.DataFrame | None = None  # ds, y, GPT_Score
		self.trained_until: pd.Timestamp | None = None

		# 평가 정보 (일단 대략적인 backtest용)
		self.y_test = None
		self.y_pred = None
		self.split_idx = None
		self.rmse = None
		self.r2 = None

	def _build_np_df(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Streamlit에서 넘어오는 df_ml을 그대로 받아서
		NeuralProphet이 요구하는 형식(ds, y, GPT_Score)으로 변환.
		df는 최소한 date, price, gpt_score(선택) 컬럼이 있다고 가정.
		"""
		df = df.copy()
		df = df.sort_values("date")

		df_np = pd.DataFrame({
			"ds": df["date"],
			"y": df["price"].astype(float),
		})

		if "gpt_score" in df.columns:
			df_np["GPT_Score"] = df["gpt_score"].astype(float)
		else:
			df_np["GPT_Score"] = 0.0

		# 결측 제거
		df_np = df_np.dropna(subset=["ds", "y"])

		return df_np

	def train(self, df: pd.DataFrame, features: list[str]):
		"""
		BasePriceModel 인터페이스 맞추기용.
		- features는 여기선 사용하지 않고 df에서 직접 ds/y/GPT_Score를 뽑아쓴다.
		"""
		df_np = self._build_np_df(df)
		self.df_np = df_np
		self.trained_until = pd.to_datetime(df_np["ds"].max())

		m = NeuralProphet(
			n_forecasts=self.forecast_horizon,
			n_lags=self.n_lags,
			n_changepoints=10,
			trend_reg=1.0,
			weekly_seasonality=True,
			daily_seasonality=True,
			yearly_seasonality=False,
			learning_rate=0.01,
			growth="off",
		)

		# GPT 점수 (공지사항) → 미래를 아는 변수로 추가
		m.add_future_regressor("GPT_Score")

		# 학습
		m.fit(
			df_np,
			freq="30min",
			num_workers=0,
			checkpointing=False,
			# epochs, batch_size 등은 나중에 필요하면 파라미터로 뺄 수 있음
		)

		self.model = m

		# -------------------------------------------------
		# 간단한 backtest: 마지막 horizon 구간을 "테스트"로 두고
		# 그에 대한 예측 yhat을 뽑아서 RMSE 정도만 계산
		# (엄청 정교한 건 아니고, UI용 대략 지표 수준)
		# -------------------------------------------------
		try:
			from sklearn.metrics import mean_squared_error, r2_score

			# historic prediction 전부 계산
			future_hist = m.make_future_dataframe(
				df_np,
				periods=0,
				n_historic_predictions=True,
				regressors_df=df_np[["GPT_Score"]],
			)
			fc_hist = m.predict(future_hist)

			# 마지막 horizon 구간만 테스트로 사용
			# y: 실제값, yhat1: 1-step-ahead 예측으로 간주
			y_all = fc_hist["y"].values
			yhat1_all = fc_hist["yhat1"].values

			# 뒤쪽 일부만 테스트로 사용 (예: horizon * 2개 구간 정도)
			test_len = min(self.forecast_horizon * 2, len(y_all) // 3)
			if test_len > 10:
				y_test = y_all[-test_len:]
				y_pred = yhat1_all[-test_len:]

				rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
				r2 = float(r2_score(y_test, y_pred))

				self.y_test = pd.Series(y_test)
				self.y_pred = pd.Series(y_pred)
				self.rmse = rmse
				self.r2 = r2

				# split_idx는 "원본 df 기준 테스트 시작 위치" 대략값
				self.split_idx = len(df) - test_len
			else:
				# 데이터가 너무 짧으면 그냥 None
				self.y_test = pd.Series([], dtype=float)
				self.y_pred = pd.Series([], dtype=float)
				self.rmse = None
				self.r2 = None
				self.split_idx = len(df)

		except Exception as e:
			# 평가 실패해도 예측 기능만이라도 동작하게 둔다.
			print(f"[WARN] NeuralProphet backtest metric 실패: {e}")
			self.y_test = pd.Series([], dtype=float)
			self.y_pred = pd.Series([], dtype=float)
			self.rmse = None
			self.r2 = None
			self.split_idx = len(df)

	def predict_test(self):
		"""
		RandomForestPriceModel / LGBMPriceModel / LSTMPriceModel 과
	 인터페이스를 맞추기 위한 테스트 구간 반환.

		NeuralProphet 특성상 backtest가 완벽하진 않지만,
		대략적인 성능 지표(RMSE, R²)와 테스트 구간 y_test / y_pred를 제공.
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
		향후 steps(30분 단위) 시점에 대한 예측선을 DataFrame으로 반환.
		- 팀원 코드의 '대각선 추출' 로직을 streamlit에서 쓰기 좋게 감싼 버전.
		- GPT_Score는 기본적으로 0.0으로 채운 상태에서 예측.
		"""
		if self.model is None or self.df_np is None:
			raise RuntimeError("NeuralProphet 모델이 학습되지 않았습니다. 먼저 train()을 호출하세요.")

		m = self.model
		df_np = self.df_np

		# 미래 GPT 스코어 (공지사항이 없다고 가정 → 0.0)
		future_regressors = pd.DataFrame({
			"GPT_Score": [0.0] * steps
		})

		# 과거 + 미래 dataframe 생성
		future = m.make_future_dataframe(
            df_np,
            periods=steps,
            n_historic_predictions=False,
            regressors_df=future_regressors,
        )

		forecast = m.predict(future)

		# y가 NaN인 행만 미래 구간
		future_rows = forecast[forecast["y"].isnull()].copy()

		# 대각선 추출 (i번째 행 → yhat{i})
		valid_preds = []
		for i in range(1, len(future_rows) + 1):
			row_idx = future_rows.index[i - 1]
			col_name = f"yhat{i}"
			valid_preds.append(future_rows.loc[row_idx, col_name])

		forecast_df = pd.DataFrame({
			"date": future_rows["ds"].values,
			"price": valid_preds,
		})

		return forecast_df
