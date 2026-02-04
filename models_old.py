# models.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_random_forest(df_ml: pd.DataFrame, features: list[str]):
	X = df_ml[features]
	y = df_ml["price"]

	split_idx = int(len(df_ml) * 0.8)

	X_train = X.iloc[:split_idx]
	y_train = y.iloc[:split_idx]
	X_test = X.iloc[split_idx:]
	y_test = y.iloc[split_idx:]

	model = RandomForestRegressor(
		n_estimators=200,
		n_jobs=-1,
		random_state=42
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)

	return model, y_test, y_pred, split_idx, rmse, r2


def forecast_future(
	model: RandomForestRegressor,
	df_ml: pd.DataFrame,
	features: list[str],
	steps: int,
	freq: str = "10T",  # 10분 간격
) -> pd.DataFrame:
	"""
	간단한 auto-regressive 방식의 미래 예측.
	- df_ml: 과거 ML용 데이터 (date, price, feature 포함)
	- features: 모델이 사용하는 feature 컬럼 리스트
	- steps: 앞으로 예측할 스텝 수 (10분 단위 기준 1일=144)
	- freq: 시간 간격 (기본 10분)
	"""
	# 히스토리 복사 (index 정리)
	history = df_ml.copy().reset_index(drop=True)

	future_rows: list[dict] = []

	for _ in range(steps):
		last_row = history.iloc[-1].copy()

		# 다음 시점 시간 계산
		next_time = last_row["date"] + pd.Timedelta(freq)

		# ---- 새 row 베이스 만들기 ----
		new_row = last_row.copy()
		new_row["date"] = next_time

		# 1) 시간 관련 피처 갱신
		if "hour" in new_row.index:
			new_row["hour"] = next_time.hour
		if "day_of_week" in new_row.index:
			new_row["day_of_week"] = next_time.dayofweek

		# 2) lag 피처 갱신 (있을 때만)
		if "lag_10m" in new_row.index:
			new_row["lag_10m"] = history["price"].iloc[-1]
		if "lag_1h" in new_row.index:
			if len(history) >= 6:
				new_row["lag_1h"] = history["price"].iloc[-6]
			else:
				new_row["lag_1h"] = history["price"].iloc[0]
		if "lag_24h" in new_row.index:
			if len(history) >= 144:
				new_row["lag_24h"] = history["price"].iloc[-144]
			else:
				new_row["lag_24h"] = history["price"].iloc[0]

		# 3) 그 외 feature들은 일단 직전 값 유지 (RSI, Bollinger 등)
		#    -> 나중에 필요하면 features.py 로직 재사용해서 정교하게 계산할 수 있음.

		# 4) 모델 입력용 X_row 구성
		X_row = pd.DataFrame([new_row[features]])

		# 5) 예측
		y_hat = float(model.predict(X_row)[0])

		# 6) price 갱신 및 기록
		new_row["price"] = y_hat

		future_rows.append({
			"date": next_time,
			"price": y_hat,
		})

		# 히스토리에 새 row 추가 (다음 스텝을 위해)
		history = pd.concat(
			[history, pd.DataFrame([new_row])],
			ignore_index=True
		)

	future_df = pd.DataFrame(future_rows)
	return future_df