# models/rf_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .base import BasePriceModel


class RandomForestPriceModel(BasePriceModel):
	def __init__(self):
		self.model = None
		self.df = None
		self.features = None
		self.split_idx = None
		self.y_test = None
		self.y_pred = None
		self.rmse = None
		self.r2 = None

	def train(self, df: pd.DataFrame, features: list[str]):
		self.df = df
		self.features = features

		X = df[features]
		y = df["price"]

		self.split_idx = int(len(df) * 0.8)

		X_train = X.iloc[:self.split_idx]
		y_train = y.iloc[:self.split_idx]
		X_test = X.iloc[self.split_idx:]
		y_test = y.iloc[self.split_idx:]

		self.model = RandomForestRegressor(
			n_estimators=200,
			n_jobs=-1,
			random_state=42,
		)

		self.model.fit(X_train, y_train)

		self.y_test = y_test
		self.y_pred = self.model.predict(X_test)

		self.rmse = np.sqrt(mean_squared_error(y_test, self.y_pred))
		self.r2 = r2_score(y_test, self.y_pred)

	def predict_test(self):
		return (
			self.y_test,
			self.y_pred,
			self.split_idx,
			self.rmse,
			self.r2,
		)

	def predict_future(self, steps: int):
		"""
		단순 반복 예측 (기존 forecast_future 로직 이식)
		"""
		last_row = self.df.iloc[-1].copy()
		future_rows = []

		current_row = last_row.copy()

		for _ in range(steps):
			X_current = current_row[self.features].values.reshape(1, -1)
			pred_price = self.model.predict(X_current)[0]

			current_row["price"] = pred_price
			current_row["date"] = current_row["date"] + pd.Timedelta(minutes=10)

			future_rows.append(
				{
					"date": current_row["date"],
					"price": pred_price,
				}
			)

		return pd.DataFrame(future_rows)
