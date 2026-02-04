# models/base.py

class BasePriceModel:
	def train(self, df, features):
		"""
		모델 학습
		"""
		raise NotImplementedError

	def predict_test(self):
		"""
		검증 구간 예측 결과 반환

		return:
			y_test, y_pred, split_idx, rmse, r2
		"""
		raise NotImplementedError

	def predict_future(self, steps: int):
		"""
		미래 steps 만큼 예측

		return:
			future_df (date, price)
		"""
		raise NotImplementedError
