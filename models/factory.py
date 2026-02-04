# models/factory.py

from .rf_model import RandomForestPriceModel


def get_model(model_name: str):
	if model_name == "rf":
		return RandomForestPriceModel()

	# 추후 확장
	# elif model_name == "lstm":
	#	return LSTMPriceModel()

	# elif model_name == "ensemble":
	#	return EnsemblePriceModel()

	else:
		raise ValueError(f"Unknown model: {model_name}")
