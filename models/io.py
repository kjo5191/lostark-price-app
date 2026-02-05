# models/io.py

import os
from pathlib import Path
from typing import Optional

import joblib

from .factory import get_model


# ---------------------------------------------------------------------
# 1. 모델 저장 기본 경로
#    예) trained_models/rf/item_123.pkl
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트 기준
MODEL_DIR = BASE_DIR / "trained_models"


def _ensure_model_dir():
	if not MODEL_DIR.exists():
		MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _model_filename(model_key: str, item_id: Optional[int] = None) -> Path:
	"""
	모델 키 + 아이템 ID 조합으로 파일 경로 생성.
	item_id가 없으면 공통 모델로 취급.
	"""
	_ensure_model_dir()

	subdir = MODEL_DIR / model_key
	if not subdir.exists():
		subdir.mkdir(parents=True, exist_ok=True)

	if item_id is not None:
		filename = f"{model_key}_item_{item_id}.pkl"
	else:
		filename = f"{model_key}_global.pkl"

	return subdir / filename


# ---------------------------------------------------------------------
# 2. 저장 / 로드 헬퍼
# ---------------------------------------------------------------------
def save_model(model_key: str, item_id: Optional[int], price_model) -> Path:
	"""
	RF / LGBM / XGB / LSTM PriceModel 인스턴스를 그대로 joblib으로 저장.
	NeuralProphet(np)는 매번 다시 학습하도록 디스크에 저장하지 않는다. (저장한거 불러오는게 메모리 에러남)
	"""
	path = _model_filename(model_key, item_id)

	# 🔹 NeuralProphet은 항상 새로 학습 → 저장 스킵
	if model_key == "np":
		# 필요하면 디버깅용 로그만 남겨도 됨
		# print(f"[INFO] NeuralProphet 모델은 디스크에 저장하지 않습니다: {path}")
		return path

	joblib.dump(price_model, path)
	return path


def load_model(model_key: str, item_id: Optional[int]):
	"""
	기존에 저장된 모델을 로드. 없으면 None 반환.
	NeuralProphet(np)는 항상 새로 학습하므로 로드하지 않는다.
	"""
	# 🔹 NeuralProphet은 디스크에서 로드하지 않음 → 항상 None
	if model_key == "np":
		return None

	path = _model_filename(model_key, item_id)
	if not path.exists():
		return None

	price_model = joblib.load(path)
	return price_model



# ---------------------------------------------------------------------
# 3. Streamlit에서 쓸 "load or train" 헬퍼
# ---------------------------------------------------------------------
def load_or_train_model(
	model_key: str,
	item_id: Optional[int],
	df_ml,
	features,
	force_retrain: bool = False,
):
	# 1) 기존 모델이 있으면 우선 로드
	if not force_retrain:
		existing = load_model(model_key, item_id)
		if existing is not None:
			# 🔹 공통: 최신 데이터 프레임 / 피처 연결
			if hasattr(existing, "df"):
				existing.df = df_ml
			if hasattr(existing, "features"):
				existing.features = features

			# 🔹 NeuralProphet 전용: df_np / backtest 갱신
			#    - _build_np_df, _compute_backtest_metrics 는 우리가 앞에서 구현한 메서드
			if hasattr(existing, "_build_np_df"):
				try:
					existing.df_np = existing._build_np_df(df_ml)

					# split 기준도 새 길이에 맞춰서 다시 설정
					if hasattr(existing, "_compute_backtest_metrics"):
						n_np = len(existing.df_np)
						existing.split_idx = int(n_np * 0.8)
						existing._compute_backtest_metrics()
				except Exception as e:
					print(f"[WARN] NeuralProphet df_np / backtest 갱신 실패: {e}")

			return existing, "loaded"

	# 2) 기존 모델이 없거나 강제 재학습이면 새로 학습
	price_model = get_model(model_key)
	price_model.train(df_ml, features)

	try:
		save_model(model_key, item_id, price_model)
	except Exception as e:
		print(f"[WARN] 모델 저장 실패: {e}")

	return price_model, "trained"

