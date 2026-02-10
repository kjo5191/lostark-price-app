# 시세_예측_대시보드.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import os
import shutil

from data_loader import load_merged_data, load_gpt_scores
from features import filter_item, make_ml_dataset
from models.io import load_or_train_model
from backtest import simulate_strict_investor
from preprocess import apply_gpt_scores, clean_outliers_rolling, resample_to_30min_for_app


# -------------------------------------------------------------------------
# 1. 시간 해상도 / 예측 구간 설정 (30분 단위 기준)
# -------------------------------------------------------------------------
TIME_STEP_MINUTES = 30
POINTS_PER_DAY = int(24 * 60 / TIME_STEP_MINUTES)  # 48
FORECAST_DAYS = 3
FORECAST_STEPS = FORECAST_DAYS * POINTS_PER_DAY    # 144


# -------------------------------------------------------------------------
# 2. 페이지 설정 & 세션 초기화
# -------------------------------------------------------------------------
st.set_page_config(
	page_title="디지털 자산 시세 변동 예측 모델",
	layout="wide"
)

if "rf_result" not in st.session_state:
	st.session_state.rf_result = None

st.title("디지털 자산 시세 변동 예측 모델")
st.caption("로스트아크 거래소 아이템 시세를 앙상블 모델(LightGBM / XGBoost / NeuralProphet)로 예측합니다.")


# -------------------------------------------------------------------------
# 3. 사이드바 - 검색 / 학습 범위 설정 + 검증 모델 선택
# -------------------------------------------------------------------------
with st.sidebar:
	st.header("아이템 검색")

	# 1) 데이터 로딩
	df_final = load_merged_data()
	df_gpt_all = load_gpt_scores()

	grade_list = sorted(df_final["grade"].dropna().unique())
	grade_options = ["전체"] + grade_list

	# 2) 등급 선택
	target_grade = st.selectbox(
		"아이템 등급",
		grade_options,
		index=grade_options.index("유물") if "유물" in grade_options else 0,
	)

	# 3) 등급에 따라 아이템 후보 리스트 동적 생성
	if target_grade == "전체":
		item_options = sorted(df_final["name"].dropna().unique())
	else:
		item_options = sorted(
			df_final.loc[df_final["grade"] == target_grade, "name"]
			.dropna()
			.unique()
		)

	if len(item_options) == 0:
		st.warning("선택한 등급에 해당하는 아이템이 없습니다.")
		st.stop()

	default_item_name = "유물 원한 각인서"

	default_index = 0
	if default_item_name in item_options:
		default_index = item_options.index(default_item_name)


	# 4) 아이템 이름 선택 (타이핑하면 자동 필터링됨)
	target_item_name = st.selectbox(
		"아이템 이름",
		item_options,
		index=default_index,
		help="아이템 이름의 일부를 타이핑하면 자동으로 필터링됩니다.",
	)


	# filter_item() 에서 쓰던 변수명 유지
	target_keyword = target_item_name

	# 5) 그래프 표시 기간
	days_to_show = st.slider(
		"그래프 표시 기간 (일)",
		min_value=1,
		max_value=14,
		value=3,
		step=1,
	)
	zoom_n = days_to_show * POINTS_PER_DAY

	# 🔹 Y축 범위를 전체 기간 기준으로 고정할지 여부
	use_global_scale = st.checkbox(
		"Y축 범위를 전체 기간으로 고정",
		value=False,
	)	

	# 6) 실행 버튼 (폼 대신 일반 버튼)
	# run_button = st.button("학습 & 예측 실행")
	run_button = st.button("AI 예측 시작", type="primary", use_container_width=True)

	zoom_n = days_to_show * POINTS_PER_DAY

	# 3-3. 검증용 단일 모델 선택 (예측은 항상 앙상블 모델)
	st.markdown("---")
	st.subheader("🔍 검증 모델 선택")

	eval_model_key = st.selectbox(
		"검증에 사용할 단일 모델",
		["lgbm", "xgb", "rf", "lstm", "np"],
		format_func=lambda k: {
			"lgbm": "LightGBM",
			"xgb": "XGBoost",
			"rf": "RandomForest",
			"lstm": "LSTM",
			"np": "NeuralProphet",
		}[k],
	)

	# 🔹 관리자 설정 영역 추가
	st.markdown("---")
	with st.expander("⚙️ 관리자 설정"):
		st.caption("저장된 모델 학습 결과(.pkl)와 예측 캐시를 초기화합니다.")

		if st.button("모델 초기화", type="secondary", use_container_width=True):
			model_dir = "trained_models"  # 🔥 반드시 여기

			try:
				# 1) 학습된 모델 파일만 삭제
				if os.path.exists(model_dir):
					shutil.rmtree(model_dir)

				# 2) 빈 폴더 재생성
				os.makedirs(model_dir, exist_ok=True)

				# 3) 세션 캐시 초기화
				st.session_state.rf_result = None

				st.success(
					"학습된 모델(.pkl)과 세션 캐시를 초기화했습니다.\n"
					"다시 [학습 & 예측 실행]을 눌러 모델을 재학습해주세요."
				)
			except Exception as e:
				st.error(f"모델 초기화 중 오류가 발생했습니다: {e}")


# -------------------------------------------------------------------------
# 4. 버튼 눌렀을 때만 새로 계산 → 전처리 + Feature Engineering
# -------------------------------------------------------------------------
if run_button:
	with st.spinner("데이터 필터링 중..."):
		result = filter_item(df_final, target_keyword, target_grade)

	if result is None:
		st.error(f"'{target_keyword}' (등급: {target_grade}) 에 해당하는 데이터가 없습니다.")
	else:
		# 🔹 UI용 원본 (10분 단위)
		df_target, top_item = result

		# 4-1. 30분봉으로 변환 (ML 전용)
		df_target_30 = resample_to_30min_for_app(df_target)

		# 4-2. item_id 추출
		item_id = None
		if "item_id" in df_target.columns:
			try:
				item_id = int(df_target["item_id"].iloc[0])
			except Exception:
				item_id = None

		# 4-3. 해당 아이템에 대한 GPT 점수만 필터링
		if item_id is not None:
			df_gpt_item = df_gpt_all[df_gpt_all["item_id"] == item_id].copy()
		else:
			df_gpt_item = None

		# 4-4. GPT 점수 매핑 (date index 기준)
		df_target_for_ml = (
			df_target_30
			.sort_values("date")
			.set_index("date")
		)

		df_target_with_gpt = apply_gpt_scores(
			df_target_for_ml,
			df_gpt_item,
			score_col="gpt_score",
		)

		# 4-5. 이상치 정제 (30분 기준)
		df_target_clean = clean_outliers_rolling(
			df_target_with_gpt,
			column="price",
			window=POINTS_PER_DAY,   # 하루 기준
			sigma=3.0,
		)

		df_target_clean = df_target_clean.reset_index()

		# 4-6. Feature Engineering
		with st.spinner("Feature Engineering 처리 중..."):
			df_ml, features = make_ml_dataset(df_target_clean)

		if len(df_ml) < 300:
			st.warning(
				f"Feature 생성 후 데이터가 {len(df_ml)}개입니다. "
				"(최소 300개 이상일 때가 더 안정적)"
			)
		else:
			# -----------------------------------------------------------------
			# 5. 앙상블 모델 (LightGBM / XGBoost / NeuralProphet)
			#    - 항상 학습/로드 후 미래 예측
			#    - 예측값은 날짜 기준으로 merge 후 가중 평균
			# -----------------------------------------------------------------
			with st.spinner("앙상블 모델 학습 / 로드 중..."):
				ensemble_keys = ["lgbm", "xgb", "np"]
				ensemble_weights = {
					"lgbm": 4.0,
					"xgb": 4.5,
					"np": 1.5,
				}

				ensemble_models: dict[str, object] = {}
				ensemble_status: dict[str, str] = {}
				ensemble_future: dict[str, pd.DataFrame | None] = {}

				for key in ensemble_keys:
					m, status = load_or_train_model(
						model_key=key,
						item_id=item_id,
						df_ml=df_ml,
						features=features,
					)
					ensemble_models[key] = m
					ensemble_status[key] = status

					try:
						fut = m.predict_future(steps=FORECAST_STEPS)
					except NotImplementedError:
						fut = None

					ensemble_future[key] = fut

				# 5-2. 앙상블 모델 미래 예측 (날짜 기준 merge + 가중 평균)
				valid_keys = [
					k for k in ensemble_keys
					if ensemble_future.get(k) is not None
					and not ensemble_future[k].empty
				]

				if len(valid_keys) == 0:
					ensemble_future_df = None
				else:
					df_ens = None
					for k in valid_keys:
						df_k = ensemble_future[k][["date", "price"]].copy()
						df_k = df_k.rename(columns={"price": f"price_{k}"})
						if df_ens is None:
							df_ens = df_k
						else:
							df_ens = pd.merge(df_ens, df_k, on="date", how="inner")

					# 사용 가능한 모델만으로 가중 평균 계산
					total_w = sum(ensemble_weights[k] for k in valid_keys)
					weighted_sum = 0.0
					for k in valid_keys:
						w = ensemble_weights[k]
						weighted_sum += df_ens[f"price_{k}"] * w

					df_ens["ensemble_price"] = weighted_sum / total_w
					ensemble_future_df = df_ens  # date + price_lgbm/xgb/np + ensemble_price

			# -----------------------------------------------------------------
			# 6. 검증용 단일 모델 학습 / 평가
			#    - eval_model_key 기준
			#    - lgbm/xgb는 이미 ensemble에서 학습된 인스턴스 재사용
			# -----------------------------------------------------------------
			with st.spinner("선택한 검증 모델 학습 / 평가 중..."):
				if eval_model_key in ensemble_models:
					eval_model = ensemble_models[eval_model_key]
					eval_status = ensemble_status[eval_model_key]
				else:
					eval_model, eval_status = load_or_train_model(
						model_key=eval_model_key,
						item_id=item_id,
						df_ml=df_ml,
						features=features,
					)

				eval_model_name = {
					"lgbm": "LightGBM",
					"xgb": "XGBoost",
					"rf": "RandomForest",
					"lstm": "LSTM",
					"np": "NeuralProphet",
				}[eval_model_key]

				if eval_status == "loaded":
					st.info(f"📦 검증 모델({eval_model_name})을 저장된 상태에서 불러왔습니다.")
				else:
					st.success(f"🧠 검증 모델({eval_model_name})을 새로 학습하고 저장했습니다.")

				y_test, y_pred, split_idx, rmse, r2 = eval_model.predict_test()

			# -----------------------------------------------------------------
			# 7. 세션에 결과 저장 (앙상블 모델 + 검증 모델)
			# -----------------------------------------------------------------
			st.session_state.rf_result = {
				"df_target": df_target,          # UI용 (10분)
				"df_ml": df_ml,                  # ML용 (30분, gpt_score 포함)
				"top_item": top_item,
				"y_test": y_test,
				"y_pred": y_pred,
				"split_idx": split_idx,
				"rmse": rmse,
				"r2": r2,
				# "days_to_show": days_to_show,
				"future_df_ensemble": ensemble_future_df,  # 🔥 앙상블 모델 예측 + 개별
				"eval_model_key": eval_model_key,
				"eval_model_name": eval_model_name,
				"features": features,
				# "use_global_scale": use_global_scale,
			}


# -------------------------------------------------------------------------
# 8. 세션에 결과 없으면 안내 후 종료
# -------------------------------------------------------------------------
if st.session_state.rf_result is None:
	st.info("아이템 등급, 이름 설정 후 **[AI 예측 시작]** 버튼을 눌러주세요.")
	st.stop()


# -------------------------------------------------------------------------
# 9. 세션에서 결과 꺼내서 공통 변수 준비
# -------------------------------------------------------------------------
res = st.session_state.rf_result

df_target = res["df_target"]
df_ml = res["df_ml"]
top_item = res["top_item"]
y_test = res["y_test"]
y_pred = res["y_pred"]
split_idx = res["split_idx"]
rmse = res["rmse"]
r2 = res["r2"]
# days_to_show = res["days_to_show"]
future_df_ensemble = res["future_df_ensemble"]
eval_model_key = res["eval_model_key"]
eval_model_name = res["eval_model_name"]
# use_global_scale = res.get("use_global_scale", False)
zoom_n = days_to_show * POINTS_PER_DAY


st.subheader(f"🎯 분석 대상: {top_item}")


# -------------------------------------------------------------------------
# 10. 현재 가격 & 전일 평균 가격
# -------------------------------------------------------------------------
latest_ts = df_target["date"].max()
latest_row = df_target.loc[df_target["date"] == latest_ts].iloc[-1]
current_price = float(latest_row["price"])

current_day_start = latest_ts.normalize()  # 당일 00:00
prev_day_start = current_day_start - pd.Timedelta(days=1)
prev_day_end = current_day_start          # 전날 23:59:59까지

mask_prev = (df_target["date"] >= prev_day_start) & (df_target["date"] < prev_day_end)
df_prev = df_target.loc[mask_prev]

if not df_prev.empty:
	yesterday_avg_price = float(df_prev["price"].mean())
	yesterday_text = f"{yesterday_avg_price:,.0f} G"
else:
	yesterday_avg_price = None
	yesterday_text = "데이터 없음"

price_col1, price_col2 = st.columns(2)
with price_col1:
	st.metric("현재 가격", f"{current_price:,.0f} G")
with price_col2:
	st.metric("전일 평균 가격", yesterday_text)


# -------------------------------------------------------------------------
# 11. 앙상블 모델 기반 메인 예측 그래프 (히스토리 + 미래)
# -------------------------------------------------------------------------
st.markdown("### 🔮 앙상블 모델 기반 향후 3일 시세 예측")

st.caption(
	"앙상블 모델 (LightGBM 5.5 : XGBoost 3.5 : NeuralProphet 1.0 가중 평균)\n"
	"점선은 각 개별 모델의 예측, 실선은 앙상블 모델과 실제 히스토리입니다."
)

if future_df_ensemble is None or future_df_ensemble.empty:
	st.info("앙상블 모델 예측을 생성할 수 없습니다. (필요 모델의 predict_future 미구현 또는 데이터 부족)")
else:
	if zoom_n > len(df_ml):
		zoom_n = len(df_ml)

	# 11-1. 최근 히스토리 구간
	hist_tail = df_ml[["date", "price"]].iloc[-zoom_n:].copy()
	hist_tail["type"] = "History"

	# 11-2. 미래 예측: 앙상블 + 개별 3개
	df_ens_raw = future_df_ensemble.copy()

	# 앙상블 메인 라인
	main_future = pd.DataFrame({
		"date": df_ens_raw["date"],
		"price": df_ens_raw["ensemble_price"],
		"type": "Ensemble Forecast",
	})

	# 개별 모델들 (점선 + 투명)
	indiv_frames = []

	if "price_lgbm" in df_ens_raw.columns:
		indiv_frames.append(pd.DataFrame({
			"date": df_ens_raw["date"],
			"price": df_ens_raw["price_lgbm"],
			"type": "LightGBM",
		}))

	if "price_xgb" in df_ens_raw.columns:
		indiv_frames.append(pd.DataFrame({
			"date": df_ens_raw["date"],
			"price": df_ens_raw["price_xgb"],
			"type": "XGBoost",
		}))

	if "price_np" in df_ens_raw.columns:
		indiv_frames.append(pd.DataFrame({
			"date": df_ens_raw["date"],
			"price": df_ens_raw["price_np"],
			"type": "NeuralProphet",
		}))

	if len(indiv_frames) > 0:
		df_indiv = pd.concat(indiv_frames, ignore_index=True)
	else:
		df_indiv = pd.DataFrame(columns=["date", "price", "type"])

	# 🔹 수요일 06:00 세로선용 데이터 생성
	# "최근 zoom_n 히스토리 구간 + 미래 예측" 범위에 대해서만 생성
	date_start = hist_tail["date"].min()          # ✅ 최근 구간 시작 시점
	date_end = df_ens_raw["date"].max()           # ✅ 예측 마지막 시점

	wednesday_6am = (
		pd.date_range(
			start=date_start.normalize(),
			end=date_end.normalize(),
			freq="W-WED",
		)
		+ pd.Timedelta(hours=6)
	)

	df_wed = pd.DataFrame({"date": wednesday_6am})

	# 11-3. y축 범위 계산 (히스토리 + 앙상블 + 개별 모두 포함)
	df_main = pd.concat([hist_tail, main_future], ignore_index=True)

	if use_global_scale:
		# 🔹 전체 기간 가격 + 미래 예측까지 포함해서 Y축 범위 계산
		df_all_hist = df_ml[["date", "price"]].copy()
		df_all_hist["type"] = "History (전체)"

		df_for_range = pd.concat(
			[df_all_hist, main_future, df_indiv],
			ignore_index=True
		)
	else:
		# 기존처럼: 최근 구간(hist_tail) + 앙상블 + 개별 모델 기준
		df_for_range = pd.concat([df_main, df_indiv], ignore_index=True)

	y_min_f = df_for_range["price"].min()
	y_max_f = df_for_range["price"].max()
	padding_f = (y_max_f - y_min_f) * 0.05
	y_domain_f = [y_min_f - padding_f, y_max_f + padding_f]

	# 11-4. Altair 레이어 구성 (히스토리 + 앙상블 + 개별 + 수요일 06:00 점선)
	base_chart = (
		alt.Chart(df_main)
		.mark_line()
		.encode(
			x=alt.X("date:T", title="시간"),
			y=alt.Y(
				"price:Q",
				title="가격 (Gold)",
				scale=alt.Scale(domain=y_domain_f)
			),
			color=alt.Color("type:N", title="구분"),
			tooltip=[
				alt.Tooltip("date:T", title="시간"),
				alt.Tooltip("type:N", title="구분"),
				alt.Tooltip("price:Q", title="가격"),
			],
		)
	)

	indiv_chart = (
		alt.Chart(df_indiv)
		.mark_line(strokeDash=[4, 4], opacity=0.35)  # 점선 + 반투명
		.encode(
			x=alt.X("date:T", title="시간"),
			y=alt.Y(
				"price:Q",
				title="가격 (Gold)",
				scale=alt.Scale(domain=y_domain_f)
			),
			color=alt.Color("type:N", title="모델"),
			tooltip=[
				alt.Tooltip("date:T", title="시간"),
				alt.Tooltip("type:N", title="모델"),
				alt.Tooltip("price:Q", title="가격"),
			],
		)
	)

	# 🔹 수요일 06:00 세로 점선 레이어
	wed_rule = (
		alt.Chart(df_wed)
		.mark_rule(
			color="orange",
			strokeDash=[6, 6],
			opacity=0.6,
		)
		.encode(
			x="date:T"
		)
	)

	chart_future = (
		(base_chart + indiv_chart + wed_rule)
		.properties(
			title=f"[{top_item}] 최근 {days_to_show}일 + 앙상블 모델 기반 향후 {FORECAST_DAYS}일 시세 예측",
		)
		.interactive()
	)

	st.altair_chart(chart_future, use_container_width=True)



# -------------------------------------------------------------------------
# 12. 검증 결과 (RMSE / R² + 최근 테스트 구간 그래프) - expander
# -------------------------------------------------------------------------
with st.expander("📊 검증 모델 성능 및 최근 테스트 구간 보기", expanded=False):
	st.markdown(f"#### 검증 모델: {eval_model_name}")

	col1, col2 = st.columns(2)
	with col1:
		st.metric("RMSE (골드)", f"{rmse:,.2f}")
	with col2:
		st.metric("R²", f"{r2:.3f}")

	st.markdown("##### 📈 최근 테스트 구간 확대 그래프 (인터랙티브)")

	test_len = len(y_test)

	test_dates = (
		df_ml["date"]
		.iloc[-test_len:]
		.reset_index(drop=True)
		.to_numpy()
	)

	actual = (
		pd.Series(y_test)
		.reset_index(drop=True)
		.to_numpy()
	)

	pred = (
		pd.Series(y_pred)
		.reset_index(drop=True)
		.to_numpy()
	)

	if zoom_n > len(test_dates):
		zoom_n_local = len(test_dates)
	else:
		zoom_n_local = zoom_n

	zoom_slice = slice(-zoom_n_local, None)

	df_plot = pd.DataFrame({
		"date": test_dates[zoom_slice],
		"Actual (실제)": actual[zoom_slice],
		"Prediction (예측)": pred[zoom_slice],
	})

	df_plot_melt = df_plot.melt("date", var_name="type", value_name="price")

	y_min = df_plot_melt["price"].min()
	y_max = df_plot_melt["price"].max()
	padding = (y_max - y_min) * 0.05
	y_domain = [y_min - padding, y_max + padding]

	chart = (
		alt.Chart(df_plot_melt)
		.mark_line()
		.encode(
			x=alt.X("date:T", title="시간"),
			y=alt.Y(
				"price:Q",
				title="가격 (Gold)",
				scale=alt.Scale(domain=y_domain)
			),
			color=alt.Color("type:N", title="구분"),
			tooltip=[
				alt.Tooltip("date:T", title="시간"),
				alt.Tooltip("type:N", title="구분"),
				alt.Tooltip("price:Q", title="가격"),
			],
		)
		.properties(
			title=f"[{top_item}] 검증 모델({eval_model_name}) 기준 최근 {days_to_show}일 테스트 구간 예측",
		)
		.interactive()
	)

	st.altair_chart(chart, use_container_width=True)


# -------------------------------------------------------------------------
# 13. 전체 시세 & 수요일(Reset) 하이라이트 - expander
# -------------------------------------------------------------------------
with st.expander("📉 전체 시세 흐름 & 수요일(Reset) 영향 분석", expanded=False):
	st.markdown("#### 전체 시세 + 검증 구간 + 예측 구간 표시")

	all_dates = df_ml["date"].reset_index(drop=True).to_numpy()
	all_prices = df_ml["price"].reset_index(drop=True).to_numpy()

	df_line_all = pd.DataFrame({
		"date": all_dates,
		"price": all_prices,
		"type": "History (전체 흐름)",
	})

	test_len = len(y_test)
	test_dates_full = all_dates[-test_len:]
	real_test_price = all_prices[-test_len:]
	pred_price = np.asarray(y_pred)

	df_line_test = pd.DataFrame({
		"date": test_dates_full,
		"price": real_test_price,
		"type": "Actual (검증 구간)",
	})

	df_line_pred = pd.DataFrame({
		"date": test_dates_full,
		"price": pred_price,
		"type": "Prediction (검증 구간 예측)",
	})

	df_lines = pd.concat([df_line_all, df_line_test, df_line_pred], ignore_index=True)

	unique_days = pd.to_datetime(df_ml["date"]).dt.normalize().drop_duplicates()
	weds = unique_days[unique_days.dt.dayofweek == 2]

	df_weds = pd.DataFrame({
		"start": weds,
		"end": weds + pd.Timedelta(days=1),
		"label": "수요일 (Reset)",
	})

	split_idx_all = len(all_dates) - test_len
	split_time = all_dates[split_idx_all]
	df_split = pd.DataFrame({"date": [split_time]})

	y_all_min = all_prices.min()
	y_all_max = all_prices.max()
	padding_all = (y_all_max - y_all_min) * 0.05
	y_domain_all = [y_all_min - padding_all, y_all_max + padding_all]

	rect = (
		alt.Chart(df_weds)
		.mark_rect()
		.encode(
			x="start:T",
			x2="end:T",
			color=alt.value("orange"),
			opacity=alt.value(0.12),
		)
	)

	lines = (
		alt.Chart(df_lines)
		.mark_line()
		.encode(
			x=alt.X("date:T", title="날짜"),
			y=alt.Y("price:Q", title="가격 (Gold)", scale=alt.Scale(domain=y_domain_all)),
			color=alt.Color("type:N", title="구분"),
			tooltip=[
				alt.Tooltip("date:T", title="날짜"),
				alt.Tooltip("type:N", title="구분"),
				alt.Tooltip("price:Q", title="가격"),
			],
		)
	)

	rule = (
		alt.Chart(df_split)
		.mark_rule(color="green", strokeDash=[4, 4])
		.encode(
			x="date:T",
			size=alt.value(2),
		)
	)

	chart_all = (
		(rect + lines + rule)
		.properties(
			title=f"[{top_item}] 전체 시세 & 수요일(Reset) 영향 분석 (검증 모델: {eval_model_name})",
			height=400,
		)
		.interactive()
	)

	st.altair_chart(chart_all, use_container_width=True)


# -------------------------------------------------------------------------
# 14. 원시 데이터 / Feature 데이터 확인 - expander
# -------------------------------------------------------------------------
with st.expander("📂 원시 데이터 / Feature 데이터 확인"):
	st.markdown("#### 🔹 원본 타겟 데이터 (df_target)")
	st.dataframe(df_target[["date", "name", "grade", "price"]].tail(50))

	st.markdown("#### 🔹 ML 학습용 데이터 (df_ml)")
	base_cols = ["date", "price", "lag_30m", "rsi", "is_overbought", "is_oversold"]
	if "gpt_score" in df_ml.columns:
		base_cols.append("gpt_score")

	st.dataframe(df_ml[base_cols].tail(50))


# -------------------------------------------------------------------------
# 15. 투자 시뮬레이션 페이지 링크
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown("")
st.markdown("### 💼 투자 시뮬레이션")

st.caption(
	"현재 분석한 아이템과 동일한 데이터로 백테스트를 돌려보고 싶다면, "
	"아래 버튼을 눌러 투자 시뮬레이션 페이지로 이동하세요."
)

st.page_link(
	"pages/투자_시뮬레이션.py",
	label="투자 시뮬레이션 페이지 열기",
	icon="➡️",
)
