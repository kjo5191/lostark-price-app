# 시세_예측_대시보드.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt

from data_loader import load_merged_data, load_gpt_scores
from features import filter_item, make_ml_dataset
from models.factory import get_model
from models.io import load_or_train_model
from backtest import simulate_strict_investor
from preprocess import apply_gpt_scores, clean_outliers_rolling, resample_to_30min_for_app


# -------------------------------------------------------------------------
# 시간 해상도 설정 (30분 단위 기준)
# -------------------------------------------------------------------------
TIME_STEP_MINUTES = 30
POINTS_PER_DAY = int(24 * 60 / TIME_STEP_MINUTES)  # 48
FORECAST_DAYS = 3
FORECAST_STEPS = FORECAST_DAYS * POINTS_PER_DAY    # 144

# -------------------------------------------------------------------------
# 0. 페이지 설정 & 세션 초기화
# -------------------------------------------------------------------------
st.set_page_config(
	page_title="디지털 자산 시세 변동 예측 모델",
	layout="wide"
)

if "rf_result" not in st.session_state:
	st.session_state.rf_result = None

st.title("디지털 자산 시세 변동 예측 모델")
st.caption("로스트아크 거래소 아이템 시세를 여러 모델(RandomForest / LightGBM / LSTM / NeuralProphet)로 예측합니다.")

# -------------------------------------------------------------------------
# 1. 사이드바 - 검색/학습 설정 (폼 + Enter 제출)
# -------------------------------------------------------------------------
with st.sidebar:
	st.header("검색 / 학습 설정")

	df_final = load_merged_data()
	df_gpt_all = load_gpt_scores()

	grade_list = sorted(df_final["grade"].dropna().unique())
	grade_options = ["전체"] + grade_list

	with st.form("search_form"):
		target_grade = st.selectbox(
			"아이템 등급",
			grade_options,
			index=grade_options.index("유물") if "유물" in grade_options else 0
		)

		target_keyword = st.text_input(
			"아이템 이름 키워드",
			value="원한"
		)
		
		days_to_show = st.slider(
			"최근 예측 기간 (일)",
			min_value=1,
			max_value=14,
			value=3,
			step=1
		)

		zoom_n = days_to_show * POINTS_PER_DAY

		model_key = st.selectbox(
			"모델 선택",
			["rf", "lgbm", "lstm", "np"],
			format_func=lambda k: {
				"rf": "RandomForest",
				"lgbm": "LightGBM",
				"lstm": "LSTM",
				"np": "NeuralProphet",
			}[k],
		)
		
		run_button = st.form_submit_button("학습 & 예측 실행")


# -------------------------------------------------------------------------
# 2. 버튼 눌렀을 때만 새로 계산 → 세션에 저장
# -------------------------------------------------------------------------
if run_button:
	with st.spinner("데이터 필터링 중..."):
		result = filter_item(df_final, target_keyword, target_grade)

	if result is None:
		st.error(f"'{target_keyword}' (등급: {target_grade}) 에 해당하는 데이터가 없습니다.")
	else:
		# 🔹 UI용 원본 (10분)
		df_target, top_item = result

		# 🔥 1) 30분봉으로 변환 (ML 전용)
		df_target_30 = resample_to_30min_for_app(df_target)

		# 🔹 item_id 추출
		item_id = None
		if "item_id" in df_target.columns:
			try:
				item_id = int(df_target["item_id"].iloc[0])
			except Exception:
				item_id = None

		# 🔹 해당 아이템에 대한 GPT 점수만 필터링
		if item_id is not None:
			df_gpt_item = df_gpt_all[df_gpt_all["item_id"] == item_id].copy()
		else:
			df_gpt_item = None

		# 🔹 2) GPT 점수 매핑 (date index 기준)
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

		# 🔹 3) 이상치 정제 (30분 기준)
		df_target_clean = clean_outliers_rolling(
			df_target_with_gpt,
			column="price",
			window=POINTS_PER_DAY,   # 하루 기준
			sigma=3.0,
		)

		df_target_clean = df_target_clean.reset_index()

		# 🔹 4) Feature Engineering
		with st.spinner("Feature Engineering 처리 중..."):
			df_ml, features = make_ml_dataset(df_target_clean)

		if len(df_ml) < 300:
			st.warning(
				f"Feature 생성 후 데이터가 {len(df_ml)}개입니다. "
				"(최소 300개 이상일 때가 더 안정적)"
			)
		else:
			with st.spinner("학습 및 예측 중..."):
				# price_model = get_model(model_key)
				# price_model.train(df_ml, features)
				price_model, model_status = load_or_train_model(
					model_key=model_key,
					item_id=item_id,
					df_ml=df_ml,
					features=features,
				)				

				if model_status == "loaded":
					st.info("📦 저장된 모델을 불러와 예측했습니다.")
				else:
					st.success("🧠 새로운 모델을 학습하고 저장했습니다.")

				y_test, y_pred, split_idx, rmse, r2 = price_model.predict_test()
				try:
					future_df = price_model.predict_future(steps=FORECAST_STEPS)
				except NotImplementedError:
					future_df = None

			st.session_state.rf_result = {
				"df_target": df_target,     # UI용 (10분)
				"df_ml": df_ml,             # ML용 (30분, gpt_score 포함 가능)
				"top_item": top_item,
				"y_test": y_test,
				"y_pred": y_pred,
				"split_idx": split_idx,
				"rmse": rmse,
				"r2": r2,
				"days_to_show": days_to_show,
				"future_df": future_df,
				"features": features,
			}


# -------------------------------------------------------------------------
# 3. 세션에 결과 없으면 안내 후 종료
# -------------------------------------------------------------------------
if st.session_state.rf_result is None:
	st.info("왼쪽에서 등급/키워드 설정 후 **[학습 & 예측 실행]** 버튼 또는 Enter 를 눌러줘.")
	st.stop()

# -------------------------------------------------------------------------
# 4. 세션에서 결과 꺼내서 화면에 표시
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
days_to_show = res["days_to_show"]
future_df = res["future_df"]
zoom_n = days_to_show * POINTS_PER_DAY

st.subheader(f"🎯 분석 대상: {top_item}")

# -----------------------------
# 현재 가격 & 전일 평균 가격
# -----------------------------
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

# -----------------------------
# 모델 성능 지표
# -----------------------------
col1, col2 = st.columns(2)
with col1:
	st.metric("RMSE (골드)", f"{rmse:,.2f}")
with col2:
	st.metric("R²", f"{r2:.3f}")


# -----------------------------------------------------------------
# 투자 시뮬레이션 페이지로 이동 링크
# -----------------------------------------------------------------
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


# -------------------------------------------------------------------------
# 5. 시각화 1: 테스트 구간 확대
# -------------------------------------------------------------------------
st.markdown("### 📈 최근 테스트 구간 확대 그래프 (인터랙티브)")

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
	zoom_n = len(test_dates)

zoom_slice = slice(-zoom_n, None)

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
		title=f"[{top_item}] 최근 {days_to_show}일 시세 예측"
	)
	.interactive()
)

st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------------------------------
# 6. 시각화 2: 전체 + 수요일 하이라이트
# -------------------------------------------------------------------------
st.markdown("### 📊 전체 시세 & 수요일(Reset) 하이라이트 (인터랙티브)")

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
	"type": "Prediction (예측)",
})

df_lines = pd.concat([df_line_all, df_line_test, df_line_pred], ignore_index=True)

unique_days = pd.to_datetime(df_ml["date"]).dt.normalize().drop_duplicates()
weds = unique_days[unique_days.dt.dayofweek == 2]

df_weds = pd.DataFrame({
	"start": weds,
	"end": weds + pd.Timedelta(days=1),
	"label": "수요일 (Reset)",
})

split_idx = len(all_dates) - test_len
split_time = all_dates[split_idx]
df_split = pd.DataFrame({"date": [split_time]})

y_all_min = all_prices.min()
y_all_max = all_prices.max()
padding = (y_all_max - y_all_min) * 0.05
y_domain = [y_all_min - padding, y_all_max + padding]

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
		y=alt.Y("price:Q", title="가격 (Gold)", scale=alt.Scale(domain=y_domain)),
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
		title=f"[{top_item}] 전체 시세 & 수요일(Reset) 영향 분석",
		height=400,
	)
	.interactive()
)

st.altair_chart(chart_all, use_container_width=True)


# -------------------------------------------------------------------------
# 7. 시각화 3: 히스토리 + 미래 예측
# -------------------------------------------------------------------------
st.markdown("### 🔮 향후 3일 시세 예측 (히스토리 + 미래)")

if future_df is None or future_df.empty:
	st.info("현재 선택한 모델에서는 미래 예측(predict_future)이 구현되지 않았습니다.")
else:
	hist_tail = df_ml[["date", "price"]].iloc[-zoom_n:].copy()
	hist_tail["type"] = "History"

	future_plot = future_df.rename(columns={"price": "price"}).copy()
	future_plot["type"] = "Forecast"

	df_future_plot = pd.concat([hist_tail, future_plot], ignore_index=True)

	y_min_f = df_future_plot["price"].min()
	y_max_f = df_future_plot["price"].max()
	padding_f = (y_max_f - y_min_f) * 0.05
	y_domain_f = [y_min_f - padding_f, y_max_f + padding_f]

	chart_future = (
		alt.Chart(df_future_plot)
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
		.properties(
			title=f"[{top_item}] 최근 {days_to_show}일 + 향후 1일 시세 예측"
		)
		.interactive()
	)

	st.altair_chart(chart_future, use_container_width=True)


# -------------------------------------------------------------------------
# 8. 원시 데이터 보기
# -------------------------------------------------------------------------
with st.expander("원시 데이터 / Feature 데이터 확인"):
	st.markdown("#### 🔹 원본 타겟 데이터 (df_target)")
	st.dataframe(df_target[["date", "name", "grade", "price"]].tail(50))

	st.markdown("#### 🔹 ML 학습용 데이터 (df_ml)")
	base_cols = ["date", "price", "lag_30m", "rsi", "is_overbought", "is_oversold"]
	if "gpt_score" in df_ml.columns:
		base_cols.append("gpt_score")

	st.dataframe(df_ml[base_cols].tail(50))
