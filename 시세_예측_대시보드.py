# 시세_예측_대시보드.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

from data_loader import load_merged_data
from features import filter_item, make_ml_dataset
# from models_old import train_random_forest, forecast_future
from models.factory import get_model
from backtest import simulate_strict_investor

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
st.caption("로스트아크 거래소 아이템 시세 예측 (RandomForest 예시 버전)")

# -------------------------------------------------------------------------
# 1. 사이드바 - 검색/학습 설정 (폼 + Enter 제출)
# -------------------------------------------------------------------------
with st.sidebar:
	st.header("검색 / 학습 설정")

	df_final = load_merged_data()

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
		POINTS_PER_DAY = 144  # 10분 단위 기준
		
		zoom_n = days_to_show * POINTS_PER_DAY
		
		run_button = st.form_submit_button("학습 & 예측 실행")

	# 페이지 분리 하면서 주석처리함.
	# st.sidebar.subheader("🧪 투자자 시뮬레이션")

	# enable_investor_mode = st.sidebar.checkbox("깐깐한 투자자 시뮬레이션", value=False)

	# initial_balance = st.sidebar.number_input(
	# 	"초기 투자금 (G)",
	# 	min_value=1_000_000,
	# 	max_value=100_000_000,
	# 	value=10_000_000,
	# 	step=1_000_000,
	# )

	# max_inventory = st.sidebar.slider(
	# 	"최대 보유 개수",
	# 	min_value=1,
	# 	max_value=20,
	# 	value=5,
	# )

	# target_margin = st.sidebar.slider(
	# 	"매수 기준 기대 수익률 (%)",
	# 	min_value=1,
	# 	max_value=30,
	# 	value=10,
	# ) / 100.0

	# fee_rate = st.sidebar.slider(
	# 	"거래 수수료율 (%)",
	# 	min_value=0.0,
	# 	max_value=10.0,
	# 	value=5.0,
	# 	step=0.5,
	# ) / 100.0


# -------------------------------------------------------------------------
# 2. 버튼 눌렀을 때만 새로 계산 → 세션에 저장
# -------------------------------------------------------------------------
if run_button:
	with st.spinner("데이터 필터링 중..."):
		result = filter_item(df_final, target_keyword, target_grade)

	if result is None:
		st.error(f"'{target_keyword}' (등급: {target_grade}) 에 해당하는 데이터가 없습니다.")
	else:
		df_target, top_item = result

		with st.spinner("Feature Engineering 처리 중..."):
			df_ml, features = make_ml_dataset(df_target)

		if len(df_ml) < 300:
			st.warning(f"Feature 생성 후 데이터가 {len(df_ml)}개입니다. (최소 300개 이상일 때가 더 안정적)")
		else:
			with st.spinner("학습 및 예측 중..."):
				# model, y_test, y_pred, split_idx, rmse, r2 = train_random_forest(df_ml, features)

				# # 🔮 미래 예측 (예: 1일 = 144 스텝)
				# future_steps = 144
				# future_df = forecast_future(model, df_ml, features, steps=future_steps)

				price_model = get_model("rf")	# 나중에 "ensemble" 로만 바꾸면 됨
				price_model.train(df_ml, features)

				y_test, y_pred, split_idx, rmse, r2 = price_model.predict_test()
				future_df = price_model.predict_future(steps=144)

			st.session_state.rf_result = {
				"df_target": df_target,
				"df_ml": df_ml,
				"top_item": top_item,
				"y_test": y_test,
				"y_pred": y_pred,
				"split_idx": split_idx,
				"rmse": rmse,
				"r2": r2,
				"days_to_show": days_to_show,
				"future_df": future_df,
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
zoom_n = days_to_show * 144

st.subheader(f"🎯 분석 대상: {top_item}")

# -----------------------------
# 현재 가격 & 전일 평균 가격
# -----------------------------
# 1) 가장 최근 시점(현재 가격)
latest_ts = df_target["date"].max()
latest_row = df_target.loc[df_target["date"] == latest_ts].iloc[-1]
current_price = float(latest_row["price"])

# 2) 전일 평균 가격 계산
#    - 현재 시점 날짜의 전날 0시 ~ 당일 0시 직전
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

# Streamlit 멀티페이지용 내비게이션 링크
st.page_link(
	"pages/투자_시뮬레이션.py",  # 투자 모드 페이지 파일 경로
	label="투자 시뮬레이션 페이지 열기",
	icon="➡️",
)


# -------------------------------------------------------------------------
# 5. 시각화 1: 테스트 구간 확대
# -------------------------------------------------------------------------
st.markdown("### 📈 최근 테스트 구간 확대 그래프 (인터랙티브)")

test_dates = df_ml["date"].iloc[split_idx:]

if zoom_n > len(test_dates):
	zoom_n = len(test_dates)

zoom_slice = slice(-zoom_n, None)

df_plot = pd.DataFrame({
	"date": test_dates.iloc[zoom_slice],
	"Actual (실제)": y_test.iloc[zoom_slice].values,
	"Prediction (예측)": y_pred[zoom_slice]
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

all_dates = df_ml["date"]
all_prices = df_ml["price"]

df_line_all = pd.DataFrame({
	"date": all_dates,
	"price": all_prices,
	"type": "History (전체 흐름)"
})

test_dates_full = all_dates.iloc[split_idx:]
real_test_price = all_prices.iloc[split_idx:]

df_line_test = pd.DataFrame({
	"date": test_dates_full,
	"price": real_test_price,
	"type": "Actual (검증 구간)"
})

df_line_pred = pd.DataFrame({
	"date": test_dates_full,
	"price": y_pred,
	"type": "Prediction (예측)"
})

df_lines = pd.concat([df_line_all, df_line_test, df_line_pred], ignore_index=True)

unique_days = df_ml["date"].dt.normalize().drop_duplicates()
weds = unique_days[unique_days.dt.dayofweek == 2]

df_weds = pd.DataFrame({
	"start": weds,
	"end": weds + pd.Timedelta(days=1),
	"label": "수요일 (Reset)"
})

split_time = all_dates.iloc[split_idx]
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
		opacity=alt.value(0.12)
	)
)

lines = (
	alt.Chart(df_lines)
	.mark_line()
	.encode(
		x=alt.X("date:T", title="날짜"),
		y=alt.Y(
			"price:Q",
			title="가격 (Gold)",
			scale=alt.Scale(domain=y_domain)
		),
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
		size=alt.value(2)
	)
)

chart_all = (
	(rect + lines + rule)
	.properties(
		title=f"[{top_item}] 전체 시세 & 수요일(Reset) 영향 분석",
		height=400
	)
	.interactive()
)

st.altair_chart(chart_all, use_container_width=True)

# -------------------------------------------------------------------------
# 7. 시각화 3: 히스토리 + 미래 예측
# -------------------------------------------------------------------------
st.markdown("### 🔮 향후 1일 시세 예측 (히스토리 + 미래)")

# 최근 구간 히스토리 (같은 zoom_n 사용)
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
# 투자자 모드
# -------------------------------------------------------------------------
# if enable_investor_mode:
# 	st.subheader("💼 깐깐한 투자자 모드 결과")

# 	if st.button("가상 투자 시뮬레이션 실행"):
# 		result = simulate_strict_investor(
# 			test_dates=test_dates,
# 			y_test=y_test,
# 			y_pred=y_pred,
# 			initial_balance=initial_balance,
# 			fee_rate=fee_rate,
# 			max_inventory=max_inventory,
# 			target_margin=target_margin,
# 		)

# 		st.metric("순수익", f"{result['net_profit']:+,.0f} G")
# 		st.metric("수익률 (ROI)", f"{result['roi']:+,.2f} %")
# 		st.metric("최종 자산 가치", f"{result['final_asset_value']:,.0f} G")

# -------------------------------------------------------------------------
# 8. 원시 데이터 보기
# -------------------------------------------------------------------------
with st.expander("원시 데이터 / Feature 데이터 확인"):
	st.markdown("#### 🔹 원본 타겟 데이터 (df_target)")
	st.dataframe(df_target[["date", "name", "grade", "price"]].tail(50))

	st.markdown("#### 🔹 ML 학습용 데이터 (df_ml)")
	st.dataframe(df_ml[["date", "price", "lag_10m", "rsi", "is_overbought", "is_oversold"]].tail(50))
