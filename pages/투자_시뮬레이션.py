# pages/투자_시뮬레이션.py

import streamlit as st
import pandas as pd
import numpy as np

from ai_advisor import get_ai_advice
from data_loader import load_merged_data
from features import filter_item, make_ml_dataset
# from models_old import train_random_forest
from models.factory import get_model
from backtest import simulate_strict_investor

st.set_page_config(
	page_title="투자 시뮬레이션",
	layout="wide"
)

st.title("💼 투자 시뮬레이션")


# -------------------------------------------------------------------------
# 0. 세션에 메인 페이지 결과가 있는지 확인
# -------------------------------------------------------------------------
has_session_result = (
	"rf_result" in st.session_state
	and st.session_state.rf_result is not None
)

with st.sidebar:
	st.header("시뮬레이션 설정")

	if has_session_result:
		use_session = st.checkbox(
			"메인 페이지 결과 사용 (다시 학습 안 함)",
			value=True,
			help="메인 대시보드에서 마지막으로 학습한 아이템의 예측 결과를 그대로 사용합니다.",
		)
	else:
		use_session = False
		st.caption("⚠ 메인 페이지에서 먼저 한 번 학습을 돌리면, 그 결과를 재사용할 수 있어요.")

	# 기준 자산 (표시용 + 비율 계산용)
	initial_balance = st.number_input(
		"초기 투자금 (G)",
		min_value=1_000_000,
		max_value=100_000_000,
		value=10_000_000,
		step=1_000_000,
		help="비율 기준이 되는 자산입니다. 비율 기반 전략이라 아이템 간 비교에 유리합니다.",
	)

	# 🔥 비율 기반 파라미터
	per_trade_ratio = (
		st.slider(
			"1회 매수 비율 (%)",
			min_value=1,
			max_value=50,
			value=5,
			help="한 번 매수할 때 전체 자산의 몇 %를 사용할지 설정합니다.",
		)
		/ 100.0
	)

	max_position_ratio = (
		st.slider(
			"최대 투자 비율 (%)",
			min_value=5,
			max_value=100,
			value=30,
			help="한 아이템에 최대 몇 %까지 투자할지 설정합니다.",
		)
		/ 100.0
	)

	target_margin = (
		st.slider(
			"매수 기준 기대 수익률 (%)",
			min_value=1,
			max_value=30,
			value=10,
			help="예측가가 현재가보다 몇 % 이상 높을 때만 매수할지 기준을 정합니다.",
		)
		/ 100.0
	)

	fee_rate = (
		st.slider(
			"거래 수수료율 (%)",
			min_value=0.0,
			max_value=10.0,
			value=5.0,
			step=0.5,
		)
		/ 100.0
	)

	if not use_session:
		st.markdown("---")
		st.subheader("아이템 선택")

		df_final = load_merged_data()

		grade_list = sorted(df_final["grade"].dropna().unique())
		grade_options = ["전체"] + grade_list

		target_grade = st.selectbox(
			"아이템 등급",
			grade_options,
			index=grade_options.index("유물") if "유물" in grade_options else 0,
		)

		target_keyword = st.text_input(
			"아이템 이름 키워드",
			value="원한",
		)

	run_button = st.button("시뮬레이션 실행")



# -------------------------------------------------------------------------
# 1. 버튼 안 눌렀으면 안내 후 종료
# -------------------------------------------------------------------------
if not run_button:
	st.info("왼쪽에서 조건을 설정하고 **[시뮬레이션 실행]** 버튼을 눌러줘.")
	st.stop()


# -------------------------------------------------------------------------
# 2-A. 메인 페이지 세션 결과 재사용 (빠른 모드)
# -------------------------------------------------------------------------
if use_session and has_session_result:
	res = st.session_state.rf_result

	df_ml = res["df_ml"]
	top_item = res["top_item"]
	y_test = res["y_test"]
	y_pred = res["y_pred"]
	split_idx = res["split_idx"]

	test_dates = df_ml["date"].iloc[split_idx:]

	with st.spinner("메인 페이지 결과를 기반으로 시뮬레이션 중..."):
		sim_result = simulate_strict_investor(
			test_dates=test_dates,
			y_test=y_test,
			y_pred=y_pred,
			initial_balance=initial_balance,
			fee_rate=fee_rate,
			per_trade_ratio=per_trade_ratio,
			max_position_ratio=max_position_ratio,
			target_margin=target_margin,
		)


# -------------------------------------------------------------------------
# 2-B. 세션이 없거나, 강제로 다시 학습하는 경우 (느린 모드)
# -------------------------------------------------------------------------
else:
	# 세션 재사용이 불가능한 경우: 여기서 다시 전체 파이프라인 실행
	with st.spinner("데이터 필터링 중..."):
		result = filter_item(df_final, target_keyword, target_grade)

	if result is None:
		st.error(f"'{target_keyword}' (등급: {target_grade}) 에 해당하는 데이터가 없습니다.")
		st.stop()

	df_target, top_item = result

	with st.spinner("Feature Engineering 처리 중..."):
		df_ml, features = make_ml_dataset(df_target)

	if len(df_ml) < 300:
		st.warning(f"Feature 생성 후 데이터가 {len(df_ml)}개입니다. (최소 300개 이상일 때가 더 안정적)")
		st.stop()

	# with st.spinner("RandomForest 학습 & 예측 중..."):
	# 	model, y_test, y_pred, split_idx, rmse, r2 = train_random_forest(df_ml, features)

	with st.spinner("모델 학습 & 예측 중..."):
		price_model = get_model("rf")  # 나중에 "ensemble", "lstm" 등으로 교체만 하면 됨
		price_model.train(df_ml, features)

		y_test, y_pred, split_idx, rmse, r2 = price_model.predict_test()

	test_dates = df_ml["date"].iloc[split_idx:]

	with st.spinner("투자 시뮬레이션 실행 중..."):
		sim_result = simulate_strict_investor(
			test_dates=test_dates,
			y_test=y_test,
			y_pred=y_pred,
			initial_balance=initial_balance,
			fee_rate=fee_rate,
			per_trade_ratio=per_trade_ratio,
			max_position_ratio=max_position_ratio,
			target_margin=target_margin,
		)


# -------------------------------------------------------------------------
# 3. 결과 표시
# -------------------------------------------------------------------------
st.subheader(f"🎯 대상 아이템: {top_item}")

col1, col2, col3 = st.columns(3)
with col1:
	st.metric("최종 자산 가치", f"{sim_result['final_asset_value']:,.0f} G")
with col2:
	st.metric("순수익", f"{sim_result['net_profit']:+,.0f} G")
with col3:
	st.metric("수익률 (ROI)", f"{sim_result['roi']:+.2f} %")

st.markdown("#### 📜 거래 기록")
trade_df = sim_result["trade_history"]
if trade_df.empty:
	st.info("거래가 발생하지 않았습니다. (조건이 너무 깐깐한지 확인해보세요)")
else:
	st.dataframe(trade_df.sort_values("date"))

# -------------------------------------------------------------------------
# 📌 현재 전략 기준 투자 판단 (세션 기반)
# -------------------------------------------------------------------------
# st.subheader("📌 현재 전략 기준 투자 판단")

# 세션 결과를 사용하는 경우에만 디테일한 의견 제공
if use_session and has_session_result:
	res = st.session_state.rf_result

	df_target = res["df_target"]
	top_item = res["top_item"]
	future_df = res.get("future_df", None)

	prices = df_target["price"].reset_index(drop=True)

	# 현재 가격
	if len(prices) == 0:
		st.info("시세 데이터가 부족해서 현재 의견을 계산할 수 없습니다.")
	else:
		current_price = float(prices.iloc[-1])

		# 1) 단기/장기 이동평균 기반 추세 계산
		WINDOW_SHORT = 144		# 1일 (10분 단위 * 144)
		WINDOW_LONG = 288		# 2일

		trend_label = "데이터 부족"
		trend_score = 0.0

		if len(prices) >= WINDOW_SHORT:
			short_window = min(WINDOW_SHORT, len(prices))
			short_ma = prices.iloc[-short_window:].mean()

			if len(prices) >= WINDOW_LONG:
				long_ma = prices.iloc[-WINDOW_LONG:].mean()
			else:
				# 데이터가 부족하면 전체 평균을 장기 기준으로 사용
				long_ma = prices.mean()

			if long_ma > 0:
				trend_score = (short_ma - long_ma) / long_ma
			else:
				trend_score = 0.0

			if trend_score > 0.03:
				trend_label = "상승 추세"
			elif trend_score < -0.03:
				trend_label = "하락 추세"
			else:
				trend_label = "횡보"
		else:
			trend_label = "데이터 부족"
			trend_score = 0.0

		# 2) 모델 기준 향후 1일 기대 수익률 (future_df 기반)
		expected_return = None

		if future_df is not None and not future_df.empty:
			if "price" in future_df.columns:
				future_prices = future_df["price"]
			else:
				# 혹시 컬럼명이 다르면 숫자형 첫 컬럼 사용
				future_prices = future_df.select_dtypes("number").iloc[:, 0]

			horizon = min(144, len(future_prices))	# 1일(144포인트) 또는 그 이하
			if horizon > 0 and current_price > 0:
				future_mean = future_prices.iloc[:horizon].mean()
				expected_return = (future_mean - current_price) / current_price

		# 3) 최종 매수/관망/비추천 판단
		if expected_return is None or trend_label == "데이터 부족":
			signal = "판단 보류"
			reason = "데이터가 부족하거나 미래 예측 정보가 없어서 뚜렷한 의견을 내기 어렵습니다."
		else:
			# 🔧 임계값은 나중에 같이 튜닝 가능
			if expected_return >= 0.08 and trend_score > 0.03:
				signal = "매수 추천"
				reason = (
					"단기 상승 추세이고, 모델 기준 향후 1일 기대 수익률이 8% 이상입니다. "
					"다만 실제 거래에서는 분할 매수를 고려하는 것이 안전합니다."
				)
			elif expected_return <= -0.02 and trend_score < -0.03:
				signal = "매수 비추천"
				reason = (
					"하락 추세이며, 모델이 단기적으로 수익을 기대하지 않습니다. "
					"당분간 관망하는 편이 더 안전해 보입니다."
				)
			else:
				signal = "관망"
				reason = (
					"추세와 기대 수익률이 애매한 구간입니다. "
					"지금은 과도한 진입보다는 추세를 조금 더 지켜보는 것을 권장합니다."
				)

		sig_col1, sig_col2, sig_col3 = st.columns(3)

		with sig_col1:
			st.metric("투자 의견", signal)

		with sig_col2:
			if expected_return is not None:
				st.metric(
					"향후 1일 기대 수익률",
					f"{expected_return * 100:+.2f} %",
				)

		with sig_col3:
			if trend_label != "데이터 부족":
				st.metric(
					"단기 추세",
					trend_label,
					f"{trend_score * 100:+.2f} %",
				)

		st.caption(
			"※ 본 의견은 과거 시세와 단기 예측을 기반으로 한 참고용 정보이며, "
			"실제 게임 내 거래 결정에 따른 책임은 플레이어 본인에게 있습니다."
		)

else:
	# 세션을 사용하지 않는 경우엔, 과감히 판단 보류만 표기
	st.info(
		"현재 전략 기준 투자 의견은 메인 대시보드에서 먼저 학습을 실행한 뒤, "
		"'메인 페이지 결과 사용' 옵션으로 시뮬레이션할 때 제공됩니다."
	)

# -------------------------------------------------------------------------
# 4. AI 투자 전략 가이드
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 AI 투자 전략 가이드")

# 메인 대시보드에서 학습한 결과를 사용할 때만 AI 가이드 제공
if use_session and has_session_result:
	res = st.session_state.rf_result

	df_target = res["df_target"]
	top_item = res["top_item"]
	future_df_ensemble = res.get("future_df_ensemble", None)

	# 데이터 체크
	if df_target is None or df_target.empty:
		st.info("시세 데이터가 부족해서 AI 가이드를 생성할 수 없습니다.")
	elif future_df_ensemble is None or future_df_ensemble.empty:
		st.info("미래 예측 데이터가 없어 AI 가이드를 생성할 수 없습니다.")
	else:
		# 현재 가격
		current_price = float(df_target["price"].iloc[-1])

		# 🔹 ai_advisor.py 형식(df_forecast: ds, forecast)으로 변환
		df_forecast = future_df_ensemble.copy()

		rename_map = {}
		if "date" in df_forecast.columns:
			rename_map["date"] = "ds"
		if "ensemble_price" in df_forecast.columns:
			rename_map["ensemble_price"] = "forecast"

		df_forecast = df_forecast.rename(columns=rename_map)

		if "ds" not in df_forecast.columns or "forecast" not in df_forecast.columns:
			st.warning("AI 가이드를 생성하기 위한 'ds' / 'forecast' 컬럼이 없습니다.")
		else:
			# 예측 최저/최고 (앙상블 forecast 기준)
			min_pred = int(df_forecast["forecast"].min())
			max_pred = int(df_forecast["forecast"].max())

			# 🔹 AI 응답 캐시 (아이템 단위)
			if "ai_advice_cache" not in st.session_state:
				st.session_state.ai_advice_cache = {}

			cache_key = top_item
			if cache_key not in st.session_state.ai_advice_cache:
				with st.spinner("AI 전략 분석 중..."):
					# ai_advisor.py 시그니처에 맞게 전달
					advice_text = get_ai_advice(
						top_item,
						current_price,
						df_forecast,
					)
					st.session_state.ai_advice_cache[cache_key] = advice_text

			cached_advice = st.session_state.ai_advice_cache[cache_key]

			# 🔹 메트릭 + AI 텍스트 출력
			c1, c2, c3 = st.columns(3)
			with c1:
				st.metric("현재 시세", f"{current_price:,.0f} G")
			with c2:
				st.metric(
					"예측 최저",
					f"{min_pred:,.0f} G",
					delta=f"{min_pred - current_price:,.0f} G",
					delta_color="inverse",
				)
			with c3:
				st.metric(
					"예측 최고",
					f"{max_pred:,.0f} G",
					delta=f"{max_pred - current_price:,.0f} G",
				)

			st.info(cached_advice, icon="📊")
			st.caption(
				"※ 본 AI 가이드는 과거 시세와 예측 결과를 바탕으로 생성된 참고용 의견이며, "
				"실제 게임 내 거래 결정에 따른 책임은 플레이어 본인에게 있습니다."
			)

else:
	# 메인 결과를 사용하지 않는 경우엔 AI 가이드 비활성화
	st.info(
		"AI 투자 전략 가이드는 메인 대시보드에서 먼저 학습을 실행하고, "
		"'메인 페이지 결과 사용' 옵션으로 시뮬레이션할 때 제공됩니다."
	)
