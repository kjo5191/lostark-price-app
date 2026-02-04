# pages/투자_시뮬레이션.py

import streamlit as st
import pandas as pd

from data_loader import load_merged_data
from features import filter_item, make_ml_dataset
from models import train_random_forest
from backtest import simulate_strict_investor

st.set_page_config(
	page_title="투자 시뮬레이션",
	layout="wide"
)

st.title("💼 투자 시뮬레이션 (백테스트)")


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

	with st.spinner("RandomForest 학습 & 예측 중..."):
		model, y_test, y_pred, split_idx, rmse, r2 = train_random_forest(df_ml, features)

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
