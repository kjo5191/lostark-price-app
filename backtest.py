# backtest.py

import pandas as pd


def simulate_strict_investor(
	test_dates,
	y_test,
	y_pred,
	initial_balance: float,
	fee_rate: float,
	per_trade_ratio: float,
	max_position_ratio: float,
	target_margin: float,
):
	"""
	비율 기반 '깐깐한 투자자' 시뮬레이션

	- initial_balance: 기준 자산 (골드)
	- per_trade_ratio: 1회 매수 시 자산 대비 투자 비율 (0.05 = 5%)
	- max_position_ratio: 한 아이템에 최대 투자 가능한 비율 (0.3 = 30%)
	- fee_rate: 매도 시 수수료율 (0.05 = 5%)
	- target_margin: 매수 기준 기대 수익률 (0.1 = 10%)

	과거 test 구간에 대해:
		예측가와 현재가의 괴리율이 target_margin 이상이면 매수
		예측가 이상이 되었거나, 5% 이상 수익이 나면 전량 매도
	"""

	# pandas Series/Index 로 통일
	test_dates = pd.Series(test_dates).reset_index(drop=True)
	y_test = pd.Series(y_test).reset_index(drop=True)
	y_pred = pd.Series(y_pred).reset_index(drop=True)

	balance = float(initial_balance)		# 현금
	position_qty = 0						# 보유 수량
	avg_buy_price = 0.0						# 평단가

	max_position_value = initial_balance * max_position_ratio

	records = []

	for date, real_price, pred_price in zip(test_dates, y_test, y_pred):
		real_price = float(real_price)
		pred_price = float(pred_price)

		# 현재 보유 포지션 평가액
		position_value = position_qty * real_price

		# ---------------------------------------------------
		# 🔵 매수 전략: 예측가가 충분히 높고, 남은 캐파가 있을 때만
		# ---------------------------------------------------
		# 이번 트레이드에 사용할 최대 예산 (비율 기반)
		buy_budget = initial_balance * per_trade_ratio

		# 최대 포지션 비율을 넘지 않도록 남은 캐파 계산
		remaining_capacity_value = max(0.0, max_position_value - position_value)

		# 실제로 사용할 수 있는 예산 = (트레이드 예산, 남은 캐파, 현재 잔고) 중 최소
		usable_budget = min(buy_budget, remaining_capacity_value, balance)

		if usable_budget >= real_price:
			expected_profit_margin = (pred_price - real_price) / real_price

			if expected_profit_margin > target_margin:
				# 매수 가능 수량 (정수)
				buy_qty = int(usable_budget // real_price)

				if buy_qty > 0:
					cost = buy_qty * real_price
					balance -= cost

					# 평단가 갱신 (가중 평균)
					new_position_qty = position_qty + buy_qty
					if position_qty == 0:
						avg_buy_price = real_price
					else:
						avg_buy_price = (
							avg_buy_price * position_qty + real_price * buy_qty
						) / new_position_qty

					position_qty = new_position_qty

					records.append(
						{
							"type": "BUY",
							"date": date,
							"price": real_price,
							"pred_price": pred_price,
							"expected_margin": expected_profit_margin,
							"qty": buy_qty,
							"profit": None,
						}
					)

		# ---------------------------------------------------
		# 🔵 매도 전략: 예측가 이상이 되었거나, 5% 이상 수익이면 전량 매도
		# ---------------------------------------------------
		if position_qty > 0:
			current_profit_rate = (real_price - avg_buy_price) / avg_buy_price

			if real_price >= pred_price or current_profit_rate > 0.05:
				sell_qty = position_qty
				gross_amount = sell_qty * real_price
				net_amount = gross_amount * (1.0 - fee_rate)

				balance += net_amount

				profit = net_amount - sell_qty * avg_buy_price

				records.append(
					{
						"type": "SELL",
						"date": date,
						"price": real_price,
						"pred_price": pred_price,
						"expected_margin": current_profit_rate,
						"qty": sell_qty,
						"profit": profit,
					}
				)

				# 포지션 정리
				position_qty = 0
				avg_buy_price = 0.0

	# ---------------------------------------------------
	# 최종 정산: 마지막 시점 가격 기준으로 잔여 포지션 평가
	# ---------------------------------------------------
	if len(y_test) > 0:
		last_price = float(y_test.iloc[-1])
	else:
		last_price = 0.0

	unrealized_value = position_qty * last_price * (1.0 - fee_rate)
	final_asset_value = balance + unrealized_value
	net_profit = final_asset_value - initial_balance
	roi = (net_profit / initial_balance) * 100.0 if initial_balance > 0 else 0.0

	trade_history = pd.DataFrame(records)

	return {
		"final_asset_value": final_asset_value,
		"net_profit": net_profit,
		"roi": roi,
		"trade_history": trade_history,
	}
