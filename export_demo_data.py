# export_demo_data.py

import os
import pandas as pd

from data_loader import load_merged_data, load_gpt_scores

def main():
	print("[1] DB에서 통합 시세 데이터 로드 중...")
	df_merged = load_merged_data()
	print(f" - merged rows: {len(df_merged)}")

	print("[2] DB에서 GPT 점수 데이터 로드 중...")
	df_gpt = load_gpt_scores()
	print(f" - gpt rows: {len(df_gpt)}")

	# 필요하면 여기서 샘플링 / 기간 제한
	# 예) 최근 30일만 사용
	# df_merged = df_merged[df_merged["date"] >= df_merged["date"].max() - pd.Timedelta(days=30)]

	# 예) 아이템 몇 개만 남기기 (용량 줄이고 싶다면)
	# top_items = df_merged["name"].value_counts().index[:3]
	# df_merged = df_merged[df_merged["name"].isin(top_items)]

	os.makedirs("data", exist_ok=True)

	merged_path = "data/sample_merged_data.csv"
	gpt_path = "data/sample_gpt_scores.csv"

	print(f"[3] CSV로 저장 중... ({merged_path}, {gpt_path})")
	df_merged.to_csv(merged_path, index=False)
	df_gpt.to_csv(gpt_path, index=False)

	print("[완료] 데모용 CSV 저장이 끝났습니다.")

if __name__ == "__main__":
	main()
