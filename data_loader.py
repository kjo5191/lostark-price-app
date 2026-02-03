# data_loader.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

@st.cache_resource
def get_engine():
	db_connection_str = (
		f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
		f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
	)
	engine = create_engine(db_connection_str)
	return engine


@st.cache_data
def load_merged_data():
	engine = get_engine()

	df_logs = pd.read_sql("SELECT * FROM market_price_logs", engine)
	df_items = pd.read_sql("SELECT id, name, grade, category_code FROM market_items", engine)

	df_merged = pd.merge(
		df_logs,
		df_items,
		left_on="item_id",
		right_on="id",
		how="left"
	)

	df_final = df_merged.drop(columns=["id_y", "id_x"])
	df_final = df_final.rename(columns={
		"current_min_price": "price",
		"logged_at": "date"
	})

	df_final["date"] = pd.to_datetime(df_final["date"])
	df_final = df_final.sort_values("date")

	return df_final
