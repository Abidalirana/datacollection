# data_processing/preprocess.py
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
# Remove +asyncpg if using sync pandas read
sync_db_url = DATABASE_URL.replace("+asyncpg", "")
engine = create_engine(sync_db_url)


def load_tables():
    """Load all DB tables into Pandas DataFrames"""
    tables = [
        "users", "sessions", "trades", "emotions", "journals",
        "simulator_logs", "reset_challenges", "feature_usage",
        "recovery_plans", "rulebook_votes"
    ]

    dataframes = {}
    for table in tables:
        df = pd.read_sql_table(table, con=engine)
        dataframes[table] = df
    return dataframes


def preprocess_data(dfs: dict):
    """
    Clean & preprocess each table
    """
    # Fill missing exit_time in trades
    if "trades" in dfs:
        trades = dfs["trades"].copy()
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["exit_time"] = trades["exit_time"].fillna(trades["entry_time"])
        dfs["trades"] = trades

    # Fill missing emotions
    if "emotions" in dfs:
        emotions = dfs["emotions"].copy()
        emotions["emotion"] = emotions["emotion"].fillna("neutral")
        emotions["timestamp"] = pd.to_datetime(emotions["timestamp"])
        # Ensure trade_id is numeric (handle NaNs safely)
        emotions["trade_id"] = emotions["trade_id"].fillna(-1).astype(int)
        dfs["emotions"] = emotions

    return dfs


def merge_tables(dfs: dict):
    trades = dfs["trades"].copy()
    emotions = dfs["emotions"].copy()
    journals = dfs["journals"].copy()

    # Ensure merge keys have the same type
    trades["id"] = trades["id"].astype(int)

    # Ensure trade_id is int and remove invalid rows
    valid_emotions = emotions[emotions["trade_id"].notna()].copy()
    valid_emotions["trade_id"] = valid_emotions["trade_id"].astype(int)

    # ⚡ Make sure trades have user_id for later join
    if "user_id" not in trades.columns:
        raise KeyError("❌ 'user_id' column missing in trades table! Cannot merge with journals.")

    # Merge trades + emotions
    trades_emotions = pd.merge(trades, valid_emotions, left_on="id", right_on="trade_id", how="left")

    # Merge with journals
    if "user_id" not in trades_emotions.columns:
        trades_emotions["user_id"] = trades["user_id"]  # Copy from trades if lost during merge

    ml_data = pd.merge(trades_emotions, journals, on="user_id", how="left")

    return ml_data


if __name__ == "__main__":
    dfs = load_tables()
    dfs_clean = preprocess_data(dfs)
    ml_ready = merge_tables(dfs_clean)
    print("✅ Preprocessed data ready for ML/LLM:")
    print(ml_ready.head())
