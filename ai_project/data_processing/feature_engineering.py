# data_processing/feature_engineering.py
import pandas as pd
import numpy as np

def add_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from trades, emotions, and journals for ML models
    """
    # 1️⃣ Trade duration (seconds)
    df["trade_duration"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds()

    # 2️⃣ Risk/Reward ratio (fill missing with 1)
    df["risk_reward_ratio"] = df["risk_reward_ratio"].fillna(1.0)

    # 3️⃣ Max drawdown (fill missing with 0)
    df["max_drawdown"] = df["max_drawdown"].fillna(0.0)

    # 4️⃣ Encode outcome: win=1, loss=0
    df["outcome_encoded"] = df["outcome"].map({"win": 1, "loss": 0}).fillna(0).astype(int)

    return df


def add_emotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert emotions to numeric features
    """
    if "emotion" in df.columns:
        emotion_dummies = pd.get_dummies(df["emotion"], prefix="emotion")
        df = pd.concat([df, emotion_dummies], axis=1)

    return df


def add_journal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from journal text and confidence_score
    """
    if "confidence_score" in df.columns:
        mean_conf = df["confidence_score"].mean()
        df["confidence_score"] = df["confidence_score"].fillna(mean_conf)

    # Example: journal length feature
    if "content" in df.columns:
        df["journal_length"] = df["content"].apply(lambda x: len(str(x).split()))

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns like instrument, strategy
    """
    # One-hot encode instrument
    if "instrument" in df.columns:
        instrument_dummies = pd.get_dummies(df["instrument"], prefix="instr")
        df = pd.concat([df, instrument_dummies], axis=1)

    # One-hot encode strategy
    if "strategy" in df.columns:
        strategy_dummies = pd.get_dummies(df["strategy"], prefix="strategy")
        df = pd.concat([df, strategy_dummies], axis=1)

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline to generate all features
    """
    df = add_trade_features(df)
    df = add_emotion_features(df)
    df = add_journal_features(df)
    df = encode_categorical(df)

    # Drop columns not needed for ML
    drop_cols = ["content", "emotion", "instrument", "strategy", "outcome", "entry_time", "exit_time"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


if __name__ == "__main__":
    from preprocess import load_tables, preprocess_data, merge_tables

    dfs = load_tables()
    dfs_clean = preprocess_data(dfs)
    ml_ready = merge_tables(dfs_clean)

    features = generate_features(ml_ready)
    print("✅ Feature-engineered data ready for ML/LLM:")
    print(features.head())
