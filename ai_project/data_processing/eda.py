# data_processing/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_trade_outcomes(df: pd.DataFrame):
    """
    Bar chart: number of wins vs losses
    """
    if "outcome_encoded" not in df.columns:
        print("⚠ outcome_encoded column missing. Run feature_engineering first.")
        return None

    outcome_counts = df["outcome_encoded"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=outcome_counts.index, y=outcome_counts.values, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Loss", "Win"])
    ax.set_title("Trade Outcomes")
    ax.set_ylabel("Count")
    return fig

def plot_trade_duration(df: pd.DataFrame):
    """
    Histogram + KDE of trade durations
    """
    if "trade_duration" not in df.columns:
        print("⚠ trade_duration column missing. Run feature_engineering first.")
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["trade_duration"].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Trade Duration Distribution (seconds)")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    return fig

def plot_emotions(df: pd.DataFrame):
    """
    Bar chart: distribution of one-hot encoded emotions
    """
    emotion_cols = [c for c in df.columns if c.startswith("emotion_")]
    if not emotion_cols:
        print("⚠ No one-hot emotion columns found. Run feature_engineering first.")
        return None

    emotion_sums = df[emotion_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=emotion_sums.index, y=emotion_sums.values, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Emotions Distribution")
    ax.set_ylabel("Count")
    return fig

def plot_correlation_matrix(df: pd.DataFrame):
    """
    Heatmap: correlation between numeric features
    """
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    if numeric_df.empty:
        print("⚠ No numeric columns to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    return fig

def summary_statistics(df: pd.DataFrame):
    """
    Print basic stats for numeric columns
    """
    print("✅ Summary statistics:")
    print(df.describe())
