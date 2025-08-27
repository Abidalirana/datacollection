# data_processing/run_preprocess.py

import os
from datetime import datetime
from preprocess import load_tables, preprocess_data, merge_tables
from feature_engineering import generate_features
from eda import (
    summary_statistics,
    plot_trade_outcomes,
    plot_trade_duration,
    plot_emotions,
    plot_correlation_matrix,
)
import matplotlib.pyplot as plt

# Folder to save processed files and EDA plots
SAVE_FOLDER = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
PROCESSED_FILENAME = f"processed_features_{timestamp}.csv"
PROCESSED_PATH = os.path.join(SAVE_FOLDER, PROCESSED_FILENAME)
EDA_FOLDER = os.path.join(SAVE_FOLDER, "eda_outputs")

# Create folder for EDA plots if not exists
os.makedirs(EDA_FOLDER, exist_ok=True)

def save_plot(fig, name):
    """Save matplotlib figure to EDA folder with timestamp"""
    path = os.path.join(EDA_FOLDER, f"{name}_{timestamp}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"💾 EDA plot saved: {path}")

def run_full_pipeline(run_eda=False, save_file=True, save_eda=True):
    """
    Full pipeline:
    1. Load tables from DB
    2. Preprocess data
    3. Merge tables into ML-ready dataset
    4. Generate features
    5. Optionally, run EDA and save plots
    6. Optionally, save processed features as CSV
    """
    print("📥 Loading tables from database...")
    dfs = load_tables()
    
    print("🧹 Preprocessing data...")
    dfs_clean = preprocess_data(dfs)
    
    print("🔗 Merging tables...")
    ml_ready = merge_tables(dfs_clean)
    
    print("⚙ Generating features...")
    features = generate_features(ml_ready)
    
    print("✅ Data processing pipeline completed!")

    if run_eda:
        print("📊 Running EDA...")

        # Summary statistics
        summary_statistics(features)

        # Trade outcomes
        if "outcome_encoded" in features.columns:
            fig = plot_trade_outcomes(features)
            if save_eda and fig:
                save_plot(fig, "trade_outcomes")

        # Trade duration
        if "trade_duration" in features.columns:
            fig = plot_trade_duration(features)
            if save_eda and fig:
                save_plot(fig, "trade_duration")

        # Emotions
        emotion_cols = [c for c in features.columns if c.startswith("emotion_")]
        if emotion_cols:
            fig = plot_emotions(features)
            if save_eda and fig:
                save_plot(fig, "emotions_distribution")

        # Correlation matrix
        numeric_cols = features.select_dtypes(include=["float64", "int64"]).columns
        if not numeric_cols.empty:
            fig = plot_correlation_matrix(features)
            if save_eda and fig:
                save_plot(fig, "correlation_matrix")

        print("📈 EDA completed and plots saved!")

    if save_file:
        features.to_csv(PROCESSED_PATH, index=False)
        print(f"💾 Processed & feature-engineered data saved to: {PROCESSED_PATH}")
    
    return features

if __name__ == "__main__":
    # Set run_eda=True to generate plots
    features = run_full_pipeline(run_eda=True, save_file=True, save_eda=True)
    print("First 5 rows of final feature-engineered data:")
    print(features.head())
