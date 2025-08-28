# data_processing/run_preprocess.py

import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import from correct package paths
from ai_project.data_processing.preprocess import load_tables, preprocess_data, merge_tables
from ai_project.data_processing.feature_engineering import generate_features
from ai_project.data_processing.eda import (
    summary_statistics,
    plot_trade_outcomes,
    plot_trade_duration,
    plot_emotions,
    plot_correlation_matrix,
)
from ai_project.database.save_features import save_features


# ==============================
# Setup Paths
# ==============================
SAVE_FOLDER = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
PROCESSED_FILENAME = f"processed_features_{timestamp}.csv"
PROCESSED_PATH = os.path.join(SAVE_FOLDER, PROCESSED_FILENAME)

EDA_FOLDER = os.path.join(SAVE_FOLDER, "eda_outputs")
os.makedirs(EDA_FOLDER, exist_ok=True)


def save_plot(fig, name: str):
    """Save matplotlib figure to EDA folder with timestamp"""
    if fig is None:
        return
    path = os.path.join(EDA_FOLDER, f"{name}_{timestamp}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved EDA plot → {path}")


def run_full_pipeline(run_eda: bool = False, save_file: bool = True, save_eda: bool = True):
    """
    Full pipeline:
    1. Load tables from DB
    2. Preprocess data
    3. Merge into ML-ready dataset
    4. Feature engineering
    5. Optional: run EDA
    6. Optional: save features as CSV
    """
    print("📥 Step 1: Loading tables from database...")
    dfs = load_tables()

    print("🧹 Step 2: Preprocessing data...")
    dfs_clean = preprocess_data(dfs)

    print("🔗 Step 3: Merging tables...")
    ml_ready = merge_tables(dfs_clean)

    print("⚙ Step 4: Generating features...")
    features = generate_features(ml_ready)

    print("✅ Pipeline completed successfully!")

    # Run Exploratory Data Analysis (EDA)
    if run_eda:
        print("📊 Step 5: Running EDA...")

        summary_statistics(features)

        if "outcome_encoded" in features.columns:
            save_plot(plot_trade_outcomes(features), "trade_outcomes")

        if "trade_duration" in features.columns:
            save_plot(plot_trade_duration(features), "trade_duration")

        emotion_cols = [c for c in features.columns if c.startswith("emotion_")]
        if emotion_cols:
            save_plot(plot_emotions(features), "emotions_distribution")

        numeric_cols = features.select_dtypes(include=["float64", "int64"]).columns
        if not numeric_cols.empty:
            save_plot(plot_correlation_matrix(features), "correlation_matrix")

        print("📈 EDA completed and plots saved!")

    # Save processed data to CSV
    if save_file:
        features.to_csv(PROCESSED_PATH, index=False)
        print(f"💾 Processed features saved → {PROCESSED_PATH}")

    return features


if __name__ == "__main__":
    # Run the full pipeline
    features = run_full_pipeline(run_eda=True, save_file=True, save_eda=True)

    # Save features to database
    save_features(features)

    print("\n🔎 First 5 rows of final feature-engineered data:")
    print(features.head())
