# ml_models/ml_orchestrator.py
import argparse
from ai_project.data_processing.preprocess import load_tables, preprocess_data, merge_tables
from ai_project.data_processing.feature_engineering import generate_features

from ai_project.ml_models.tilt_predictor import train_tilt_predictor
from ai_project.ml_models.recovery_agent import suggest_recovery_plan
from ai_project.ml_models.clustering import cluster_traders
from ai_project.database.save_prediction import save_prediction

def main(run_tilt=True, run_recovery=True, run_clustering=True):
    print("🔁 Running full ML pipeline...")

    # Data prep
    dfs = load_tables()
    dfs_clean = preprocess_data(dfs)
    ml_ready = merge_tables(dfs_clean)
    features = generate_features(ml_ready)
    print("✅ Features prepared.")

    # Tilt predictor
    model = None
    if run_tilt:
        print("\n▶ Training tilt predictor...")
        model = train_tilt_predictor(features)

        # Save tilt predictions to DB
        if "tilt_prediction" in features.columns:
            for idx, row in features.iterrows():
                save_prediction(
                    user_id=row.get("user_id"),
                    trade_id=row.get("trade_id"),
                    model_name="tilt_predictor",
                    model_version="v1",
                    prediction=row.get("tilt_prediction")
                )

    # Recovery plan suggestion
    plan = None
    if run_recovery:
        print("\n▶ Suggesting recovery plan...")
        outcomes = list(features["outcome_encoded"].tail(10))
        plan = suggest_recovery_plan(outcomes)
        print(f"  Suggested recovery plan: {plan}")

        # Save recovery plan to DB as a pseudo-prediction
        save_prediction(
            user_id=None,
            trade_id=None,
            model_name="recovery_agent",
            model_version="v1",
            prediction=0.0  # Optionally store numeric score or leave as placeholder
        )

    # Clustering
    clustered_df, clustering_model = None, None
    if run_clustering:
        print("\n▶ Performing clustering...")
        clustered_df, clustering_model = cluster_traders(features)
        print("  Cluster labels added to DataFrame.")

        # Save cluster assignments to DB
        if "cluster_label" in clustered_df.columns:
            for idx, row in clustered_df.iterrows():
                save_prediction(
                    user_id=row.get("user_id"),
                    trade_id=row.get("trade_id"),
                    model_name="clustering_model",
                    model_version="v1",
                    prediction=row.get("cluster_label")
                )

    print("\n🎉 ML pipeline completed.")
    return model, plan, clustered_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML tasks on trade data")
    parser.add_argument("--tilt", action="store_true", help="Train tilt predictor")
    parser.add_argument("--recovery", action="store_true", help="Suggest a recovery plan")
    parser.add_argument("--cluster", action="store_true", help="Run clustering analysis")
    parser.add_argument("--all", action="store_true", help="Run all tasks (default)")

    args = parser.parse_args()
    run_tilt = args.all or args.tilt
    run_recovery = args.all or args.recovery
    run_clustering = args.all or args.cluster

    main(run_tilt=run_tilt, run_recovery=run_recovery, run_clustering=run_clustering)
