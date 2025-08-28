# ml_models/ml_orchestrator.py
import argparse
from ai_project.data_processing.preprocess import load_tables, preprocess_data, merge_tables
from ai_project.data_processing.feature_engineering import generate_features

from ai_project.ml_models.tilt_predictor import train_tilt_predictor
from ai_project.ml_models.recovery_agent import suggest_recovery_plan
from ai_project.ml_models.clustering import cluster_traders


def main(run_tilt=True, run_recovery=True, run_clustering=True):
    print("🔁 Running full ML pipeline...")

    # ​​​ Data prep
    dfs = load_tables()
    dfs_clean = preprocess_data(dfs)
    ml_ready = merge_tables(dfs_clean)
    features = generate_features(ml_ready)
    print("✅ Features prepared.")

    # ​​​ Tilt predictor
    model = None
    if run_tilt:
        print("\n▶ Training tilt predictor...")
        model = train_tilt_predictor(features)

    # ​​​ Recovery plan suggestion
    if run_recovery:
        print("\n▶ Suggesting recovery plan...")
        # Assume features have 'outcome_encoded' in order; extract latest N outcomes:
        outcomes = list(features["outcome_encoded"].tail(10))
        plan = suggest_recovery_plan(outcomes)
        print(f"  Suggested recovery plan: {plan}")

    # ​​​ Clustering
    if run_clustering:
        print("\n▶ Performing clustering...")
        clustered_df, clustering_model = cluster_traders(features)
        print("  Cluster labels added to DataFrame.")

    print("\n🎉 ML pipeline completed.")
    return model, plan if run_recovery else None, clustered_df if run_clustering else None


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
