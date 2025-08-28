# ml_models/clustering.py
import pandas as pd
from sklearn.cluster import KMeans


def cluster_traders(df: pd.DataFrame, n_clusters=3):
    """
    Cluster traders based on trading + emotion features
    """
    X = df.drop(columns=["outcome_encoded", "user_id", "id"], errors="ignore")

    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)

    df["cluster"] = clusters
    print("✅ Cluster distribution:")
    print(df["cluster"].value_counts())
    return df, model


if __name__ == "__main__":
    from ai_project.data_processing.preprocess import load_tables, preprocess_data, merge_tables
    from ai_project.data_processing.feature_engineering import generate_features

    dfs = load_tables()
    dfs_clean = preprocess_data(dfs)
    ml_ready = merge_tables(dfs_clean)
    features = generate_features(ml_ready)

    clustered_df, model = cluster_traders(features)
    print(clustered_df.head())
