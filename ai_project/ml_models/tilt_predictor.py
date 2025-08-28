# ml_models/tilt_predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_tilt_predictor(df: pd.DataFrame):
    """
    Train a simple tilt/risk classifier from features
    Label: outcome_encoded (1=win, 0=loss)
    """
    if "outcome_encoded" not in df.columns:
        raise KeyError("❌ outcome_encoded missing in features")

    X = df.drop(columns=["outcome_encoded", "user_id", "id"], errors="ignore")
    y = df["outcome_encoded"]

    # ✅ Ensure all column names are clean strings
    X.columns = [str(col).replace(" ", "_").replace("'", "") for col in X.columns]

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Tilt Predictor Accuracy: {acc:.2f}")
    return model



if __name__ == "__main__":
    from ai_project.data_processing.preprocess import load_tables, preprocess_data, merge_tables
    from ai_project.data_processing.feature_engineering import generate_features

    dfs = load_tables()
    dfs_clean = preprocess_data(dfs)
    ml_ready = merge_tables(dfs_clean)
    features = generate_features(ml_ready)

    train_tilt_predictor(features)
