"""Train a Logistic Regression model for credit-card fraud detection."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from utils import ensure_dir, load_data, save_json

# ── Configuration ────────────────────────────────────────────────────
DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N = 200

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    # 1) Load data
    df = load_data(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 2) Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Fraud in train: {y_train.sum()}  |  Fraud in test: {y_test.sum()}")

    # 3) Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 4) Train model
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_sc, y_train)
    print("Model trained.")

    # 5) Predict on test set
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # 6) Metrics
    metrics = {
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
    }
    print("Metrics:", metrics)
    save_json(metrics, f"{OUTPUT_DIR}/metrics.json")

    # 7) Save model & scaler
    joblib.dump(model, f"{OUTPUT_DIR}/model.joblib")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.joblib")
    print("Saved model.joblib and scaler.joblib")

    # 8) Top alerts (from test set)
    alerts = pd.DataFrame({
        "index": X_test.index,
        "probability": y_prob,
        "true_label": y_test.values,
        "Time": X_test["Time"].values,
        "Amount": X_test["Amount"].values,
    })
    alerts = alerts.sort_values("probability", ascending=False).head(TOP_N)
    alerts.to_csv(f"{OUTPUT_DIR}/top_alerts.csv", index=False)
    print(f"Saved top {TOP_N} alerts to {OUTPUT_DIR}/top_alerts.csv")

    print("\nDone! Check the outputs/ folder.")


if __name__ == "__main__":
    main()
