import pandas as pd
import joblib

from utils import load_data, ensure_dir

OUTPUT_DIR = "outputs"
DATA_PATH = "data/creditcard.csv"
TOP_N = 200


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    # Load saved model & scaler
    model = joblib.load(f"{OUTPUT_DIR}/model.joblib")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.joblib")
    print("Loaded model and scaler from outputs/")

    # Load dataset
    df = load_data(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Predict
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]

    # Build ranked alerts
    alerts = pd.DataFrame({
        "index": X.index,
        "probability": probs,
        "true_label": y.values,
        "Time": X["Time"].values,
        "Amount": X["Amount"].values,
    })
    alerts = alerts.sort_values("probability", ascending=False).head(TOP_N)
    alerts.to_csv(f"{OUTPUT_DIR}/top_alerts.csv", index=False)
    print(f"Saved top {TOP_N} alerts to {OUTPUT_DIR}/top_alerts.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
