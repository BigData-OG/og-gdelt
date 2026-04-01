import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import json
import joblib
from google.cloud import storage

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_model(artifact, model_name, project_id, gcs_path):
    local_path = "model.joblib"
    joblib.dump(artifact, local_path)
    path_parts = gcs_path.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    prefix = "/".join(path_parts[1:])
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
        
    if not prefix.endswith("/"):
        prefix += "/"
    blob = bucket.blob(f"{prefix}{local_path}")
    blob.upload_from_filename(local_path)
    print(f"Model successfully uploaded to {gcs_path}{local_path}")


def train(ticker, bucket, input_path):
    print('starting trainings')
    project_id = os.environ.get("CLOUD_ML_PROJECT_ID")

    df = pd.read_csv(f"/gcs/{bucket}/{input_path}")
    df = df.dropna()
    output_path = os.environ.get('AIP_MODEL_DIR')

    price_features = [
        "Open", "High", "Low", "Close", "Volume",
        "daily_return_pct",
        "day_of_week"
    ]
    
    sentiment_features = [
        "daily_exposure_count", "daily_avg_tone"
    ]
    
    results = {}
    print(f"\n===== {ticker} ======ss")

    sub_df = df[df["ticker"] == ticker].sort_values("event_date")

    y = sub_df["next_day_close"]

    split = int(len(sub_df) * 0.8)

    y_train, y_test = y.iloc[:split], y.iloc[split:]


    # baseline

    baseline_pred = sub_df["Close"].iloc[split:]
    baseline_mae = mean_absolute_error(y_test, baseline_pred)


    # Random Forest (price + sentiment)

    X_sent = sub_df[price_features + sentiment_features]
    X_train_s, X_test_s = X_sent.iloc[:split], X_sent.iloc[split:]

    rf_sent = RandomForestRegressor(random_state=42)
    rf_sent.fit(X_train_s, y_train)

    y_rf_sent_pred = rf_sent.predict(X_test_s)
    rf_sent_mae = mean_absolute_error(y_test, y_rf_sent_pred)


    # Print results

    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"RF (price+sentiment) MAE: {rf_sent_mae:.4f}")


    # Store results

    results[ticker] = {
        "baseline_mae": baseline_mae,
        "rf_price_sentiment_mae": rf_sent_mae
    }
    
    print("\n===== FINAL RESULTS ======")
    print(results)

    save_model(rf_sent, ticker, project_id, output_path)

if __name__ == "__main__":
    ticker = sys.argv[1]
    bucket_name = sys.argv[2]
    input_path = sys.argv[3] if len(sys.argv) > 2 else None
    train(ticker, bucket_name, input_path)