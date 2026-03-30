import os
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

def save_model(model, project_id, gcs_path):
    local_path = "model.joblib"
    joblib.dump(model, local_path)
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

def main():
    print('starting traininggggsgggghhdd')
    config = load_config()
    project_id = os.environ.get("project_id")
    output_path = os.environ['AIP_MODEL_DIR']
    data_path = config.get("data_path")
    bucket = config.get("bucket")

    df = pd.read_csv(f"/gcs/{bucket}/{data_path}")
    df = df.dropna()

    # # features
    # features = [
    #     "Open", "High", "Low", "Close", "Volume",
    #     "daily_return_pct",
    #     "daily_exposure_count", "daily_avg_tone",
    #     "day_of_week"
    # ]

    # results = {}
    # features
    price_features = [
        "Open", "High", "Low", "Close", "Volume",
        "daily_return_pct",
        "day_of_week"
    ]
    
    sentiment_features = [
        "daily_exposure_count", "daily_avg_tone"
    ]
    
    results = {}

#     for ticker in df["ticker"].unique():
#         print(f"\nTraining for {ticker}")

#         sub_df = df[df["ticker"] == ticker].sort_values("event_date")

#         X = sub_df[features]
#         y = sub_df["next_day_close"]

#         split = int(len(sub_df) * 0.8)
#         X_train, X_test = X.iloc[:split], X.iloc[split:]
#         y_train, y_test = y.iloc[:split], y.iloc[split:]

#         model = RandomForestRegressor()
#         model.fit(X_train, y_train)

#         # model prediction
#         y_pred = model.predict(X_test)

#         # baseline
#         baseline_pred = X_test["Close"]

#         # evaluate
#         mae = mean_absolute_error(y_test, y_pred)
#         baseline_mae = mean_absolute_error(y_test, baseline_pred)

#         print(f"{ticker} Model MAE:", mae)
#         print(f"{ticker} Baseline MAE:", baseline_mae)

#         results[ticker] = {
#         "model_mae": mae,
#         "baseline_mae": baseline_mae
#     }
#     print("\nFinal Results:", results)
#     save_model(model, project_id, output_path)



# if __name__ == "__main__":
#     main()
    for ticker in df["ticker"].unique():
        print(f"\n===== {ticker} =====")
    
        sub_df = df[df["ticker"] == ticker].sort_values("event_date")
    
        y = sub_df["next_day_close"]
    
        split = int(len(sub_df) * 0.8)
    
        y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    
        # baseline
    
        baseline_pred = sub_df["Close"].iloc[split:]
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
    
    
        # Linear Regression (price only)
     
        X_price = sub_df[price_features]
        X_train_p, X_test_p = X_price.iloc[:split], X_price.iloc[split:]
    
        lr = LinearRegression()
        lr.fit(X_train_p, y_train)
    
        y_lr_pred = lr.predict(X_test_p)
        lr_mae = mean_absolute_error(y_test, y_lr_pred)
    
    
        # Random Forest (price only)
    
        rf_price = RandomForestRegressor(random_state=42)
        rf_price.fit(X_train_p, y_train)
    
        y_rf_price_pred = rf_price.predict(X_test_p)
        rf_price_mae = mean_absolute_error(y_test, y_rf_price_pred)
    
    
        # Random Forest (price + sentiment)
    
        X_sent = sub_df[price_features + sentiment_features]
        X_train_s, X_test_s = X_sent.iloc[:split], X_sent.iloc[split:]
    
        rf_sent = RandomForestRegressor(random_state=42)
        rf_sent.fit(X_train_s, y_train)
    
        y_rf_sent_pred = rf_sent.predict(X_test_s)
        rf_sent_mae = mean_absolute_error(y_test, y_rf_sent_pred)
    
    
        # Print results
    
        print(f"Baseline MAE: {baseline_mae:.4f}")
        print(f"Linear Regression MAE: {lr_mae:.4f}")
        print(f"RF (price) MAE: {rf_price_mae:.4f}")
        print(f"RF (price+sentiment) MAE: {rf_sent_mae:.4f}")
    
    
        # Store results
    
        results[ticker] = {
            "baseline_mae": baseline_mae,
            "linear_regression_mae": lr_mae,
            "rf_price_mae": rf_price_mae,
            "rf_price_sentiment_mae": rf_sent_mae
        }
    
    print("\n===== FINAL RESULTS =====")
    print(results)
