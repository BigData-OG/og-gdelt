import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


def train_local_model(ticker: str, data_path: str):
    print(f'Starting local training for {ticker} using data from {data_path}')

    # Look for the file locally inside the container

    df = pd.read_csv(data_path)
    df = df.dropna()

    price_features = ["Open", "High", "Low", "Close", "Volume", "daily_return_pct", "day_of_week"]
    sentiment_features = ["daily_exposure_count", "daily_avg_tone"]

    # Filter for the ticker and sort chronologically
    sub_df = df[df["ticker"] == ticker].sort_values("event_date")

    if sub_df.empty:
        raise ValueError(f"No data found for {ticker} in {data_path}.")

    y = sub_df["next_day_close"]
    split = int(len(sub_df) * 0.8)
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train Random Forest
    X_sent = sub_df[price_features + sentiment_features]
    X_train_s, X_test_s = X_sent.iloc[:split], X_sent.iloc[split:]

    # Previously - too fast training
    rf_sent = RandomForestRegressor(random_state=42)
    rf_sent.fit(X_train_s, y_train)

    # ToDo - which approach is better?

    # Now
    # rf_sent = RandomForestRegressor(n_estimators=2000, random_state=42)
    # X_train_heavy = pd.concat([X_train_s] * 200, ignore_index=True)
    # y_train_heavy = pd.concat([y_train] * 200, ignore_index=True)
    # rf_sent.fit(X_train_heavy, y_train_heavy)

    # Calculate MAE for sanity checking
    y_rf_sent_pred = rf_sent.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_rf_sent_pred)

    # Save the model locally (this mimics deploying a new model)
    model_filename = f"{ticker}_model.joblib"
    joblib.dump(rf_sent, model_filename)

    return {"status": "success", "ticker": ticker, "mae": mae}
