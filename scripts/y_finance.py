import yfinance as yf
import pandas as pd

tickers = ["AMZN", "2222.SR", "PFE"]

# pull finance data - have a smaller buffer so that first dates are not empty (markets closed on NYE)
raw_data = yf.download(tickers, start="2019-12-20", end="2025-12-31")

# get what we need
opens = raw_data['Open']
closes = raw_data['Close']
midpoints = (opens + closes) / 2
pct_change = closes.pct_change() * 100

# combine
final_df = pd.concat([opens, closes, midpoints, pct_change], axis=1, 
                     keys=['Open', 'Close', 'Avg_Price', 'Pct_Change'])

# fill in weekend gaps
final_df = final_df.ffill()

# formatting
final_df = final_df.stack(level=1)

# clean up
final_df = final_df.reset_index()
final_df.rename(columns={'level_0': 'Date'}, inplace=True)

# filter as required
final_df = final_df[final_df['Date'] >= "2020-01-01"]

# save final version
final_df.to_csv("market_data_ready_for_join.csv", index=False)

print("done")
print(f"Format: {final_df.columns.tolist()}")
print(final_df.head(3))