import yfinance as yf
import pandas as pd
 
# tickers
tickers = ["AMZN", "2222.SR", "PFE"]
 
# buffer start so that we have some data fro the start of 2020 (markets closed EoY)
BUFFER_START = "2019-12-20"
ACTUAL_START = "2020-01-01"
END_DATE = "2025-12-31"
 
# get raw data
print("Getting the data")
raw_data = yf.download(tickers, start=BUFFER_START, end=END_DATE)


# yFinance raw columns: Open, High, Low, Close, Adj Close, Volume
# edit format to make join easier
stacked = raw_data.stack(level=1, future_stack=True)
stacked = stacked.reset_index()


# rename first two columns to Date and Ticker
col_names = stacked.columns.tolist()
stacked.rename(columns={col_names[0]: 'Date', col_names[1]: 'Ticker'}, inplace=True)


# forward fill for weekends
stacked = stacked.sort_values(['Ticker', 'Date']).reset_index(drop=True)

for ticker in tickers:
    mask = stacked['Ticker'] == ticker
    stacked.loc[mask] = stacked.loc[mask].ffill()

stacked['Date'] = pd.to_datetime(stacked['Date'])
stacked = stacked[stacked['Date'] >= ACTUAL_START]
stacked['Date'] = stacked['Date'].dt.strftime('%Y-%m-%d')
 
join_keys = ['Date', 'Ticker']
price_cols = [c for c in stacked.columns if c not in join_keys]
stacked = stacked[join_keys + sorted(price_cols)]
 
# local save
output_file = "market_data_raw.csv"
stacked.to_csv(output_file, index=False)