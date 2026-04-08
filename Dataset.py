import pandas as pd
import numpy as np
import yfinance as yf

tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "LLY", "ABBV", "BAC",
    "KO", "MRK", "PEP", "AVGO", "COST", "WMT", "TMO", "MCD", "CSCO", "ADBE",
    "CRM", "ACN", "DHR", "ABT", "VZ", "LIN", "NFLX", "CMCSA", "TXN", "PFE",
    "NKE", "DIS", "INTC", "AMD", "QCOM", "PM", "UNP", "HON", "LOW", "IBM",
    "INTU", "AMGN", "CAT", "GS", "SPGI", "BLK", "NOW", "MDT", "GE", "AMAT",
    "ISRG", "BKNG", "AXP", "DE", "SYK", "TJX", "ADP", "PLD", "LMT", "MO",
    "VRTX", "GILD", "MMC", "CI", "CB", "ZTS", "C", "T", "SO", "BDX",
    "DUK", "USB", "CL", "SCHW", "PNC", "TMUS", "ELV", "REGN", "MU", "ADI",
    "ETN", "SLB", "EOG", "AON", "COP", "CSX", "ITW", "NSC", "ICE", "FIS"
]

start_date = "2020-01-01"
end_date = "2026-01-01"

# Download market data
spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True, progress=False)
spy = spy.rename(columns=str.lower)
spy["market_ret_5d"] = spy["close"].pct_change(5)
spy_feature = spy[["market_ret_5d"]].copy()

all_dfs = []
failed_tickers = []

for ticker in tickers:
    print(f"Downloading {ticker}...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

        # Skip empty downloads
        if df.empty:
            print(f"Skipping {ticker}: no data returned")
            failed_tickers.append(ticker)
            continue

        df = df.rename(columns=str.lower)

        # Features
        df["ret_1d"] = df["close"].pct_change(1)
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_20d"] = df["close"].pct_change(20)
        df["vol_20d"] = df["ret_1d"].rolling(20).std()
        df["vol_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
        ma20 = df["close"].rolling(20).mean()
        df["ma_gap_20d"] = df["close"] / ma20 - 1
        df["high_low_range_1d"] = (df["high"] - df["low"]) / df["close"]
        df["ret_10d"] = df["close"].pct_change(10)
        df["ret_60d"] = df["close"].pct_change(60)
        df["vol_5d"] = df["ret_1d"].rolling(5).std()
        df["vol_60d"] = df["ret_1d"].rolling(60).std()
        ma5 = df["close"].rolling(5).mean()
        ma60 = df["close"].rolling(60).mean()
        df["ma_gap_5d"] = df["close"] / ma5 - 1
        df["ma_gap_60d"] = df["close"] / ma60 - 1
        vol_mean_20 = df["volume"].rolling(20).mean()
        vol_std_20 = df["volume"].rolling(20).std()
        df["volume_z_20d"] = (df["volume"] - vol_mean_20) / vol_std_20
        df["range_5d_avg"] = ((df["high"] - df["low"]) / df["close"]).rolling(5).mean()

        # Target
        df["target_fwd_5d"] = df["close"].shift(-5) / df["close"] - 1
        

        # Add ticker
        df["ticker"] = ticker

        # Merge market feature
        df = df.merge(spy_feature, left_index=True, right_index=True, how="left")

        # Remove missing rows
        df = df.dropna()

        # Make date a normal column
        df = df.reset_index().rename(columns={"Date": "date", "index": "date"})

        # Keep only final columns
        final_cols = [
            "date",
            "ticker",
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "vol_ratio_20d",
            "ma_gap_20d",
            "high_low_range_1d",
            # "market_ret_5d",
            "ret_10d",
            "ret_60d",
            "vol_5d",
            "vol_60d",
            "ma_gap_5d",
            "ma_gap_60d",
            "volume_z_20d",
            "range_5d_avg",
            "target_fwd_5d",
        ]

        df = df[final_cols]
        all_dfs.append(df)

    except Exception as e:
        print(f"Skipping {ticker}: {e}")
        failed_tickers.append(ticker)

dataset = pd.concat(all_dfs, ignore_index=True)
dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
dataset["target_up_5d"] = (dataset["target_fwd_5d"] > 0).astype(int)

print(dataset.head())
print(dataset.shape)
print("Failed tickers:", failed_tickers)

# save dataset to csv
dataset.to_csv("sp100_dataset.csv", index=False)

