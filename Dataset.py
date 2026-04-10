import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

SECTOR_CACHE_PATH = "ticker_sector_map.csv"


def load_or_build_sector_map(tickers):
    """
    Build a ticker -> sector mapping once and cache it locally.
    Falls back to 'Unknown' when sector is unavailable.
    """
    cache_file = Path(SECTOR_CACHE_PATH)

    if cache_file.exists():
        sector_df = pd.read_csv(cache_file)
        sector_map = dict(zip(sector_df["ticker"], sector_df["sector"]))
        return sector_map

    rows = []
    for ticker in tickers:
        sector = "Unknown"
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown") or "Unknown"
        except Exception:
            pass

        rows.append({"ticker": ticker, "sector": sector})

    sector_df = pd.DataFrame(rows)
    sector_df.to_csv(cache_file, index=False)

    return dict(zip(sector_df["ticker"], sector_df["sector"]))

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

sector_map = load_or_build_sector_map(tickers)

start_date = "2020-01-01"
end_date = "2026-01-01"

# Download market data
spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True, progress=False)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

spy.columns = spy.columns.str.lower()
# Core market features
spy["market_ret_1d"] = spy["close"].pct_change(1)
spy["market_ret_5d"] = spy["close"].pct_change(5)
spy["market_ret_20d"] = spy["close"].pct_change(20)

spy["market_vol_20d"] = spy["market_ret_1d"].rolling(20).std()
spy["market_ma_gap_20d"] = spy["close"] / spy["close"].rolling(20).mean() - 1
spy["market_breakout_20d"] = spy["close"] / spy["close"].rolling(20).max() - 1

spy_features = spy[
    [
        "market_ret_1d",
        "market_ret_5d",
        "market_ret_20d",
        "market_vol_20d",
        "market_ma_gap_20d",
        "market_breakout_20d",
    ]
].copy()

all_dfs = []
failed_tickers = []

for ticker in tickers:
    print(f"Downloading {ticker}...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

        if df.empty:
            print(f"Skipping {ticker}: no data returned")
            failed_tickers.append(ticker)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = df.columns.str.lower()

        # Features
        df["ret_1d"] = df["close"].pct_change(1)
        df["ret_3d"] = df["close"].pct_change(3)
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_10d"] = df["close"].pct_change(10)
        df["ret_20d"] = df["close"].pct_change(20)
        df["ret_60d"] = df["close"].pct_change(60)
        df["vol_5d"] = df["ret_1d"].rolling(5).std()
        df["vol_20d"] = df["ret_1d"].rolling(20).std()
        df["vol_60d"] = df["ret_1d"].rolling(60).std()
        df["vol_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
        
        df["sector"] = sector_map.get(ticker, "Unknown")

        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_ratio"] = df["close"] / df["open"] - 1
        df["high_open_ratio"] = df["high"] / df["open"] - 1
        df["low_open_ratio"] = df["low"] / df["open"] - 1
        df["close_high_ratio"] = df["close"] / df["high"] - 1
        df["close_low_ratio"] = df["close"] / df["low"] - 1

        ma5 = df["close"].rolling(5).mean()
        ma10 = df["close"].rolling(10).mean()
        ma20 = df["close"].rolling(20).mean()
        ma60 = df["close"].rolling(60).mean()

        df["ma_gap_5d"] = df["close"] / ma5 - 1
        df["ma_gap_10d"] = df["close"] / ma10 - 1
        df["ma_gap_20d"] = df["close"] / ma20 - 1
        df["ma_gap_60d"] = df["close"] / ma60 - 1
        
        vol_mean_20 = df["volume"].rolling(20).mean()
        vol_std_20 = df["volume"].rolling(20).std()
        df["volume_z_20d"] = (df["volume"] - vol_mean_20) / vol_std_20
        df["range_5d_avg"] = ((df["high"] - df["low"]) / df["close"]).rolling(5).mean()
        df["breakout_20d"] = df["close"] / df["close"].rolling(20).max() - 1

        # Target
        df["target_fwd_5d"] = df["close"].shift(-5) / df["close"] - 1

        # Add ticker
        df["ticker"] = ticker

        # Merge market feature
        df = df.merge(spy_features, left_index=True, right_index=True, how="left")

        # Remove missing rows
        df = df.dropna()

        # Make date a normal column
        df = df.reset_index().rename(columns={"Date": "date", "index": "date"})

        # Keep only final columns
        final_cols = [
            "date",
            "ticker",
            "sector",
            "ret_1d",
            "ret_3d",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            "ret_60d",
            "vol_5d",
            "vol_20d",
            "vol_60d",
            "vol_ratio_20d",
            "high_low_range",
            "close_open_ratio",
            "high_open_ratio",
            "low_open_ratio",
            "close_high_ratio",
            "close_low_ratio",
            "market_ret_1d",
            "market_ret_5d",
            "market_ret_20d",
            "market_vol_20d",
            "market_ma_gap_20d",
            "market_breakout_20d",
            "ma_gap_5d",
            "ma_gap_10d",
            "ma_gap_20d",
            "ma_gap_60d",
            "volume_z_20d",
            "range_5d_avg",
            "breakout_20d",
            "target_fwd_5d",
        ]

        df = df[final_cols]
        all_dfs.append(df)

    except Exception as e:
        print(f"Skipping {ticker}: {e}")
        failed_tickers.append(ticker)

dataset = pd.concat(all_dfs, ignore_index=True)
dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

sector_daily = (
    dataset.groupby(["date", "sector"], as_index=False)
    .agg(
        sector_ret_1d_mean=("ret_1d", "mean"),
        sector_ret_5d_mean=("ret_5d", "mean"),
        sector_ret_20d_mean=("ret_20d", "mean"),
        sector_vol_20d_mean=("vol_20d", "mean"),
        sector_breakout_20d_mean=("breakout_20d", "mean"),
        sector_volume_z_20d_mean=("volume_z_20d", "mean"),
        sector_size=("ticker", "count"),
    )
)

dataset = dataset.merge(
    sector_daily,
    on=["date", "sector"],
    how="left",
)

# ---------------------------
# Stock vs sector comparison features
# ---------------------------
dataset["ret_1d_vs_sector"] = dataset["ret_1d"] - dataset["sector_ret_1d_mean"]
dataset["ret_5d_vs_sector"] = dataset["ret_5d"] - dataset["sector_ret_5d_mean"]
dataset["ret_20d_vs_sector"] = dataset["ret_20d"] - dataset["sector_ret_20d_mean"]

dataset["vol_20d_vs_sector"] = dataset["vol_20d"] - dataset["sector_vol_20d_mean"]
dataset["breakout_20d_vs_sector"] = dataset["breakout_20d"] - dataset["sector_breakout_20d_mean"]
dataset["volume_z_20d_vs_sector"] = dataset["volume_z_20d"] - dataset["sector_volume_z_20d_mean"]

print(dataset.head())
print(dataset.shape)
print("Failed tickers:", failed_tickers)

# save dataset to csv
dataset.to_csv("sp100_dataset.csv", index=False)

