# =========================
# src/feature_utils.py
# =========================
import pandas as pd
import numpy as np
import requests

# NOTE:
# This file provides a lightweight way to fetch BTC close history at runtime
# (so you do NOT need to commit BitstampData.csv to GitHub).
# It uses CoinGecko's market_chart endpoint.

def get_bitcoin_historical_prices(days: int = 365) -> pd.DataFrame:
    """
    Fetch historical BTC prices in USD from CoinGecko.
    Returns a DataFrame with columns:
      - Date (datetime)
      - Close Price (USD) (float)
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": int(days)}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    prices = data.get("prices", [])
    if not prices:
        raise ValueError("No 'prices' returned from CoinGecko response.")

    df = pd.DataFrame(prices, columns=["timestamp", "Close Price (USD)"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop(columns=["timestamp"])
    df["Close Price (USD)"] = pd.to_numeric(df["Close Price (USD)"], errors="coerce")
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close Price (USD)"]]

def get_bitcoin_close_history(days: int = 365, tail_n: int = 300) -> pd.DataFrame:
    """
    Returns a DataFrame with a single column 'Close' containing a recent history window.
    This is designed to match HW3 pipelines that compute rolling/EMA indicators internally.
    """
    df = get_bitcoin_historical_prices(days=days).copy()
    df = df.rename(columns={"Close Price (USD)": "Close"})
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    if tail_n is not None:
        df = df.tail(int(tail_n))

    df = df.reset_index(drop=True)
    return df[["Close"]]


