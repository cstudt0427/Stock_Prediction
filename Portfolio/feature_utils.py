"""
feature_utils.py
----------------
Helper utilities shared between the Jupyter notebook and the Streamlit app.

Provides:
  - extract_features()   : loads / computes a reference DataFrame of features
                           so the Streamlit app can show sensible default values
  - get_feature_bounds() : returns (min, max, default) per feature column

Place this file at  src/feature_utils.py  (same level as Custom_Classes.py).
"""

import os
import json
import numpy as np
import pandas as pd

# ── Optional: import custom classes only when available ──────────────────────
try:
    from src.Custom_Classes import FeatureEngineer, PairFeatureEngineer
except ImportError:
    FeatureEngineer      = None
    PairFeatureEngineer  = None


# ---------------------------------------------------------------------------
# Constants – must match what the notebook used
# ---------------------------------------------------------------------------
TICKER         = "AAPL"
DATA_PATH      = "stock_dataset_2010_2018.csv"
SP500_PATH     = "SP500Data_2010_2018.csv"
FEATURE_META   = "feature_meta.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features() -> pd.DataFrame:
    """
    Return a DataFrame whose columns are the model's input features.

    The Streamlit app uses this to:
      1. Build the input form (one widget per column).
      2. Provide sensible default / min / max values.

    If the data files are unavailable the function returns a small placeholder
    DataFrame so the app can still render.
    """
    # ── Try loading feature metadata written by the notebook ─────────────────
    feature_keys = _load_feature_keys()

    # ── Try building features from the raw CSVs ──────────────────────────────
    try:
        df = _build_feature_df(feature_keys)
        return df
    except Exception as exc:
        print(f"[feature_utils] Could not load CSVs ({exc}). Using placeholder.")
        return _placeholder_df(feature_keys)


def get_feature_bounds(df: pd.DataFrame) -> dict:
    """
    Return {col: (min, max, default)} for every numeric column in *df*.
    Used to configure Streamlit number_input widgets.
    """
    bounds = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min  = float(np.nanmin(df[col]))
        col_max  = float(np.nanmax(df[col]))
        col_mean = float(np.nanmean(df[col]))
        bounds[col] = (
            round(col_min,  4),
            round(col_max,  4),
            round(col_mean, 4),
        )
    return bounds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_feature_keys() -> list:
    """Read feature column names from the JSON file written by the notebook."""
    if os.path.exists(FEATURE_META):
        with open(FEATURE_META) as f:
            meta = json.load(f)
        return meta.get("feature_keys", [])
    # Fallback: minimal set based on the sentiment strategy
    return [
        "sentiment_textblob", "sentiment_LSTM", "sentiment_lex",
        "sentiment_textblob_lag1", "sentiment_LSTM_lag1", "sentiment_lex_lag1",
        "sentiment_textblob_lag2", "sentiment_LSTM_lag2", "sentiment_lex_lag2",
    ]


def _build_feature_df(feature_keys: list) -> pd.DataFrame:
    """
    Build a reference DataFrame from the raw CSVs.  Only columns that exist in
    *feature_keys* AND can be computed from the CSVs are included.
    """
    # Load prices
    prices  = pd.read_csv(DATA_PATH)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices  = prices.set_index("Date")
    returns = prices.pct_change().dropna(how="all")

    parts = []

    # Correlated-stock returns (GOOG, MSFT, AMZN, ADBE, JPM from correlation analysis)
    corr_cols = ["GOOG", "MSFT", "AMZN", "ADBE", "JPM"]
    available  = [c for c in corr_cols if c in returns.columns]
    if available:
        parts.append(returns[available])
        for c in available:
            parts.append(returns[[c]].rename(columns={c: f"{c}_lag1"}).shift(1))

    # Technical indicators for AAPL
    if FeatureEngineer is not None and TICKER in prices.columns:
        fe   = FeatureEngineer(windows=[5, 14, 20])
        tech = fe.transform(prices[[TICKER]])
        parts.append(tech.add_prefix("tech_"))

    combined = pd.concat(parts, axis=1).dropna(how="all")

    # Filter to only the keys the model actually uses
    available_keys = [k for k in feature_keys if k in combined.columns]
    if available_keys:
        return combined[available_keys].dropna()

    return combined


def _placeholder_df(feature_keys: list) -> pd.DataFrame:
    """Return a single-row zeros DataFrame with the correct column names."""
    return pd.DataFrame(
        np.zeros((1, len(feature_keys))),
        columns=feature_keys if feature_keys else ["feature_0"]
    )
