#!/usr/bin/env python3
"""Download 5 years of 5-minute kline data for SPY, QQQ, and copper (CPER) from Tiingo IEX."""

import requests
import pandas as pd
from datetime import datetime, timedelta, date as date_type
import time
import os

TIINGO_API_KEY = "34d03d1d1382e36010bdb817d2512a4bfa5585f3"
TIINGO_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}",
}
TIINGO_BASE = "https://api.tiingo.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _tiingo_get(url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=TIINGO_HEADERS, params=params, timeout=90)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"\n  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n  Retry {attempt+1}: {e}", flush=True)
                time.sleep(5)
            else:
                print(f"\n  FAILED: {e}", flush=True)
                return None
    return None


def fetch_iex_5m(ticker: str, start_date: date_type, end_date: date_type):
    """Fetch 5-minute bars from Tiingo IEX in 2-year chunks."""
    all_rows = []
    chunk_start = start_date
    chunk_size = timedelta(days=120)  # Smaller chunks to avoid 10k row cap
    total_chunks = ((end_date - start_date).days // chunk_size.days) + 1
    n = 0

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + chunk_size, end_date)
        url = f"{TIINGO_BASE}/iex/{ticker}/prices"
        params = {
            "startDate": chunk_start.isoformat(),
            "endDate": chunk_end.isoformat(),
            "resampleFreq": "5min",
            "columns": "open,high,low,close,volume",
        }
        n += 1
        print(f"  [{n}/{total_chunks}]", end="", flush=True)
        data = _tiingo_get(url, params)
        if data and isinstance(data, list) and len(data) > 0:
            all_rows.extend(data)
            print(f" +{len(data)} rows", end="", flush=True)
        else:
            print(" +0", end="", flush=True)
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(1)  # Be nice to API

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    col_map = {"date": "time_str", "open": "o", "high": "h", "low": "l",
               "close": "c", "volume": "v"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "time_str" not in df.columns:
        print(f"\n  No date column found. Columns: {list(df.columns)}")
        return None

    if "v" not in df.columns:
        df["v"] = 0.0

    df["time"] = pd.to_datetime(df["time_str"], utc=True)
    df["ts"] = df["time"].astype("int64") // 10**6
    df = df[["o", "h", "l", "c", "v", "time", "ts"]].copy()
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[(df["o"] > 0) | (df["c"] > 0)].reset_index(drop=True)
    return df


def main():
    end = datetime.now().date()
    start = end - timedelta(days=5 * 365)

    tickers = {
        "SPY": "spy",
        "QQQ": "qqq",
        "COPPER": "cper",  # United States Copper Index Fund ETF
    }

    os.makedirs(CACHE_DIR, exist_ok=True)

    for label, ticker in tickers.items():
        out_path = os.path.join(CACHE_DIR, f"{label}_5m.parquet")
        print(f"\n{'='*60}")
        print(f"Downloading {label} ({ticker}) 5m data: {start} -> {end}")
        print(f"{'='*60}")

        df = fetch_iex_5m(ticker, start, end)

        if df is not None and len(df) > 0:
            df.to_parquet(out_path, index=False)
            print(f"\n  Saved {len(df)} rows to {out_path}")
            print(f"  Date range: {df['time'].min()} -> {df['time'].max()}")
            print(f"  Columns: {list(df.columns)}")
        else:
            print(f"\n  NO DATA returned for {label}")


if __name__ == "__main__":
    main()
