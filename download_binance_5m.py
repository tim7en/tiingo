#!/usr/bin/env python3
"""Download 5-minute kline data from Binance for crypto assets and save as parquet.

Output format matches cache/{SYMBOL}_5m.parquet with columns: o, h, l, c, v, time, ts
"""

import os
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
FIVE_MIN_MS = 5 * 60 * 1000
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def safe_request_json(url: str, params: dict | None = None, retries: int = 6):
    backoff = 1.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (418, 429, 500, 502, 503, 504):
                wait = min(backoff * 2.0, 20.0)
                print(f"  HTTP {resp.status_code}, retrying in {wait:.0f}s...", flush=True)
                time.sleep(wait)
                backoff = wait
                continue
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 20.0)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def fetch_klines(symbol: str, start_ms: int, end_ms: int, sleep_sec: float = 0.05) -> pd.DataFrame:
    """Fetch all 5m klines between start_ms and end_ms from Binance."""
    rows = []
    cursor = start_ms
    batch_count = 0
    start_dt = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_dt = pd.to_datetime(end_ms, unit="ms", utc=True)
    print(f"  Fetching {symbol}: {start_dt.date()} -> {end_dt.date()}", flush=True)

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": 1000,
        }
        payload = safe_request_json(BINANCE_KLINES_URL, params=params)
        if not payload:
            break
        if isinstance(payload, dict) and payload.get("code"):
            raise RuntimeError(f"Binance error for {symbol}: {payload}")

        rows.extend(payload)
        batch_count += 1
        if batch_count % 50 == 0:
            latest_dt = pd.to_datetime(int(payload[-1][0]), unit="ms", utc=True)
            print(f"    batches={batch_count} bars={len(rows):,} up_to={latest_dt}", flush=True)

        last_open = int(payload[-1][0])
        next_cursor = last_open + FIVE_MIN_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(sleep_sec)

    if not rows:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v", "time", "ts"])

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
        ],
    )

    # Convert to match project parquet format: o, h, l, c, v, time, ts
    df["ts"] = df["open_time"].astype(np.int64)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    df = df[["o", "h", "l", "c", "v", "time", "ts"]].copy()
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def main():
    # Symbols: label -> Binance trading pair
    symbols = {
        "LTC": "LTCUSDT",
        "ENJ": "ENJUSDT",
        "BCH": "BCHUSDT",
        "ATOM": "ATOMUSDT",
    }

    start_date = date(2022, 1, 1)
    end_date = date(2026, 3, 30)

    start_ms = int(datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc).timestamp() * 1000)

    os.makedirs(CACHE_DIR, exist_ok=True)

    for label, symbol in symbols.items():
        out_path = os.path.join(CACHE_DIR, f"{label}_5m.parquet")
        print(f"\n{'='*60}")
        print(f"Downloading {label} ({symbol}) 5m data: {start_date} -> {end_date}")
        print(f"{'='*60}")

        df = fetch_klines(symbol, start_ms, end_ms)

        if len(df) > 0:
            df.to_parquet(out_path, index=False)
            print(f"\n  Saved {len(df):,} rows to {out_path}")
            print(f"  Date range: {df['time'].min()} -> {df['time'].max()}")
            print(f"  Columns: {list(df.columns)}")
        else:
            print(f"\n  NO DATA returned for {label}")


if __name__ == "__main__":
    main()
