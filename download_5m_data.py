#!/usr/bin/env python3
"""Download long-range 5-minute bars from Tiingo into the local cache."""

import argparse
import os
import time
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import requests

TIINGO_API_KEY = "34d03d1d1382e36010bdb817d2512a4bfa5585f3"
TIINGO_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}",
}
TIINGO_BASE = "https://api.tiingo.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ROW_CAP = 10000
REQUEST_GAP_SEC = 75
_last_request_ts = 0.0


def _pace():
    global _last_request_ts
    now = time.time()
    elapsed = now - _last_request_ts
    if _last_request_ts > 0 and elapsed < REQUEST_GAP_SEC:
        wait = REQUEST_GAP_SEC - elapsed
        mins, secs = divmod(int(wait), 60)
        print(f"\n  Pacing requests for {mins}m{secs:02d}s...", flush=True)
        time.sleep(wait)
    _last_request_ts = time.time()

ASSET_CONFIG: Dict[str, Dict[str, str]] = {
    "SPY": {"ticker": "spy", "source": "iex", "start": "2021-03-12"},
    "QQQ": {"ticker": "qqq", "source": "iex", "start": "2021-03-12"},
    "COPPER": {"ticker": "cper", "source": "iex", "start": "2017-01-03"},
    # Tiingo IEX does not expose Brent futures directly; use the liquid Brent ETF proxy.
    "BRENT": {"ticker": "bno", "source": "iex", "start": "2021-03-30"},
    "UUP": {"ticker": "uup", "source": "iex", "start": "2017-01-03"},
    "XAU": {"ticker": "gld", "source": "iex", "start": "2017-01-03"},
    "XAG": {"ticker": "slv", "source": "iex", "start": "2017-01-03"},
    # Tiingo IEX does not expose front-month nat gas futures directly; use the liquid ETF proxy.
    "NATGAS": {"ticker": "ung", "source": "iex", "start": "2021-03-30"},
    "PAXG": {"ticker": "paxgusd", "source": "crypto", "start": "2021-03-30"},
    "XAUT": {"ticker": "xautusd", "source": "crypto", "start": "2021-03-30"},
    # Tiingo does not expose spot platinum/palladium directly via IEX; use liquid ETF proxies.
    "XPT": {"ticker": "pplt", "source": "iex", "start": "2021-03-30"},
    "BTC": {"ticker": "btcusd", "source": "crypto", "start": "2021-03-30"},
    "XPD": {"ticker": "pall", "source": "iex", "start": "2021-03-30"},
    "EWJ": {"ticker": "ewj", "source": "iex", "start": "2021-03-30"},
    "EWY": {"ticker": "ewy", "source": "iex", "start": "2021-03-30"},
    "ETH": {"ticker": "ethusd", "source": "crypto", "start": "2021-03-30"},
    "SOL": {"ticker": "solusd", "source": "crypto", "start": "2021-03-30"},
    "GOOGL": {"ticker": "googl", "source": "iex", "start": "2021-03-30"},
    "INTC": {"ticker": "intc", "source": "iex", "start": "2021-03-30"},
    "NVDA": {"ticker": "nvda", "source": "iex", "start": "2021-03-30"},
    "TSLA": {"ticker": "tsla", "source": "iex", "start": "2021-03-30"},
    "AMZN": {"ticker": "amzn", "source": "iex", "start": "2021-03-30"},
    "PLTR": {"ticker": "pltr", "source": "iex", "start": "2021-03-30"},
    "META": {"ticker": "meta", "source": "iex", "start": "2021-03-30"},
    "MSTR": {"ticker": "mstr", "source": "iex", "start": "2021-03-30"},
    # Circle began trading on the NYSE on 2025-06-05, so earlier requests are guaranteed empty.
    "CRCL": {"ticker": "crcl", "source": "iex", "start": "2025-06-05"},
    "HOOD": {"ticker": "hood", "source": "iex", "start": "2021-03-30"},
    "COIN": {"ticker": "coin", "source": "iex", "start": "2021-03-30"},
}


def _tiingo_get(url, params, max_retries=6):
    for attempt in range(max_retries):
        try:
            _pace()
            resp = requests.get(url, headers=TIINGO_HEADERS, params=params, timeout=90)
            if resp.status_code == 429:
                wait = min(60 * (attempt + 1), 600)
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


def _fetch_iex_chunk(ticker: str, start_date: date_type, end_date: date_type, depth: int = 0):
    url = f"{TIINGO_BASE}/iex/{ticker}/prices"
    params = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "resampleFreq": "5min",
        "columns": "open,high,low,close,volume",
    }
    data = _tiingo_get(url, params)
    if data is None:
        raise RuntimeError(f"Request failed for {ticker} {start_date} -> {end_date}")
    if not isinstance(data, list) or len(data) == 0:
        return []
    if len(data) < ROW_CAP or start_date >= end_date:
        return data

    span_days = (end_date - start_date).days
    if span_days <= 1:
        raise RuntimeError(
            f"Hit Tiingo row cap on a {span_days + 1}-day request for {ticker} "
            f"{start_date} -> {end_date}"
        )

    mid_date = start_date + timedelta(days=span_days // 2)
    print(
        f"\n    chunk capped at {len(data)} rows, splitting "
        f"{start_date} -> {end_date}",
        flush=True,
    )
    left = _fetch_iex_chunk(ticker, start_date, mid_date, depth + 1)
    right = _fetch_iex_chunk(ticker, mid_date + timedelta(days=1), end_date, depth + 1)
    return left + right


def _fetch_crypto_chunk(ticker: str, start_date: date_type, end_date: date_type):
    url = f"{TIINGO_BASE}/tiingo/crypto/prices"
    params = {
        "tickers": ticker,
        "startDate": f"{start_date.isoformat()}T00:00:00Z",
        "endDate": f"{end_date.isoformat()}T23:59:59Z",
        "resampleFreq": "5min",
    }
    data = _tiingo_get(url, params)
    if data is None:
        raise RuntimeError(f"Request failed for {ticker} {start_date} -> {end_date}")
    rows = []
    if not isinstance(data, list):
        return rows
    for entry in data:
        price_data = entry.get("priceData", [])
        if isinstance(price_data, list):
            rows.extend(price_data)
    return rows


def fetch_iex_5m(ticker: str, start_date: date_type, end_date: date_type):
    """Fetch 5-minute bars from Tiingo IEX while automatically splitting capped ranges."""
    all_rows = []
    chunk_start = start_date
    # This value has proven small enough to stay under Tiingo's 10k row cap for
    # active IEX names without triggering recursive split requests.
    chunk_size = timedelta(days=150)
    total_chunks = ((end_date - start_date).days // chunk_size.days) + 1
    n = 0

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + chunk_size, end_date)
        n += 1
        print(f"  [{n}/{total_chunks}]", end="", flush=True)
        data = _fetch_iex_chunk(ticker, chunk_start, chunk_end)
        if data:
            all_rows.extend(data)
            print(f" +{len(data)} rows", end="", flush=True)
        else:
            print(" +0", end="", flush=True)
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(1)

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


def fetch_crypto_5m(ticker: str, start_date: date_type, end_date: date_type):
    """Fetch 5-minute bars from Tiingo crypto in 180-day chunks."""
    all_rows = []
    chunk_start = start_date
    chunk_size = timedelta(days=180)
    total_chunks = ((end_date - start_date).days // chunk_size.days) + 1
    n = 0

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + chunk_size, end_date)
        n += 1
        print(f"  [{n}/{total_chunks}]", end="", flush=True)
        data = _fetch_crypto_chunk(ticker, chunk_start, chunk_end)
        if data:
            all_rows.extend(data)
            print(f" +{len(data)} rows", end="", flush=True)
        else:
            print(" +0", end="", flush=True)
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(1)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    col_map = {
        "date": "time_str",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
        "tradesDone": "v_trades",
        "volumeNotional": "v_notional",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "time_str" not in df.columns:
        print(f"\n  No date column found. Columns: {list(df.columns)}")
        return None

    if "v" not in df.columns:
        if "v_notional" in df.columns:
            df["v"] = df["v_notional"]
        elif "v_trades" in df.columns:
            df["v"] = df["v_trades"]
        else:
            df["v"] = 0.0

    df["time"] = pd.to_datetime(df["time_str"], utc=True)
    df["ts"] = df["time"].astype("int64") // 10**6
    df = df[["o", "h", "l", "c", "v", "time", "ts"]].copy()
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[(df["o"] > 0) | (df["c"] > 0)].reset_index(drop=True)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        nargs="*",
        help=f"Asset labels to download. Available: {', '.join(sorted(ASSET_CONFIG))}",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().date().isoformat(),
        help="End date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--request-gap-sec",
        type=float,
        default=REQUEST_GAP_SEC,
        help="Seconds to wait between Tiingo requests. Lower values are faster but may trigger rate limiting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip labels whose cache parquet already exists.",
    )
    return parser.parse_args()


def resolve_labels(raw_labels: List[str]) -> List[str]:
    if not raw_labels:
        return list(ASSET_CONFIG.keys())

    labels = []
    unknown = []
    for label in raw_labels:
        label_norm = label.upper()
        if label_norm in ASSET_CONFIG:
            labels.append(label_norm)
        else:
            unknown.append(label)

    if unknown:
        raise SystemExit(
            f"Unknown label(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(ASSET_CONFIG))}"
        )

    return labels


def main():
    args = parse_args()
    global REQUEST_GAP_SEC
    REQUEST_GAP_SEC = max(0.0, float(args.request_gap_sec))
    end = date_type.fromisoformat(args.end_date)
    labels = resolve_labels(args.labels)

    os.makedirs(CACHE_DIR, exist_ok=True)

    for label in labels:
        cfg = ASSET_CONFIG[label]
        ticker = cfg["ticker"]
        source = cfg["source"]
        start = date_type.fromisoformat(cfg["start"])
        out_path = os.path.join(CACHE_DIR, f"{label}_5m.parquet")
        if args.skip_existing and os.path.exists(out_path):
            print(f"\nSkipping {label}: cache already exists at {out_path}")
            continue
        print(f"\n{'='*60}")
        print(f"Downloading {label} ({ticker}, {source}) 5m data: {start} -> {end}")
        print(f"{'='*60}")

        if source == "crypto":
            df = fetch_crypto_5m(ticker, start, end)
        else:
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
