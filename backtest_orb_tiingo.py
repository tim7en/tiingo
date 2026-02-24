#!/usr/bin/env python3
"""
ORB BACKTEST — Tiingo API — Intraday 5min + 15min
==================================================
Backtests ORB (Opening Range Breakout) strategy using Tiingo IEX intraday data.
Runs for three periods: 1Y, 3Y, and 5Y to compare performance stability.

API  : Tiingo  (https://api.tiingo.com)
Data : IEX intraday (US stocks/ETFs), Crypto endpoint (SOL)
       GLD/SLV used as proxies for gold/silver futures.

Periods:
  1Y  — last 1 year of intraday 5min data
  3Y  — last 3 years
  5Y  — last 5 years

Variants per period:
  US_5min           — 5min OR at NYSE open
  US_15min          — 15min OR at NYSE open
  CombinedUS_5and15 — shared wallet: 5min + 15min OR same day

Run:  python backtest_orb_tiingo.py
"""

import sys, io, os, time as _time, json
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date as date_type
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from pathlib import Path

# ====================================================================
# TIINGO CONFIG
# ====================================================================
TIINGO_API_KEY = "34d03d1d1382e36010bdb817d2512a4bfa5585f3"
TIINGO_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}",
}
TIINGO_BASE = "https://api.tiingo.com"
CACHE_DIR   = Path("d:/tiingo/cache")

# ====================================================================
# PARAM SETS — same as S11
# ====================================================================
BASE_CFG = dict(
    starting_balance     = 100.0,
    max_leverage         = 10,
    commission_pct       = 0.05,

    tp1_mult             = 1.0,
    tp2_mult             = 2.0,
    tp1_close_frac       = 0.50,

    min_or_range_pct     = 0.10,
    max_or_range_pct     = 15.0,
    max_or_atr_mult      = 2.0,

    max_gap_pct          = 0,
    breakout_vol_mult    = 1.5,
    entry_cutoff_minutes = 90,
    breakeven_after_tp1  = True,

    sma_period           = 20,
    max_daily_trades     = 2,
    daily_risk_budget    = 7.5,
    kelly_lookback       = 20,
    kelly_fraction       = 0.5,
)

PARAM_SETS = {
    "S9": {
        **BASE_CFG,
        "risk_pct":       5.0,
        "vol_score_mult": 2.0,
        "min_score":      8,
        "label":          "S9 (min_score=8, vol=2.0x, risk=5%)",
    },
    "S10": {
        **BASE_CFG,
        "risk_pct":       7.0,
        "vol_score_mult": 1.5,
        "min_score":      7,
        "label":          "S10 (min_score=7, vol=1.5x, risk=7%)",
    },
    "S12": {
        **BASE_CFG,
        "risk_pct":       5.0,
        "vol_score_mult": 1.3,
        "min_score":      8,
        "label":          "S12 (min_score=8, vol=1.3x, risk=5%)",
    },
}

# ====================================================================
# BACKTEST PERIODS
# ====================================================================
PERIODS = {
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
}

# ====================================================================
# VARIANTS  (US_5min, US_15min)
# ====================================================================
VARIANTS = [
    {
        "name":             "US_5min",
        "description":      "US Market Open — 5min OR (NYSE 9:30 ET)",
        "or_duration_bars": 1,
        "session_type":     "us",
        "force_close_hour": 21,   # 16:00 ET = 21:00 UTC (summer)
    },
    {
        "name":             "US_15min",
        "description":      "US Market Open — 15min OR (NYSE 9:30 ET)",
        "or_duration_bars": 3,
        "session_type":     "us",
        "force_close_hour": 21,
    },
]

# ====================================================================
# ASSET UNIVERSE — Tiingo IEX for stocks/ETFs, Tiingo Crypto for SOL
# ====================================================================
ASSETS = [
    # US Equities (Tiingo IEX)
    dict(label="TSLA", tiingo_sym="tsla",  source="iex", tier=1),
    dict(label="COIN", tiingo_sym="coin",  source="iex", tier=1),
    dict(label="HOOD", tiingo_sym="hood",  source="iex", tier=1),
    dict(label="MSTR", tiingo_sym="mstr",  source="iex", tier=1),
    dict(label="AMZN", tiingo_sym="amzn",  source="iex", tier=2),
    dict(label="PLTR", tiingo_sym="pltr",  source="iex", tier=2),
    dict(label="INTC", tiingo_sym="intc",  source="iex", tier=2),
    # Metal ETF proxies (Tiingo IEX — GLD=gold, SLV=silver)
    dict(label="XAU",  tiingo_sym="gld",   source="iex", tier=3),
    dict(label="XAG",  tiingo_sym="slv",   source="iex", tier=3),
    # Crypto (Tiingo Crypto endpoint)
    dict(label="SOL",  tiingo_sym="solusd", source="crypto", tier=1),
]

ASSET_CLASS = {
    "TSLA": "equity", "COIN": "equity", "HOOD": "equity", "MSTR": "equity",
    "AMZN": "equity", "PLTR": "equity", "INTC": "equity",
    "XAU":  "metals", "XAG":  "metals",
    "SOL":  "crypto",
}

# ====================================================================
# DST-AWARE US MARKET OPEN — returns UTC hour of 9:30 ET
# ====================================================================
def _us_market_open_utc_hour(d) -> int:
    """Returns 13 (EDT/summer) or 14 (EST/winter) for 9:30 ET in UTC."""
    if isinstance(d, str):
        d = date_type.fromisoformat(d)
    year = d.year
    mar1 = date_type(year, 3, 1)
    edt_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = date_type(year, 11, 1)
    edt_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return 13 if edt_start <= d < edt_end else 14

# ====================================================================
# TIINGO DATA FETCHING  (rate-limit aware: 50 req/hr free tier)
# ====================================================================
_last_request_ts = 0.0   # global pacer — enforces minimum gap between requests
REQUEST_GAP_SEC  = 75    # 50 req/hr ≈ 1 per 72s; use 75s for safety margin
_request_count   = 0

def _pace():
    """Wait if needed so we don't exceed 50 requests/hour."""
    global _last_request_ts, _request_count
    now = _time.time()
    elapsed = now - _last_request_ts
    if _last_request_ts > 0 and elapsed < REQUEST_GAP_SEC:
        wait = REQUEST_GAP_SEC - elapsed
        mins, secs = divmod(int(wait), 60)
        print(f" [pacing {mins}m{secs:02d}s]", end="", flush=True)
        _time.sleep(wait)
    _last_request_ts = _time.time()
    _request_count += 1


def _tiingo_get(url, params=None, retries=5):
    """Make a GET request to Tiingo with retries and rate-limit handling."""
    _pace()  # enforce global request spacing
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=TIINGO_HEADERS, params=params, timeout=90)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = min(60 * (2 ** attempt), 600)  # 60s, 120s, 240s, 480s, 600s
                print(f" [429 rate-limited, wait {wait}s]", end="", flush=True)
                _time.sleep(wait)
                # Reset pacer so next attempt doesn't double-wait
                global _last_request_ts
                _last_request_ts = _time.time()
                continue
            elif resp.status_code == 404:
                return None
            else:
                print(f" [HTTP {resp.status_code}]", end="", flush=True)
                if attempt < retries - 1:
                    _time.sleep(10)
                    continue
                return None
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                _time.sleep(10)
                continue
            print(f" [err: {e}]", end="", flush=True)
            return None
    return None


def fetch_tiingo_iex_5m(ticker: str, start_date: date_type, end_date: date_type):
    """
    Fetch 5-minute intraday bars from Tiingo IEX endpoint in ~2-year chunks.
    Tiingo IEX typically allows large date ranges.
    Returns pd.DataFrame with columns: o, h, l, c, v, time, ts  (UTC).
    """
    all_rows = []
    chunk_start = start_date
    chunk_size  = timedelta(days=730)  # 2-year chunks → 3 requests for 5Y

    n_chunks = 0
    total_chunks = ((end_date - start_date).days // chunk_size.days) + 1
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + chunk_size, end_date)
        url = f"{TIINGO_BASE}/iex/{ticker}/prices"
        params = {
            "startDate":    chunk_start.isoformat(),
            "endDate":      chunk_end.isoformat(),
            "resampleFreq": "5min",
        }
        n_chunks += 1
        print(f" [{n_chunks}/{total_chunks}]", end="", flush=True)
        data = _tiingo_get(url, params)
        if data and isinstance(data, list) and len(data) > 0:
            all_rows.extend(data)
            print(f" +{len(data)}", end="", flush=True)
        else:
            print(" +0", end="", flush=True)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_rows:
        return None, "iex_empty"

    df = pd.DataFrame(all_rows)
    # Normalise columns
    col_map = {"date": "time_str", "open": "o", "high": "h", "low": "l",
               "close": "c", "volume": "v"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "time_str" not in df.columns:
        return None, "iex_no_date_col"

    # IEX 5min resampled data does NOT include volume — add synthetic v=1
    if "v" not in df.columns:
        df["v"] = 1.0

    df["time"] = pd.to_datetime(df["time_str"], utc=True)
    df["ts"]   = df["time"].astype("int64") // 10**6
    df = df[["o", "h", "l", "c", "v", "time", "ts"]].copy()
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    # Drop bars where OHLC are all zero (non-trading)
    df = df[(df["o"] > 0) | (df["c"] > 0)].reset_index(drop=True)
    return df, "tiingo_iex"


def fetch_tiingo_crypto_5m(ticker: str, start_date: date_type, end_date: date_type):
    """
    Fetch 5-minute crypto bars from Tiingo Crypto endpoint in ~180-day chunks.
    ticker should be like 'solusd', 'btcusd', etc.
    """
    all_rows = []
    chunk_start = start_date
    chunk_size  = timedelta(days=180)  # 6-month chunks → ~10 requests for 5Y

    n_chunks = 0
    total_chunks = ((end_date - start_date).days // chunk_size.days) + 1
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + chunk_size, end_date)
        url = f"{TIINGO_BASE}/tiingo/crypto/prices"
        params = {
            "tickers":      ticker,
            "startDate":    f"{chunk_start.isoformat()}T00:00:00Z",
            "endDate":      f"{chunk_end.isoformat()}T23:59:59Z",
            "resampleFreq": "5min",
        }
        n_chunks += 1
        print(f" [{n_chunks}/{total_chunks}]", end="", flush=True)
        data = _tiingo_get(url, params)
        if data and isinstance(data, list) and len(data) > 0:
            for entry in data:
                price_data = entry.get("priceData", [])
                if price_data:
                    all_rows.extend(price_data)
                    print(f" +{len(price_data)}", end="", flush=True)
        else:
            print(" +0", end="", flush=True)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_rows:
        return None, "crypto_empty"

    df = pd.DataFrame(all_rows)
    col_map = {"date": "time_str", "open": "o", "high": "h", "low": "l",
               "close": "c", "volume": "v",
               "tradesDone": "v_trades",
               "volumeNotional": "v_notional"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "time_str" not in df.columns:
        return None, "crypto_no_date_col"

    # Crypto endpoint: volume may be named differently
    if "v" not in df.columns:
        if "v_notional" in df.columns:
            df["v"] = df["v_notional"]
        elif "v_trades" in df.columns:
            df["v"] = df["v_trades"]
        else:
            df["v"] = 1.0

    df["time"] = pd.to_datetime(df["time_str"], utc=True)
    df["ts"]   = df["time"].astype("int64") // 10**6
    df = df[["o", "h", "l", "c", "v", "time", "ts"]].copy()
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[(df["o"] > 0) | (df["c"] > 0)].reset_index(drop=True)
    return df, "tiingo_crypto"


def fetch_tiingo_daily(ticker: str, start_date: date_type, end_date: date_type,
                       source: str = "iex", sma_period: int = 20):
    """
    Fetch daily OHLCV from Tiingo and compute SMA20 / ATR20.
    For IEX tickers uses /tiingo/daily endpoint.
    For crypto uses /tiingo/crypto/prices with 1day resample.
    Returns (sma_data, atr_data, close_data) dicts keyed by date.
    """
    sma_data, atr_data, close_data = {}, {}, {}

    # Extend start date back to have enough for SMA computation
    extended_start = start_date - timedelta(days=sma_period * 3)

    if source == "crypto":
        url = f"{TIINGO_BASE}/tiingo/crypto/prices"
        params = {
            "tickers":      ticker,
            "startDate":    f"{extended_start.isoformat()}T00:00:00Z",
            "endDate":      f"{end_date.isoformat()}T23:59:59Z",
            "resampleFreq": "1day",
        }
        data = _tiingo_get(url, params)
        if not data:
            return sma_data, atr_data, close_data
        rows = []
        for entry in data:
            rows.extend(entry.get("priceData", []))
        if not rows:
            return sma_data, atr_data, close_data
        raw = pd.DataFrame(rows)
        raw = raw.rename(columns={"date": "date_str", "open": "open",
                                  "high": "high", "low": "low",
                                  "close": "close", "volume": "volume"})
    else:
        url = f"{TIINGO_BASE}/tiingo/daily/{ticker}/prices"
        params = {
            "startDate": extended_start.isoformat(),
            "endDate":   end_date.isoformat(),
        }
        data = _tiingo_get(url, params)
        if not data:
            return sma_data, atr_data, close_data
        raw = pd.DataFrame(data)
        # Daily endpoint columns: date, close, high, low, open, volume, ...
        raw = raw.rename(columns={"date": "date_str"})

    if raw.empty or "date_str" not in raw.columns:
        return sma_data, atr_data, close_data

    for col in ["open", "high", "low", "close"]:
        if col not in raw.columns:
            return sma_data, atr_data, close_data

    raw["dt"] = pd.to_datetime(raw["date_str"], utc=True)
    raw = raw.sort_values("dt").reset_index(drop=True)

    raw["sma"]        = raw["close"].rolling(sma_period, min_periods=sma_period).mean()
    raw["prev_close"] = raw["close"].shift(1)

    def _tr(r):
        return max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            abs(r["low"]  - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        )
    raw["tr"]         = raw.apply(_tr, axis=1)
    raw["atr20"]      = raw["tr"].rolling(sma_period, min_periods=5).mean()
    raw["sma_prev"]   = raw["sma"].shift(1)
    raw["atr20_prev"] = raw["atr20"].shift(1)
    raw["close_prev"] = raw["close"].shift(1)

    for _, row in raw.dropna(subset=["sma_prev"]).iterrows():
        d = row["dt"].date()
        sma_data[d]   = {"sma": float(row["sma_prev"]), "close": float(row["close_prev"])}
        close_data[d] = float(row["close_prev"])
        if pd.notna(row["atr20_prev"]):
            atr_data[d] = float(row["atr20_prev"])

    return sma_data, atr_data, close_data


def compute_daily_from_5m(df_5m, sma_period=20):
    """Fallback: compute daily SMA/ATR by aggregating 5m bars."""
    sma_data, atr_data, close_data = {}, {}, {}
    if df_5m is None or df_5m.empty:
        return sma_data, atr_data, close_data
    df = df_5m.copy()
    df["date"] = df["time"].dt.date
    daily = df.groupby("date").agg(
        open=("o", "first"), high=("h", "max"),
        low=("l", "min"), close=("c", "last"),
    ).reset_index().sort_values("date").reset_index(drop=True)
    if len(daily) < 3:
        return sma_data, atr_data, close_data
    daily["sma"] = daily["close"].rolling(sma_period, min_periods=5).mean()
    daily["prev_close"] = daily["close"].shift(1)
    daily["tr"] = daily.apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            abs(r["low"]  - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        ), axis=1
    )
    daily["atr20"]      = daily["tr"].rolling(sma_period, min_periods=5).mean()
    daily["sma_prev"]   = daily["sma"].shift(1)
    daily["atr20_prev"] = daily["atr20"].shift(1)
    daily["close_prev"] = daily["close"].shift(1)
    for _, row in daily.dropna(subset=["sma_prev"]).iterrows():
        d = row["date"]
        sma_data[d]   = {"sma": float(row["sma_prev"]), "close": float(row["close_prev"])}
        close_data[d] = float(row["close_prev"])
        if pd.notna(row["atr20_prev"]):
            atr_data[d] = float(row["atr20_prev"])
    return sma_data, atr_data, close_data


def fetch_5m(asset, start_date: date_type, end_date: date_type):
    """Route to the right Tiingo endpoint based on asset source."""
    ticker = asset["tiingo_sym"]
    source = asset["source"]
    if source == "crypto":
        return fetch_tiingo_crypto_5m(ticker, start_date, end_date)
    else:
        return fetch_tiingo_iex_5m(ticker, start_date, end_date)

# ====================================================================
# CACHE LAYER — save/load 5m data to avoid re-fetching
# ====================================================================
def _cache_path(label: str) -> Path:
    return CACHE_DIR / f"{label}_5m.parquet"

def _cache_daily_path(label: str) -> Path:
    return CACHE_DIR / f"{label}_daily.json"

def save_5m_cache(label: str, df: pd.DataFrame):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(label)
    df.to_parquet(path, index=False, engine="pyarrow")

def load_5m_cache(label: str) -> Optional[pd.DataFrame]:
    path = _cache_path(label)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    except Exception:
        return None

def save_daily_cache(label: str, sma_data, atr_data, close_data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_daily_path(label)
    # Convert date keys to strings for JSON
    obj = {
        "sma": {str(k): v for k, v in sma_data.items()},
        "atr": {str(k): v for k, v in atr_data.items()},
        "close": {str(k): v for k, v in close_data.items()},
    }
    with open(path, "w") as f:
        json.dump(obj, f)

def load_daily_cache(label: str):
    path = _cache_daily_path(label)
    if not path.exists():
        return None, None, None
    try:
        with open(path) as f:
            obj = json.load(f)
        sma_data   = {date_type.fromisoformat(k): v for k, v in obj["sma"].items()}
        atr_data   = {date_type.fromisoformat(k): v for k, v in obj["atr"].items()}
        close_data = {date_type.fromisoformat(k): v for k, v in obj["close"].items()}
        return sma_data, atr_data, close_data
    except Exception:
        return None, None, None

# ====================================================================
# DAILY VOLUME — fetch from Tiingo Daily endpoint (IEX stocks only)
# ====================================================================
def _cache_daily_vol_path(label: str) -> Path:
    return CACHE_DIR / f"{label}_daily_vol.json"

def save_daily_vol_cache(label: str, vol_map: dict):
    """Save {date_str: (volume, vol_ma20)} to JSON."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_daily_vol_path(label)
    obj = {str(k): list(v) for k, v in vol_map.items()}
    with open(path, "w") as f:
        json.dump(obj, f)

def load_daily_vol_cache(label: str) -> Optional[dict]:
    """Load daily volume cache. Returns {date: (volume, vol_ma20)} or None."""
    path = _cache_daily_vol_path(label)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            obj = json.load(f)
        vol_map = {}
        for k, v in obj.items():
            d = date_type.fromisoformat(k)
            vol_map[d] = (v[0], v[1])
        return vol_map
    except Exception:
        return None

def fetch_tiingo_daily_volume(ticker: str, start_date: date_type, end_date: date_type,
                              ma_period: int = 20) -> dict:
    """
    Fetch daily OHLCV from Tiingo daily endpoint (single API call for full range).
    Returns {date: (volume, vol_ma20)} dict.
    The daily endpoint has real volume data even though IEX 5min does not.
    """
    url = f"{TIINGO_BASE}/tiingo/daily/{ticker}/prices"
    # Add buffer before start_date for MA warmup
    warmup_start = start_date - timedelta(days=ma_period * 2 + 30)
    params = {
        "startDate": warmup_start.isoformat(),
        "endDate":   end_date.isoformat(),
    }
    print(f" [daily vol]", end="", flush=True)
    data = _tiingo_get(url, params)
    if not data or not isinstance(data, list) or len(data) == 0:
        return {}

    rows = []
    for bar in data:
        d = bar.get("date", "")[:10]
        vol = bar.get("adjVolume", bar.get("volume", 0))
        if vol is None:
            vol = 0
        rows.append({"date": date_type.fromisoformat(d), "vol": float(vol)})

    ddf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    ddf["vol_ma"] = ddf["vol"].rolling(ma_period, min_periods=3).mean()

    vol_map = {}
    for _, row in ddf.iterrows():
        d = row["date"]
        vol_ma_val = float(row["vol_ma"]) if pd.notna(row["vol_ma"]) else None
        vol_map[d] = (float(row["vol"]), vol_ma_val)

    return vol_map

# ====================================================================
# SESSION BUILDER
# ====================================================================
def build_sessions_variant(df, asset, atr_data, close_data, variant,
                           daily_vol_override=None):
    """
    Build trading sessions from 5m bar data.
    daily_vol_override: optional {date: (volume, vol_ma)} dict from Tiingo daily
                        endpoint.  When provided, overrides the (broken) synthetic
                        volume derived from IEX 5min bars.
    """
    or_duration_bars = variant["or_duration_bars"]
    session_type     = variant["session_type"]
    force_close_hour = variant["force_close_hour"]

    df = df.copy()
    df["date"] = df["time"].dt.date

    if daily_vol_override:
        # Use real daily volume from Tiingo daily endpoint
        vol_map = daily_vol_override
    else:
        # Fallback: aggregate from 5m bars (works for crypto which has real vol)
        daily_vol = df.groupby("date")["v"].sum().reset_index()
        daily_vol.columns = ["date", "day_vol"]
        daily_vol["vol_ma"] = daily_vol["day_vol"].rolling(20, min_periods=3).mean()
        vol_map = dict(zip(daily_vol["date"],
                           zip(daily_vol["day_vol"], daily_vol["vol_ma"])))

    sessions      = []
    prev_or_range = None
    prev_close_5m = None

    for d, grp in df.groupby("date"):
        dow = pd.Timestamp(d).dayofweek
        if dow >= 5:
            continue

        if session_type == "us":
            day_slots = [("us", _us_market_open_utc_hour(d), 30, force_close_hour)]
        else:
            continue

        if not grp.empty:
            prev_close_5m = float(grp.iloc[-1]["c"])

        for slot_type, or_hour, or_minute, slot_close_hour in day_slots:
            or_mask = (grp["time"].dt.hour == or_hour) & (grp["time"].dt.minute == or_minute)
            or_start_rows = grp[or_mask]
            if or_start_rows.empty:
                continue

            or_start_idx = or_start_rows.index[0]
            loc_in_grp   = grp.index.get_loc(or_start_idx)

            or_bar_indices = []
            for i in range(or_duration_bars):
                if loc_in_grp + i < len(grp):
                    or_bar_indices.append(grp.index[loc_in_grp + i])
            if len(or_bar_indices) < or_duration_bars:
                continue

            or_bars  = grp.loc[or_bar_indices]
            or_high  = float(or_bars["h"].max())
            or_low   = float(or_bars["l"].min())
            or_open  = float(or_bars.iloc[0]["o"])
            or_close = float(or_bars.iloc[-1]["c"])
            or_range = or_high - or_low

            last_or_bar  = or_bars.iloc[-1]
            session_bars = grp[grp["ts"] > int(last_or_bar["ts"])].copy()
            session_bars = session_bars[
                session_bars["time"].dt.hour < slot_close_hour
            ].copy()
            session_bars = session_bars.reset_index(drop=True)

            if len(session_bars) < 2:
                continue

            dv, dv_ma   = vol_map.get(d, (0, None))
            prior_close = close_data.get(d, prev_close_5m)
            gap_pct     = (abs(or_open - prior_close) / prior_close * 100
                           if prior_close and prior_close > 0 else 0.0)
            atr20       = atr_data.get(d)

            _ocm = or_minute + (or_duration_bars * 5)
            or_close_hour_val = or_hour + _ocm // 60
            or_close_minute   = _ocm % 60

            sessions.append({
                "date":               d,
                "or_bars":            or_bars,
                "or_open":            or_open,
                "or_high":            or_high,
                "or_low":             or_low,
                "or_range":           or_range,
                "or_close":           or_close,
                "bars":               session_bars,
                "vol_above_avg":      (dv_ma is not None and dv_ma > 0),
                "day_vol":            dv,
                "day_vol_ma":         dv_ma,
                "range_expanding":    (prev_or_range is not None and or_range > prev_or_range),
                "label":              asset["label"],
                "tier":               asset["tier"],
                "gap_pct":            gap_pct,
                "atr20":              atr20,
                "or_hour_utc":        or_hour,
                "or_minute_utc":      or_minute,
                "or_duration_bars":   or_duration_bars,
                "or_close_hour":      or_close_hour_val,
                "or_close_minute":    or_close_minute,
                "session_type":       slot_type,
                "session_close_hour": slot_close_hour,
            })
            prev_or_range = or_range

    return sessions

# ====================================================================
# SCORING
# ====================================================================
def score_setup(sess, sma_data, direction, cfg):
    score   = 0
    reasons = []
    d       = sess["date"]
    sma_info= sma_data.get(d)
    or_close= sess["or_close"]
    or_high = sess["or_high"]
    or_low  = sess["or_low"]
    or_range= sess["or_range"]
    ref_px  = or_close if or_close > 0 else or_high
    range_pct = (or_range / ref_px * 100) if ref_px > 0 else 0

    # +3 SMA alignment
    if sma_info:
        sma_val   = sma_info["sma"]
        prev_close= sma_info["close"]
        if direction == "long" and prev_close > sma_val:
            score += 3; reasons.append("SMA+3")
        elif direction == "short" and prev_close < sma_val:
            score += 3; reasons.append("SMA+3")
        else:
            reasons.append("SMA=0(counter-trend)")
    else:
        reasons.append("SMA=?(no data)")

    # +2 Volume above average
    dv    = sess.get("day_vol", 0)
    dv_ma = sess.get("day_vol_ma")
    vol_above = (dv_ma is not None and dv_ma > 0
                 and dv >= cfg["vol_score_mult"] * dv_ma)
    if vol_above:
        score += 2; reasons.append("VOL+2")
    else:
        reasons.append("VOL=0")

    # +2 OR range sweet spot 0.3–4%
    if 0.3 <= range_pct <= 4.0:
        score += 2; reasons.append(f"RNG+2({range_pct:.1f}%)")
    elif 0.1 <= range_pct <= 8.0:
        score += 1; reasons.append(f"RNG+1({range_pct:.1f}%)")
    else:
        reasons.append(f"RNG=0({range_pct:.1f}%)")

    # +1 OR body bias
    if or_range > 0:
        body_pos = (or_close - or_low) / or_range
        if direction == "long" and body_pos >= 0.7:
            score += 1; reasons.append("BODY+1(bull)")
        elif direction == "short" and body_pos <= 0.3:
            score += 1; reasons.append("BODY+1(bear)")

    return score, reasons


def leverage_for_score(score, cfg):
    return min(cfg["max_leverage"], 5 + score)


def compute_kelly(recent_trades, cfg, fallback_pct=5.0):
    if len(recent_trades) < cfg["kelly_lookback"]:
        return fallback_pct
    wins   = [t for t in recent_trades if t.pnl > 0]
    losses = [t for t in recent_trades if t.pnl <= 0]
    if not wins or not losses:
        return fallback_pct
    wr    = len(wins) / len(recent_trades)
    avg_w = float(np.mean([abs(t.r_mult) for t in wins]))
    avg_l = float(np.mean([abs(t.r_mult) for t in losses]))
    if avg_w <= 0 or avg_l <= 0:
        return fallback_pct
    b     = avg_w / avg_l
    kelly = (wr * b - (1 - wr)) / b
    return max(1.0, min(10.0, kelly * cfg["kelly_fraction"] * 100))

# ====================================================================
# TRADE DATACLASS
# ====================================================================
@dataclass
class Trade:
    date:             object
    direction:        str
    label:            str
    entry:            float
    or_range:         float
    sl:               float
    tp1:              float
    tp2:              float
    size_full:        float
    balance_at_entry: float
    score:            int   = 0
    exit_price:       float = 0.0
    exit_reason:      str   = ""
    pnl:              float = 0.0
    r_mult:           float = 0.0
    tp1_hit:          bool  = False

# ====================================================================
# SIMULATION ENGINE
# ====================================================================
def _manage_trade(tr, high, low, close, balance, comm, cfg):
    if tr.exit_price != 0:
        return balance, False

    if tr.direction == "long":
        if not tr.tp1_hit and high >= tr.tp1:
            tr.tp1_hit = True
            frac = cfg["tp1_close_frac"]
            pnl  = (tr.tp1 - tr.entry) * tr.size_full * frac
            fee  = tr.tp1 * tr.size_full * frac * comm
            balance += pnl - fee
            if cfg.get("breakeven_after_tp1"):
                tr.sl = tr.entry
        if tr.tp1_hit and high >= tr.tp2:
            rem = tr.size_full * (1 - cfg["tp1_close_frac"])
            pnl = (tr.tp2 - tr.entry) * rem
            fee = tr.tp2 * rem * comm + tr.entry * tr.size_full * comm
            balance += pnl - fee
            tr.exit_price, tr.exit_reason = tr.tp2, "TP2"
            return balance, True
        if low <= tr.sl:
            rem  = tr.size_full * ((1 - cfg["tp1_close_frac"]) if tr.tp1_hit else 1.0)
            pnl  = (tr.sl - tr.entry) * rem
            fees = tr.sl * rem * comm
            if not tr.tp1_hit:
                fees += tr.entry * tr.size_full * comm
            balance += pnl - fees
            tr.exit_price  = tr.sl
            tr.exit_reason = ("BE" if cfg.get("breakeven_after_tp1") else "SL*") if tr.tp1_hit else "SL"
            return balance, True
    else:
        if not tr.tp1_hit and low <= tr.tp1:
            tr.tp1_hit = True
            frac = cfg["tp1_close_frac"]
            pnl  = (tr.entry - tr.tp1) * tr.size_full * frac
            fee  = tr.tp1 * tr.size_full * frac * comm
            balance += pnl - fee
            if cfg.get("breakeven_after_tp1"):
                tr.sl = tr.entry
        if tr.tp1_hit and low <= tr.tp2:
            rem = tr.size_full * (1 - cfg["tp1_close_frac"])
            pnl = (tr.entry - tr.tp2) * rem
            fee = tr.tp2 * rem * comm + tr.entry * tr.size_full * comm
            balance += pnl - fee
            tr.exit_price, tr.exit_reason = tr.tp2, "TP2"
            return balance, True
        if high >= tr.sl:
            rem  = tr.size_full * ((1 - cfg["tp1_close_frac"]) if tr.tp1_hit else 1.0)
            pnl  = (tr.entry - tr.sl) * rem
            fees = tr.sl * rem * comm
            if not tr.tp1_hit:
                fees += tr.entry * tr.size_full * comm
            balance += pnl - fees
            tr.exit_price  = tr.sl
            tr.exit_reason = ("BE" if cfg.get("breakeven_after_tp1") else "SL*") if tr.tp1_hit else "SL"
            return balance, True

    return balance, False


def _eod_close(tr, close, balance, comm, cfg):
    if tr.exit_price != 0:
        return balance
    rem  = tr.size_full * ((1 - cfg["tp1_close_frac"]) if tr.tp1_hit else 1.0)
    sign = 1 if tr.direction == "long" else -1
    pnl  = sign * (close - tr.entry) * rem
    fees = close * rem * comm
    if not tr.tp1_hit:
        fees += tr.entry * tr.size_full * comm
    balance += pnl - fees
    tr.exit_price, tr.exit_reason = close, "EOD"
    return balance


def simulate_session(sess, direction, balance, score, cfg, force_close_hour=21):
    comm     = cfg["commission_pct"] / 100
    lev      = leverage_for_score(score, cfg)
    rpct     = cfg["risk_pct"]
    or_range = sess["or_range"]
    or_high  = sess["or_high"]
    or_low   = sess["or_low"]
    bars     = sess["bars"]

    or_close_hour   = sess["or_close_hour"]
    or_close_minute = sess["or_close_minute"]
    cutoff_min_total= or_close_hour * 60 + or_close_minute + cfg["entry_cutoff_minutes"]
    cutoff_hour     = cutoff_min_total // 60
    cutoff_min      = cutoff_min_total % 60

    vols      = bars["v"].values.astype(float)
    active_tr = None

    for idx in range(len(bars)):
        row      = bars.iloc[idx]
        high     = float(row["h"])
        low      = float(row["l"])
        close    = float(row["c"])
        bar_time = row["time"]
        bar_vol  = float(row["v"])

        if active_tr and active_tr.exit_price == 0:
            balance, _ = _manage_trade(active_tr, high, low, close, balance, comm, cfg)

        if bar_time.hour >= force_close_hour:
            if active_tr and active_tr.exit_price == 0:
                balance = _eod_close(active_tr, close, balance, comm, cfg)
            break

        if active_tr is None:
            if cfg["entry_cutoff_minutes"] > 0:
                bar_min_total = bar_time.hour * 60 + bar_time.minute
                if bar_min_total >= cutoff_hour * 60 + cutoff_min:
                    continue

            vol_ok = True
            # Skip volume filter if data has synthetic volume (all 1.0)
            has_real_vol = float(np.max(vols)) > 1.5
            if has_real_vol and cfg["breakout_vol_mult"] > 0 and idx >= 5:
                prior_avg = float(np.mean(vols[max(0, idx-5):idx]))
                if prior_avg > 0:
                    vol_ok = bar_vol >= cfg["breakout_vol_mult"] * prior_avg

            if direction == "long" and high > or_high:
                if not vol_ok:
                    continue
                entry = or_high
                sl    = or_low
                tp1   = entry + cfg["tp1_mult"] * or_range
                tp2   = entry + cfg["tp2_mult"] * or_range
                rd    = entry - sl
                if rd > 0:
                    size = min((balance * rpct / 100) / rd, balance * lev / entry)
                    if size > 0:
                        active_tr = Trade(
                            sess["date"], "long", sess["label"],
                            entry, or_range, sl, tp1, tp2, size, balance, score,
                        )

            elif direction == "short" and low < or_low:
                if not vol_ok:
                    continue
                entry = or_low
                sl    = or_high
                tp1   = entry - cfg["tp1_mult"] * or_range
                tp2   = entry - cfg["tp2_mult"] * or_range
                rd    = sl - entry
                if rd > 0:
                    size = min((balance * rpct / 100) / rd, balance * lev / entry)
                    if size > 0:
                        active_tr = Trade(
                            sess["date"], "short", sess["label"],
                            entry, or_range, sl, tp1, tp2, size, balance, score,
                        )

    if active_tr:
        if active_tr.exit_price == 0:
            active_tr.exit_price  = active_tr.entry
            active_tr.exit_reason = "NO_FILL"
            return balance, None
        rd_orig = active_tr.or_range
        sign    = 1 if active_tr.direction == "long" else -1
        active_tr.pnl    = sign * (active_tr.exit_price - active_tr.entry) * active_tr.size_full
        active_tr.r_mult = active_tr.pnl / (rd_orig * active_tr.size_full) if rd_orig > 0 else 0
        return balance, active_tr

    return balance, None

# ====================================================================
# CONCENTRATED PORTFOLIO
# ====================================================================
def run_concentrated(all_sessions_by_date, sma_data_all, atr_data_all,
                     force_close_hour=21, cfg=None):
    if cfg is None:
        cfg = list(PARAM_SETS.values())[0]

    balance       = cfg["starting_balance"]
    equity_curve  = [balance]
    trades        = []
    daily_log     = []
    skipped_score = 0
    no_setup_days = 0
    filter_stats  = {"gap": 0, "atr": 0}
    kelly_history = []

    for d in sorted(all_sessions_by_date.keys()):
        day_sessions = all_sessions_by_date[d]
        candidates   = []

        for sess in day_sessions:
            label    = sess["label"]
            ref_px   = sess["or_close"]
            or_range = sess["or_range"]
            range_pct= (or_range / ref_px * 100) if ref_px > 0 else 0

            if range_pct < cfg["min_or_range_pct"] or range_pct > cfg["max_or_range_pct"]:
                continue
            if cfg["max_gap_pct"] > 0 and sess["gap_pct"] > cfg["max_gap_pct"]:
                filter_stats["gap"] += 1; continue
            atr20 = sess.get("atr20")
            if cfg["max_or_atr_mult"] > 0 and atr20 and atr20 > 0:
                if or_range > cfg["max_or_atr_mult"] * atr20:
                    filter_stats["atr"] += 1; continue

            sma_info = sma_data_all.get(label, {})
            s_long,  r_long  = score_setup(sess, sma_info, "long",  cfg)
            s_short, r_short = score_setup(sess, sma_info, "short", cfg)

            if s_long >= s_short:
                candidates.append((s_long,  "long",  sess, r_long))
            else:
                candidates.append((s_short, "short", sess, r_short))

        if not candidates:
            no_setup_days += 1; equity_curve.append(balance); continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        qualified = [(sc, dr, ss, rs) for sc, dr, ss, rs in candidates
                     if sc >= cfg["min_score"]]

        if not qualified:
            skipped_score += 1; equity_curve.append(balance)
            best = candidates[0]
            daily_log.append({"date": d, "action": "SKIP",
                               "best_asset": best[2]["label"],
                               "best_dir":   best[1],
                               "score":      best[0],
                               "reasons":    best[3]})
            continue

        picks          = []
        picked_assets  = set()
        picked_classes = set()

        for sc, dr, ss, rs in qualified:
            if ss["label"] in picked_assets:
                continue
            asset_class = ASSET_CLASS.get(ss["label"], "unknown")
            if len(picks) == 0:
                picks.append((sc, dr, ss, rs))
                picked_assets.add(ss["label"])
                picked_classes.add(asset_class)
            elif len(picks) == 1 and cfg["max_daily_trades"] >= 2:
                if asset_class not in picked_classes:
                    picks.append((sc, dr, ss, rs))
                    picked_assets.add(ss["label"])
                    picked_classes.add(asset_class)
                    break
            else:
                break

        kelly_risk     = compute_kelly(kelly_history[-cfg["kelly_lookback"]:], cfg,
                                       fallback_pct=cfg["risk_pct"])
        per_trade_risk = min(kelly_risk, cfg["daily_risk_budget"] / len(picks))

        for sc, dr, ss, rs in picks:
            new_balance, trade = simulate_session(
                ss, dr, balance, sc, cfg,
                force_close_hour=ss.get("session_close_hour", force_close_hour)
            )
            if trade:
                trades.append(trade)
                kelly_history.append(trade)
                daily_log.append({
                    "date":        d,
                    "action":      "TRADE",
                    "asset":       trade.label,
                    "dir":         trade.direction,
                    "score":       sc,
                    "entry":       trade.entry,
                    "exit":        trade.exit_price,
                    "exit_reason": trade.exit_reason,
                    "pnl_usd":     new_balance - balance,
                    "r_mult":      trade.r_mult,
                    "balance":     new_balance,
                    "risk_pct":    per_trade_risk,
                    "kelly_risk":  kelly_risk,
                    "n_picks":     len(picks),
                    "reasons":     rs,
                })
            balance = new_balance

        equity_curve.append(max(balance, 0.01))

    return {
        "trades":        trades,
        "equity_curve":  equity_curve,
        "daily_log":     daily_log,
        "balance":       balance,
        "skipped_score": skipped_score,
        "no_setup_days": no_setup_days,
        "filter_stats":  filter_stats,
    }

# ====================================================================
# PER-ASSET INDEPENDENT
# ====================================================================
def run_per_asset(all_sessions_by_label, sma_data_all, force_close_hour=21, cfg=None):
    if cfg is None:
        cfg = list(PARAM_SETS.values())[0]
    results = {}
    for label, sessions in all_sessions_by_label.items():
        balance      = cfg["starting_balance"]
        equity_curve = [balance]
        trades       = []
        sma_info_map = sma_data_all.get(label, {})
        skipped      = 0

        for sess in sessions:
            ref_px   = sess["or_close"]
            or_range = sess["or_range"]
            range_pct= (or_range / ref_px * 100) if ref_px > 0 else 0
            if range_pct < cfg["min_or_range_pct"] or range_pct > cfg["max_or_range_pct"]:
                skipped += 1; continue
            if cfg["max_gap_pct"] > 0 and sess["gap_pct"] > cfg["max_gap_pct"]:
                skipped += 1; continue
            atr20 = sess.get("atr20")
            if cfg["max_or_atr_mult"] > 0 and atr20 and atr20 > 0:
                if or_range > cfg["max_or_atr_mult"] * atr20:
                    skipped += 1; continue

            d  = sess["date"]
            si = sma_info_map.get(d)
            if si and si["close"] > si["sma"]:
                direction = "long"
            elif si and si["close"] < si["sma"]:
                direction = "short"
            else:
                direction = "long"

            score, _ = score_setup(sess, sma_info_map, direction, cfg)
            if score < cfg["min_score"]:
                skipped += 1; continue

            new_balance, trade = simulate_session(
                sess, direction, balance, score, cfg,
                force_close_hour=sess.get("session_close_hour", force_close_hour)
            )
            if trade:
                trades.append(trade)
            balance = new_balance
            equity_curve.append(max(balance, 0.01))

        results[label] = {
            "trades":       trades,
            "equity_curve": equity_curve,
            "balance":      balance,
            "skipped":      skipped,
            "sessions":     len(sessions),
        }
    return results

# ====================================================================
# COMBINED SHARED WALLET — 5min OR + 15min OR same day, same capital
# ====================================================================
def _best_candidate(sessions_for_day, sma_data_all, cfg,
                    exclude_labels=None, exclude_classes=None):
    if exclude_labels  is None: exclude_labels  = set()
    if exclude_classes is None: exclude_classes = set()
    candidates = []
    for sess in sessions_for_day:
        label    = sess["label"]
        ref_px   = sess["or_close"]
        or_range = sess["or_range"]
        range_pct= (or_range / ref_px * 100) if ref_px > 0 else 0
        if range_pct < cfg["min_or_range_pct"] or range_pct > cfg["max_or_range_pct"]:
            continue
        if cfg["max_gap_pct"] > 0 and sess["gap_pct"] > cfg["max_gap_pct"]:
            continue
        atr20 = sess.get("atr20")
        if cfg["max_or_atr_mult"] > 0 and atr20 and atr20 > 0:
            if or_range > cfg["max_or_atr_mult"] * atr20:
                continue
        if label in exclude_labels:
            continue
        asset_class = ASSET_CLASS.get(label, "unknown")
        if asset_class in exclude_classes:
            continue
        sma_info = sma_data_all.get(label, {})
        s_long,  r_long  = score_setup(sess, sma_info, "long",  cfg)
        s_short, r_short = score_setup(sess, sma_info, "short", cfg)
        if s_long >= s_short:
            candidates.append((s_long,  "long",  sess, r_long))
        else:
            candidates.append((s_short, "short", sess, r_short))
    candidates.sort(key=lambda x: x[0], reverse=True)
    qualified = [(sc, dr, ss, rs) for sc, dr, ss, rs in candidates if sc >= cfg["min_score"]]
    return qualified[0] if qualified else None


def run_combined_us_5_15(sessions_5m_by_date, sessions_15m_by_date,
                         sma_data_all, cfg=None):
    if cfg is None:
        cfg = list(PARAM_SETS.values())[0]

    balance       = cfg["starting_balance"]
    equity_curve  = [balance]
    trades        = []
    daily_log     = []
    skipped_score = 0
    no_setup_days = 0
    filter_stats  = {"gap": 0, "atr": 0}
    kelly_history = []

    all_dates = sorted(set(sessions_5m_by_date) | set(sessions_15m_by_date))

    for d in all_dates:
        sess5  = sessions_5m_by_date.get(d, [])
        sess15 = sessions_15m_by_date.get(d, [])

        if not sess5 and not sess15:
            no_setup_days += 1; equity_curve.append(balance); continue

        kelly_risk = compute_kelly(kelly_history[-cfg["kelly_lookback"]:], cfg,
                                   fallback_pct=cfg["risk_pct"])

        pick5 = _best_candidate(sess5, sma_data_all, cfg)

        excl_labels  = {pick5[2]["label"]} if pick5 else set()
        excl_classes = {ASSET_CLASS.get(pick5[2]["label"], "?")} if pick5 else set()
        pick15 = _best_candidate(sess15, sma_data_all, cfg,
                                  exclude_labels=excl_labels,
                                  exclude_classes=excl_classes)

        picks = [p for p in [pick5, pick15] if p is not None]
        if not picks:
            skipped_score += 1; equity_curve.append(balance); continue

        n_picks       = len(picks)
        per_trade_risk= min(kelly_risk, cfg["daily_risk_budget"] / n_picks)
        base_balance  = balance
        day_pnl       = 0.0

        for sc, dr, ss, rs in picks:
            new_bal, trade = simulate_session(
                ss, dr, base_balance, sc, cfg,
                force_close_hour=ss.get("session_close_hour", 21)
            )
            trade_pnl = new_bal - base_balance
            day_pnl  += trade_pnl

            if trade:
                trades.append(trade)
                kelly_history.append(trade)
                or_dur = ss["or_duration_bars"] * 5
                daily_log.append({
                    "date"       : d,
                    "action"     : "TRADE",
                    "asset"      : trade.label,
                    "or_min"     : or_dur,
                    "dir"        : trade.direction,
                    "score"      : sc,
                    "entry"      : trade.entry,
                    "exit"       : trade.exit_price,
                    "exit_reason": trade.exit_reason,
                    "pnl_usd"    : trade_pnl,
                    "r_mult"     : trade.r_mult,
                    "risk_pct"   : per_trade_risk,
                    "kelly_risk" : kelly_risk,
                    "n_picks"    : n_picks,
                    "reasons"    : rs,
                })

        balance = max(base_balance + day_pnl, 0.01)
        equity_curve.append(balance)

    return {
        "trades"       : trades,
        "equity_curve" : equity_curve,
        "daily_log"    : daily_log,
        "balance"      : balance,
        "skipped_score": skipped_score,
        "no_setup_days": no_setup_days,
        "filter_stats" : filter_stats,
    }

# ====================================================================
# STATS
# ====================================================================
def compute_stats(trades, equity_curve, start_bal):
    if not trades:
        return None
    eq   = np.array(equity_curve)
    wins = [t for t in trades if t.pnl > 0]
    loss = [t for t in trades if t.pnl <= 0]
    wr   = len(wins) / len(trades) * 100
    avg_win_r = float(np.mean([t.r_mult for t in wins]))   if wins else 0.0
    avg_los_r = float(np.mean([t.r_mult for t in loss]))   if loss else 0.0
    pf  = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in loss))
           if loss and sum(t.pnl for t in loss) != 0 else float("inf"))
    ret     = (eq[-1] - eq[0]) / eq[0] * 100
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / peak * 100
    maxdd   = float(dd.min())
    daily_r = np.diff(np.log(np.maximum(eq, 1e-9)))
    sharpe  = float(daily_r.mean() / daily_r.std() * np.sqrt(252)) \
              if len(daily_r) > 1 and daily_r.std() > 0 else 0.0
    by_reason = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
    by_asset = defaultdict(list)
    for t in trades:
        by_asset[t.label].append(t)

    n_days = len(equity_curve) - 1
    years  = max(n_days / 252, 0.01)
    cagr   = ((eq[-1] / eq[0]) ** (1.0 / years) - 1) * 100 if eq[0] > 0 else 0.0

    return {
        "trades":       len(trades),
        "wins":         len(wins),
        "losses":       len(loss),
        "win_rate":     wr,
        "avg_win_r":    avg_win_r,
        "avg_loss_r":   avg_los_r,
        "profit_factor":pf,
        "tp1_hit_rate": sum(1 for t in trades if t.tp1_hit) / len(trades) * 100,
        "total_return": ret,
        "max_drawdown": maxdd,
        "sharpe":       sharpe,
        "cagr":         cagr,
        "final_balance":float(eq[-1]),
        "by_reason":    by_reason,
        "by_asset":     {k: len(v) for k, v in by_asset.items()},
        "avg_score":    float(np.mean([t.score for t in trades])),
        "trading_days": n_days,
    }

# ====================================================================
# OUTPUT HELPERS
# ====================================================================
def bar(n=80):  print("=" * n)
def sec(n=80):  print("-" * n)


def print_concentrated_summary(stats, result, tag):
    if not stats:
        print(f"  [{tag}] No trades executed.")
        return
    pf_s = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 1e6 else "INF"
    print(f"  [{tag}] Concentrated Portfolio")
    print(f"    Trades: {stats['trades']:>3}  WR: {stats['win_rate']:.1f}%  "
          f"PF: {pf_s}  Sharpe: {stats['sharpe']:.2f}  CAGR: {stats['cagr']:.1f}%")
    print(f"    MaxDD: {stats['max_drawdown']:.1f}%  Return: {stats['total_return']:+.1f}%  "
          f"$100 -> ${stats['final_balance']:.2f}  Days: {stats['trading_days']}")
    reasons = "  ".join(f"{k}:{v}" for k, v in sorted(stats["by_reason"].items()))
    print(f"    Exits: {reasons}")
    if stats["by_asset"]:
        assets_str = ", ".join(f"{a}({n})" for a, n in
                               sorted(stats["by_asset"].items(), key=lambda x: -x[1]))
        print(f"    Assets: {assets_str}")
    fs = result["filter_stats"]
    print(f"    ATR-filtered: {fs['atr']}  Score-skipped: {result['skipped_score']}d  "
          f"No-setup: {result['no_setup_days']}d")


def print_per_asset_table(per_asset_results, sma_counts, tag):
    print(f"  [{tag}] Per-Asset Breakdown")
    print(f"    {'Asset':<7} {'SMAd':>5} {'Sess':>4} {'Trd':>3} {'WR':>6} "
          f"{'PF':>6} {'Ret':>7} {'Final$':>7}")
    sec(72)
    for label in sorted(per_asset_results.keys()):
        data   = per_asset_results[label]
        trades = data["trades"]
        sma_d  = sma_counts.get(label, 0)
        if not trades:
            print(f"    {label:<7} {sma_d:>5} {data['sessions']:>4}  --    --     --"
                  f"      --    ${data['balance']:.2f}")
            continue
        s   = compute_stats(trades, data["equity_curve"], 100.0)
        pfs = f"{s['profit_factor']:.2f}" if s["profit_factor"] < 1e6 else "INF"
        print(f"    {label:<7} {sma_d:>5} {data['sessions']:>4} {s['trades']:>3} "
              f"{s['win_rate']:>5.1f}% {pfs:>6} {s['total_return']:>+6.1f}%  "
              f"${s['final_balance']:>6.2f}")


def print_period_comparison(all_period_results):
    """Print a comparison table across all periods."""
    bar()
    print("  PERIOD COMPARISON TABLE — 1Y vs 3Y vs 5Y")
    bar()
    print(f"  {'Period':<6} {'Variant':<20} {'Param':<5} {'T':>4} {'WR':>6} "
          f"{'PF':>6} {'Shrp':>6} {'CAGR':>6} {'MaxDD':>7} {'Ret':>8} {'$End':>8}")
    sec()
    ORDER = ["US_5min", "US_15min", "CombinedUS_5and15"]

    for period_name in ["1Y", "3Y", "5Y"]:
        if period_name not in all_period_results:
            continue
        pdata = all_period_results[period_name]
        for pname in PARAM_SETS:
            if pname not in pdata:
                continue
            for vname in ORDER:
                if vname not in pdata[pname]:
                    continue
                stats = pdata[pname][vname]["stats"]
                tag_v = vname[:18]
                if stats is None:
                    print(f"  {period_name:<6} {tag_v:<20} {pname:<5} "
                          f"{'0':>4}  {'—':>6} {'—':>6} {'—':>6} {'—':>6} "
                          f"{'—':>7} {'—':>8} {'$100.00':>8}")
                    continue
                pf_s = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 1e6 else "INF"
                mark = " *" if "Combined" in vname else ""
                print(f"  {period_name:<6} {tag_v:<20} {pname:<5} "
                      f"{stats['trades']:>4} {stats['win_rate']:>5.1f}% {pf_s:>6} "
                      f"{stats['sharpe']:>6.2f} {stats['cagr']:>5.1f}% "
                      f"{stats['max_drawdown']:>+6.1f}% {stats['total_return']:>+7.1f}% "
                      f"${stats['final_balance']:>7.2f}{mark}")
        sec()


# ====================================================================
# RUN BACKTEST FOR ONE PERIOD
# ====================================================================
def run_period(period_name, period_days, all_data, sma_data_all, atr_data_all,
               close_data_all, sma_counts, daily_vol_all=None):
    """
    Run the full backtest for a given period.
    Slices data to the date range, builds sessions, runs all variants × param sets.
    daily_vol_all: {label: {date: (vol, vol_ma)}} from Tiingo daily endpoint.
    Returns results dict: results[pset_name][variant_name] = {...}
    """
    today     = datetime.now(timezone.utc).date()
    start_dt  = today - timedelta(days=period_days)
    end_dt    = today

    print(f"  Period: {period_name}  ({start_dt} -> {end_dt})")
    sec()

    # Slice 5m data to period
    period_data = {}
    for label, info in all_data.items():
        df = info["df"]
        if df is None or df.empty:
            period_data[label] = {"df": None, "asset": info["asset"]}
            continue
        mask = (df["time"].dt.date >= start_dt) & (df["time"].dt.date <= end_dt)
        sliced = df[mask].copy().reset_index(drop=True)
        period_data[label] = {"df": sliced, "asset": info["asset"]}
        n = len(sliced)
        if n > 0:
            d0 = sliced["time"].dt.date.min()
            d1 = sliced["time"].dt.date.max()
            print(f"    {label:<6}  {n:>7,} bars  {d0} -> {d1}")
        else:
            print(f"    {label:<6}  NO DATA in period")

    print()

    # Build sessions & run variants × param sets
    all_results = {pname: {} for pname in PARAM_SETS}
    _us5_by_date  = defaultdict(list)
    _us15_by_date = defaultdict(list)

    for variant in VARIANTS:
        vname = variant["name"]

        all_sessions_by_date  = defaultdict(list)
        all_sessions_by_label = defaultdict(list)

        for label, info in period_data.items():
            df    = info["df"]
            asset = info["asset"]
            if df is None or len(df) < 20:
                continue

            # Pass daily volume override for IEX assets (have no 5m volume)
            dvol_override = None
            if daily_vol_all and label in daily_vol_all:
                dvol_override = daily_vol_all[label]

            sessions = build_sessions_variant(
                df, asset, atr_data_all.get(label, {}),
                close_data_all.get(label, {}), variant,
                daily_vol_override=dvol_override
            )
            for s in sessions:
                all_sessions_by_date[s["date"]].append(s)
                all_sessions_by_label[label].append(s)
                if vname == "US_5min":
                    _us5_by_date[s["date"]].append(s)
                elif vname == "US_15min":
                    _us15_by_date[s["date"]].append(s)

        print(f"    {vname}: {len(all_sessions_by_date)} trading days, "
              f"{sum(len(v) for v in all_sessions_by_date.values())} total sessions")

        for pname, cfg in PARAM_SETS.items():
            conc_result = run_concentrated(
                all_sessions_by_date, sma_data_all, atr_data_all,
                force_close_hour=variant["force_close_hour"], cfg=cfg
            )
            conc_stats = compute_stats(
                conc_result["trades"], conc_result["equity_curve"],
                cfg["starting_balance"]
            )
            per_asset = run_per_asset(
                all_sessions_by_label, sma_data_all,
                force_close_hour=variant["force_close_hour"], cfg=cfg
            )
            all_results[pname][vname] = {
                "variant":   variant,
                "stats":     conc_stats,
                "result":    conc_result,
                "per_asset": per_asset,
            }

    # Combined wallet
    for pname, cfg in PARAM_SETS.items():
        comb_result = run_combined_us_5_15(
            _us5_by_date, _us15_by_date, sma_data_all, cfg=cfg
        )
        comb_stats = compute_stats(
            comb_result["trades"], comb_result["equity_curve"],
            cfg["starting_balance"]
        )
        all_results[pname]["CombinedUS_5and15"] = {
            "variant": {"name": "CombinedUS_5and15",
                        "description": "Shared wallet: 5min + 15min OR same day",
                        "session_type": "combined_us", "force_close_hour": 21},
            "stats":   comb_stats,
            "result":  comb_result,
            "per_asset": {},
        }

    return all_results


# ====================================================================
# MAIN
# ====================================================================
def main():
    bar()
    print("  ORB BACKTEST — Tiingo API — Intraday 5min + 15min")
    print("  Periods: 1Y, 3Y, 5Y backtest comparison")
    print("  Data: Tiingo IEX (stocks/ETFs) + Tiingo Crypto (SOL)")
    print("  Param sets: S9 | S10 | S12")
    print("  Assets: TSLA AMZN PLTR INTC HOOD COIN MSTR | XAU(GLD) XAG(SLV) | SOL")
    bar()
    print()

    # ── Phase 1: Fetch ALL 5Y data (maximum range) ────────────────
    today     = datetime.now(timezone.utc).date()
    start_5y  = today - timedelta(days=365 * 5 + 60)  # +60d buffer for SMA warmup
    end_date  = today

    print("  Phase 1: Fetching 5-minute + daily data from Tiingo (5Y range)...")
    print(f"  Date range: {start_5y} -> {end_date}")
    sec()

    all_data       = {}
    sma_data_all   = {}
    atr_data_all   = {}
    close_data_all = {}
    sma_counts     = {}

    # Request budget estimation:
    #   IEX stocks (9 tickers): 3 chunks each for 5Y at 2yr chunks = 27 requests
    #   Crypto (1 ticker): 11 chunks for 5Y at 180d chunks            = 11 requests
    #   Daily volume (9 IEX tickers): 1 request each                   =  9 requests
    #   Daily SMA/ATR: computed from 5m data (0 extra requests)
    #   Total ≈ 47 requests.  At 75s spacing ≈ 59 minutes.
    n_assets = len(ASSETS)
    iex_assets = sum(1 for a in ASSETS if a["source"] == "iex")
    crypto_assets = sum(1 for a in ASSETS if a["source"] == "crypto")
    est_iex_chunks = iex_assets * 3  # 5Y / 2yr = 3 chunks per IEX asset
    est_crypto_chunks = crypto_assets * 11  # 5Y / 180d ≈ 11 chunks
    est_daily_vol = iex_assets  # 1 request per IEX ticker for daily volume
    est_total = est_iex_chunks + est_crypto_chunks + est_daily_vol
    est_min = est_total * REQUEST_GAP_SEC / 60
    print(f"  Estimated API calls: ~{est_total} (IEX 5m: ~{est_iex_chunks}, "
          f"Crypto 5m: ~{est_crypto_chunks}, Daily vol: ~{est_daily_vol})")
    print(f"  Estimated time: ~{est_min:.0f} min (rate limit: {REQUEST_GAP_SEC}s between requests)")
    print(f"  Daily SMA/ATR will be computed from 5m data (saves {n_assets} API calls)")
    print(f"  Daily volume from Tiingo daily endpoint (IEX 5m has no volume)")
    print(f"  Cached assets will be loaded instantly.")
    sec()

    for i, asset in enumerate(ASSETS):
        label  = asset["label"]
        ticker = asset["tiingo_sym"]
        source = asset["source"]

        # Try cache first
        cached_df = load_5m_cache(label)

        if cached_df is not None and len(cached_df) > 100:
            print(f"    [{i+1}/{n_assets}] {label:<6}  5m: {len(cached_df):>8,} bars [CACHED]", end="")
            df, src = cached_df, f"cache_{source}"
        else:
            print(f"    [{i+1}/{n_assets}] {label:<6}  5m: fetching", end="", flush=True)
            df, src = fetch_5m(asset, start_5y, end_date)
            n = len(df) if df is not None else 0
            print(f" => {n:>8,} bars [{src}]", end="", flush=True)
            if df is not None and len(df) > 50:
                save_5m_cache(label, df)

        # Compute daily SMA/ATR from 5m data (NO extra API call)
        if df is not None and len(df) > 50:
            sma, atr, closes = compute_daily_from_5m(df, BASE_CFG["sma_period"])
            print(f"  daily: sma={len(sma)}d atr={len(atr)}d [from 5m]")
            save_daily_cache(label, sma, atr, closes)
        else:
            sma, atr, closes = {}, {}, {}
            print(f"  NO 5m DATA — skipping daily")

        all_data[label]       = {"df": df, "source": src, "asset": asset}
        sma_data_all[label]   = sma
        atr_data_all[label]   = atr
        close_data_all[label] = closes
        sma_counts[label]     = len(sma)

    print()
    bar()
    print("  Phase 1 complete. Data summary:")
    sec()
    for label, info in all_data.items():
        df = info["df"]
        if df is not None and len(df) > 0:
            d0 = df["time"].dt.date.min()
            d1 = df["time"].dt.date.max()
            n  = len(df)
            days = len(df["time"].dt.date.unique())
            print(f"    {label:<6}  {n:>9,} bars  {days:>5} days  "
                  f"{d0} -> {d1}  sma={sma_counts[label]}d")
        else:
            print(f"    {label:<6}  NO DATA")
    print()

    # ── Phase 1b: Fetch daily volume for IEX assets ───────────────
    # IEX 5min data has NO volume column — fetch real daily volume
    # from Tiingo's daily endpoint for proper VOL scoring.
    daily_vol_all = {}
    iex_tickers = [(a["label"], a["tiingo_sym"]) for a in ASSETS if a["source"] == "iex"]
    print("  Phase 1b: Fetching daily volume for IEX assets (5m has no volume)...")
    sec()
    for i, (label, ticker) in enumerate(iex_tickers):
        cached_vol = load_daily_vol_cache(label)
        if cached_vol is not None and len(cached_vol) > 50:
            daily_vol_all[label] = cached_vol
            print(f"    [{i+1}/{len(iex_tickers)}] {label:<6}  daily vol: "
                  f"{len(cached_vol):>5} days [CACHED]")
        else:
            print(f"    [{i+1}/{len(iex_tickers)}] {label:<6}  daily vol: fetching",
                  end="", flush=True)
            vol_map = fetch_tiingo_daily_volume(ticker, start_5y, end_date)
            if vol_map:
                daily_vol_all[label] = vol_map
                save_daily_vol_cache(label, vol_map)
                print(f" => {len(vol_map):>5} days")
            else:
                print(f" => FAILED (no volume data)")
    print()

    # ── Phase 2: Run backtests for each period ────────────────────
    all_period_results = {}

    for period_name, period_days in PERIODS.items():
        bar()
        print(f"  BACKTEST PERIOD: {period_name} ({period_days} days)")
        bar()
        print()

        period_results = run_period(
            period_name, period_days, all_data,
            sma_data_all, atr_data_all, close_data_all, sma_counts,
            daily_vol_all=daily_vol_all
        )
        all_period_results[period_name] = period_results
        print()

        # Print detailed results for this period
        for pname, pdata in period_results.items():
            cfg = PARAM_SETS[pname]
            print(f"    {cfg['label']}")
            for vname in ["US_5min", "US_15min", "CombinedUS_5and15"]:
                if vname not in pdata:
                    continue
                tag = f"{period_name}/{vname}/{pname}"
                print_concentrated_summary(pdata[vname]["stats"],
                                           pdata[vname]["result"], tag)
            print()

    # ── Phase 3: Cross-period comparison ──────────────────────────
    print()
    print_period_comparison(all_period_results)

    # ── Per-asset breakdown for largest period with data ──────────
    best_period = None
    for pn in ["5Y", "3Y", "1Y"]:
        if pn in all_period_results:
            best_period = pn
            break

    if best_period:
        bar()
        print(f"  PER-ASSET BREAKDOWN — {best_period} / US_5min")
        bar()
        for pname, pdata in all_period_results[best_period].items():
            if "US_5min" not in pdata:
                continue
            tag = f"{best_period}/US_5min/{pname}"
            print()
            print_per_asset_table(pdata["US_5min"]["per_asset"], sma_counts, tag)
            print()

    # ── Combined wallet trade log for 1Y ──────────────────────────
    if "1Y" in all_period_results:
        bar()
        print("  COMBINED WALLET TRADE LOG — 1Y (most recent)")
        sec()
        for pname, pdata in all_period_results["1Y"].items():
            if "CombinedUS_5and15" not in pdata:
                continue
            result = pdata["CombinedUS_5and15"]["result"]
            trades_log = [e for e in result["daily_log"] if e["action"] == "TRADE"]
            print(f"  [{pname}] {len(trades_log)} trades")
            sec(72)
            for e in trades_log:
                print(f"    {e['date']}  {e['asset']:<6} {e['dir']:<5} "
                      f"{e['or_min']}min-OR  s={e['score']}/8  "
                      f"entry={e['entry']:.2f}  exit={e['exit']:.2f}  "
                      f"{e['exit_reason']:<4}  R={e['r_mult']:+.2f}  "
                      f"pnl={e['pnl_usd']:+.2f}")
            print()

    # ── Final summary ─────────────────────────────────────────────
    bar()
    print("  KEY FINDINGS — PERIOD COMPARISON SUMMARY")
    sec()
    for period_name in ["1Y", "3Y", "5Y"]:
        if period_name not in all_period_results:
            continue
        pdata = all_period_results[period_name]
        print(f"  --- {period_name} ---")
        for pname in PARAM_SETS:
            if pname not in pdata:
                continue
            for vname in ["US_5min", "US_15min", "CombinedUS_5and15"]:
                if vname not in pdata[pname]:
                    continue
                stats = pdata[pname][vname]["stats"]
                tag   = f"{pname}/{vname}"
                if stats:
                    pf_s = (f"{stats['profit_factor']:.2f}"
                            if stats['profit_factor'] < 1e6 else "INF")
                    mark = "  ** COMBINED" if "Combined" in vname else ""
                    print(f"    {tag:<26}: {stats['trades']}T  "
                          f"WR {stats['win_rate']:.1f}%  PF {pf_s}  "
                          f"CAGR {stats['cagr']:.1f}%  "
                          f"MaxDD {stats['max_drawdown']:.1f}%  "
                          f"Ret {stats['total_return']:+.1f}%{mark}")
                else:
                    print(f"    {tag:<26}: NO TRADES")
        print()

    bar()
    print("  BACKTEST COMPLETE — Tiingo API")
    bar()


if __name__ == "__main__":
    main()
