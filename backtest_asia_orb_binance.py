#!/usr/bin/env python3
"""
Asia Session ORB Backtester (Binance 5m)
========================================

Spec implemented:
- OR window: configurable by strategy mode (5m / 15m / 30m / combined 5m+15m)
- Entry check: close breakout confirmation, then execute at next candle open
- Hard exit: 04:00 UTC
- Skip Sundays
- Assets requested: BTC/USDT, ETH/USDT, SOL/USDT, XAU/USDT, XAG/USDT
- Timeframe: 5-minute candles
- Period: 2022-01-01 -> today (default)

Notes on Binance symbol availability:
- Binance spot currently does not list XAUUSDT/XAGUSDT directly.
- This script resolves requested symbols against live exchange info.
- XAU/USDT falls back to PAXGUSDT if available.
- XAG/USDT is marked unavailable when no Binance symbol exists.

Run:
  python backtest_asia_orb_binance.py
  python backtest_asia_orb_binance.py --start-date 2022-01-01 --refresh-data
  python backtest_asia_orb_binance.py --max-configs 4 --no-plots
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

UTC = timezone.utc
FIVE_MIN_MS = 5 * 60 * 1000

BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

BUFFER_PCTS = [0, 10, 20, 30]
SIZING_MODES = ["risk_1pct", "fixed_1000"]
STRATEGY_MODES = ["asia_5m", "asia_15m", "asia_30m", "asia_combined_5m_15m"]
OR_BARS_BY_MODE = {"asia_5m": 1, "asia_15m": 3, "asia_30m": 6}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass(frozen=True)
class AssetSpec:
    label: str
    requested_symbol: str
    candidates: Tuple[str, ...]
    is_metal: bool


ASSET_SPECS: Tuple[AssetSpec, ...] = (
    AssetSpec("BTC", "BTCUSDT", ("BTCUSDT",), False),
    AssetSpec("ETH", "ETHUSDT", ("ETHUSDT",), False),
    AssetSpec("SOL", "SOLUSDT", ("SOLUSDT",), False),
    AssetSpec("XAU", "XAUUSDT", ("XAUUSDT", "PAXGUSDT"), True),
    AssetSpec("XAG", "XAGUSDT", ("XAGUSDT",), True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asia Session ORB backtester on Binance 5m data")
    parser.add_argument("--start-date", default="2022-01-01", help="Backtest start date in YYYY-MM-DD")
    parser.add_argument(
        "--end-date",
        default=datetime.now(UTC).date().isoformat(),
        help="Backtest end date in YYYY-MM-DD (inclusive)",
    )
    parser.add_argument("--cache-dir", default="cache/binance_asia_orb", help="Local cache directory")
    parser.add_argument("--output-dir", default="results/asia_orb_binance", help="Output directory")
    parser.add_argument("--start-equity", type=float, default=100_000.0, help="Starting equity")
    parser.add_argument(
        "--strategy-mode",
        choices=STRATEGY_MODES,
        default="asia_combined_5m_15m",
        help="ORB mode: 5m, 15m, 30m, or combined 5m+15m shared wallet",
    )
    parser.add_argument(
        "--volume-threshold",
        type=float,
        default=0.40,
        help="Volume filter threshold as a multiple of 5-session rolling average (e.g., 0.40, 1.30)",
    )
    parser.add_argument("--refresh-data", action="store_true", help="Ignore cache and refetch data")
    parser.add_argument("--sleep-sec", type=float, default=0.03, help="Sleep between Binance requests")
    parser.add_argument("--max-configs", type=int, default=0, help="Limit number of configs for quick smoke tests")
    parser.add_argument("--no-plots", action="store_true", help="Disable chart generation")
    return parser.parse_args()


def to_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


def calendar_dates(start_d: date, end_d: date) -> List[date]:
    out = []
    cur = start_d
    while cur <= end_d:
        if cur.weekday() != 6:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def safe_request_json(url: str, params: Optional[dict] = None, retries: int = 6) -> dict | list:
    backoff = 1.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (418, 429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)
                continue
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 20.0)
    raise RuntimeError(f"Failed request: {url}")


def fetch_spot_symbols() -> set[str]:
    payload = safe_request_json(BINANCE_EXCHANGE_INFO_URL)
    return {s["symbol"] for s in payload.get("symbols", []) if s.get("status") == "TRADING"}


def resolve_assets(specs: Iterable[AssetSpec]) -> Tuple[Dict[str, dict], List[str]]:
    symbols = fetch_spot_symbols()
    resolved: Dict[str, dict] = {}
    notes: List[str] = []
    for spec in specs:
        actual = None
        for candidate in spec.candidates:
            if candidate in symbols:
                actual = candidate
                break
        resolved[spec.label] = {
            "label": spec.label,
            "requested_symbol": spec.requested_symbol,
            "symbol": actual,
            "is_metal": spec.is_metal,
            "available": actual is not None,
        }
        if actual is None:
            notes.append(f"{spec.label}: requested {spec.requested_symbol} unavailable on Binance spot")
        elif actual != spec.requested_symbol:
            notes.append(f"{spec.label}: using proxy {actual} for requested {spec.requested_symbol}")
    return resolved, notes


def fetch_klines_range(
    symbol: str,
    start_ms: int,
    end_ms: int,
    sleep_sec: float = 0.03,
    limit: int = 1000,
    progress_label: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[list] = []
    cursor = start_ms
    batch_count = 0
    if progress_label:
        start_dt = pd.to_datetime(start_ms, unit="ms", utc=True)
        end_dt = pd.to_datetime(end_ms, unit="ms", utc=True)
        print(f"    fetching {progress_label}: {start_dt.date()} -> {end_dt.date()}", flush=True)

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": limit,
        }
        payload = safe_request_json(BINANCE_KLINES_URL, params=params)
        if not payload:
            break
        if isinstance(payload, dict) and payload.get("code"):
            raise RuntimeError(f"Binance error for {symbol}: {payload}")

        batch = payload
        rows.extend(batch)
        batch_count += 1
        if progress_label and (batch_count % 25 == 0):
            latest_dt = pd.to_datetime(int(batch[-1][0]), unit="ms", utc=True)
            print(
                f"    {progress_label}: batches={batch_count} bars={len(rows):,} up_to={latest_dt}",
                flush=True,
            )
        last_open = int(batch[-1][0])
        next_cursor = last_open + FIVE_MIN_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(max(sleep_sec, 0.0))

    if not rows:
        return pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume", "close_time"]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
    df["open_time"] = df["open_time"].astype(np.int64)
    df["close_time"] = df["close_time"].astype(np.int64)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df


def load_or_fetch_5m(
    symbol: str,
    start_d: date,
    end_d: date,
    cache_dir: Path,
    refresh: bool,
    sleep_sec: float,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_{start_d}_{end_d}_5m.csv.gz"
    if cache_file.exists() and not refresh:
        print(f"    cache hit: {cache_file.name}", flush=True)
        df = pd.read_csv(cache_file)
        for c in ["open_time", "close_time"]:
            df[c] = df[c].astype(np.int64)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        return df

    start_ms = to_ms(start_d)
    end_ms = to_ms(end_d + timedelta(days=1))
    df = fetch_klines_range(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        sleep_sec=sleep_sec,
        progress_label=symbol,
    )
    df.to_csv(cache_file, index=False, compression="gzip")
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dt"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out["date"] = out["dt"].dt.date
    out["hour"] = out["dt"].dt.hour
    out["minute"] = out["dt"].dt.minute
    return out


def expanding_percentile(series: pd.Series, q: float, min_obs: int) -> pd.Series:
    hist: List[float] = []
    out: List[float] = []
    for val in series:
        if len(hist) >= min_obs:
            out.append(float(np.quantile(hist, q)))
        else:
            out.append(np.nan)
        if pd.notna(val):
            hist.append(float(val))
    return pd.Series(out, index=series.index, dtype=float)


def build_session_pack(
    df_raw: pd.DataFrame,
    or_duration_bars: int,
) -> Tuple[pd.DataFrame, Dict[date, pd.DataFrame]]:
    df = add_time_columns(df_raw)
    session_df = df[df["hour"].between(1, 3)].copy()
    bars_by_date: Dict[date, pd.DataFrame] = {}
    records: List[dict] = []
    prev_close: Optional[float] = None

    for d, grp in session_df.groupby("date", sort=True):
        grp = grp.sort_values("open_time").reset_index(drop=True)
        hard_bar = grp[(grp["hour"] == 3) & (grp["minute"] == 55)]
        hard_exit_px = float(hard_bar.iloc[-1]["close"]) if not hard_bar.empty else float(grp.iloc[-1]["close"])
        hard_exit_time_ms = int(hard_bar.iloc[-1]["close_time"]) if not hard_bar.empty else int(grp.iloc[-1]["close_time"])

        or_close_minute = int(or_duration_bars * 5)
        or_bars = grp[(grp["hour"] == 1) & (grp["minute"] < or_close_minute)].copy()
        has_full_or = len(or_bars) == or_duration_bars
        or_high = float(or_bars["high"].max()) if has_full_or else np.nan
        or_low = float(or_bars["low"].min()) if has_full_or else np.nan
        or_range = float(or_high - or_low) if has_full_or else np.nan

        if has_full_or:
            or_open = float(or_bars.iloc[0]["open"])
            or_close = float(or_bars.iloc[-1]["close"])
            or_move_pct = abs((or_close / or_open) - 1.0) * 100.0 if or_open > 0 else np.nan
        else:
            or_open = np.nan
            or_close = np.nan
            or_move_pct = np.nan

        sess_high = float(grp["high"].max())
        sess_low = float(grp["low"].min())
        sess_vol = float(grp["volume"].sum())
        tr = sess_high - sess_low if prev_close is None else max(
            sess_high - sess_low,
            abs(sess_high - prev_close),
            abs(sess_low - prev_close),
        )

        records.append(
            {
                "date": d,
                "weekday": pd.Timestamp(d).day_name(),
                "or_valid": has_full_or,
                "or_open": or_open,
                "or_close": or_close,
                "or_high": or_high,
                "or_low": or_low,
                "or_range": or_range,
                "or_duration_bars": int(or_duration_bars),
                "entry_start_minute": int(or_close_minute),
                "or_move_pct": or_move_pct,
                "session_high": sess_high,
                "session_low": sess_low,
                "session_volume": sess_vol,
                "hard_exit_price": hard_exit_px,
                "hard_exit_time": pd.to_datetime(hard_exit_time_ms, unit="ms", utc=True),
                "tr": tr,
                "bars_count": int(len(grp)),
            }
        )
        bars_by_date[d] = grp[
            ["open_time", "close_time", "open", "high", "low", "close", "volume", "hour", "minute"]
        ].copy()
        prev_close = hard_exit_px

    sessions = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if sessions.empty:
        return sessions, bars_by_date

    sessions["vol_ma5"] = sessions["session_volume"].rolling(5, min_periods=5).mean().shift(1)
    sessions["atr20"] = sessions["tr"].rolling(20, min_periods=20).mean().shift(1)
    sessions["atr20_p20"] = expanding_percentile(sessions["atr20"], q=0.20, min_obs=20)
    return sessions, bars_by_date


def simulate_trade_for_session(
    asset_label: str,
    trade_date: date,
    bars: pd.DataFrame,
    or_high: float,
    or_low: float,
    or_range: float,
    buffer_pct: int,
    entry_start_minute: int,
) -> Optional[dict]:
    if not np.isfinite(or_range) or or_range <= 0:
        return None

    bars_view = bars.sort_values("open_time").reset_index(drop=True)
    entry_mask = (bars_view["hour"] > 1) | ((bars_view["hour"] == 1) & (bars_view["minute"] >= entry_start_minute))
    candidate_idx = np.where(entry_mask.to_numpy())[0]
    if len(candidate_idx) == 0:
        return None

    buffer_frac = buffer_pct / 100.0
    long_trigger = or_high + (buffer_frac * or_range)
    short_trigger = or_low - (buffer_frac * or_range)

    confirmation_bar = None
    entry_bar = None
    direction = 0
    for pos in candidate_idx:
        row = bars_view.iloc[pos]
        c = float(row["close"])
        if c > long_trigger:
            if pos + 1 >= len(bars_view):
                return None
            confirmation_bar = row
            entry_bar = bars_view.iloc[pos + 1]
            direction = 1
            break
        if c < short_trigger:
            if pos + 1 >= len(bars_view):
                return None
            confirmation_bar = row
            entry_bar = bars_view.iloc[pos + 1]
            direction = -1
            break
    if entry_bar is None or confirmation_bar is None:
        return None

    entry_price = float(entry_bar["open"])
    entry_time = pd.to_datetime(int(entry_bar["open_time"]), unit="ms", utc=True)
    confirm_time = pd.to_datetime(int(confirmation_bar["close_time"]), unit="ms", utc=True)
    entry_open_time = int(entry_bar["open_time"])

    stop_price = or_low if direction == 1 else or_high
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return None

    tp1_price = entry_price + (direction * 1.0 * or_range)
    tp2_price = entry_price + (direction * 1.8 * or_range)

    remaining = 1.0
    unit_pnl = 0.0
    tp1_hit = False
    tp2_hit = False
    sl_hit = False

    exit_price = float(bars_view.iloc[-1]["close"])
    exit_time = pd.to_datetime(int(bars_view.iloc[-1]["close_time"]), unit="ms", utc=True)
    exit_reason = "hard_exit"

    # Include the entry bar itself so immediate TP/SL in the entry candle can be handled.
    manage_bars = bars_view[bars_view["open_time"] >= entry_open_time]
    for bar in manage_bars.itertuples(index=False):
        high = float(bar.high)
        low = float(bar.low)
        bar_exit_time = pd.to_datetime(int(bar.close_time), unit="ms", utc=True)

        if direction == 1:
            hit_stop = low <= stop_price
            hit_tp1 = (not tp1_hit) and (high >= tp1_price)
            hit_tp2 = high >= tp2_price

            if hit_stop and (hit_tp1 or hit_tp2):
                unit_pnl += (stop_price - entry_price) * remaining
                remaining = 0.0
                sl_hit = True
                exit_price = stop_price
                exit_time = bar_exit_time
                exit_reason = "sl_conflict"
                break

            if hit_stop:
                unit_pnl += (stop_price - entry_price) * remaining
                remaining = 0.0
                sl_hit = True
                exit_price = stop_price
                exit_time = bar_exit_time
                exit_reason = "sl"
                break

            if hit_tp1:
                close_qty = min(remaining, 0.5)
                unit_pnl += (tp1_price - entry_price) * close_qty
                remaining -= close_qty
                tp1_hit = True
                exit_price = tp1_price
                exit_time = bar_exit_time
                exit_reason = "tp1"

            if hit_tp2 and remaining > 0:
                unit_pnl += (tp2_price - entry_price) * remaining
                remaining = 0.0
                tp2_hit = True
                exit_price = tp2_price
                exit_time = bar_exit_time
                exit_reason = "tp2"
                break
        else:
            hit_stop = high >= stop_price
            hit_tp1 = (not tp1_hit) and (low <= tp1_price)
            hit_tp2 = low <= tp2_price

            if hit_stop and (hit_tp1 or hit_tp2):
                unit_pnl += (entry_price - stop_price) * remaining
                remaining = 0.0
                sl_hit = True
                exit_price = stop_price
                exit_time = bar_exit_time
                exit_reason = "sl_conflict"
                break

            if hit_stop:
                unit_pnl += (entry_price - stop_price) * remaining
                remaining = 0.0
                sl_hit = True
                exit_price = stop_price
                exit_time = bar_exit_time
                exit_reason = "sl"
                break

            if hit_tp1:
                close_qty = min(remaining, 0.5)
                unit_pnl += (entry_price - tp1_price) * close_qty
                remaining -= close_qty
                tp1_hit = True
                exit_price = tp1_price
                exit_time = bar_exit_time
                exit_reason = "tp1"

            if hit_tp2 and remaining > 0:
                unit_pnl += (entry_price - tp2_price) * remaining
                remaining = 0.0
                tp2_hit = True
                exit_price = tp2_price
                exit_time = bar_exit_time
                exit_reason = "tp2"
                break

    if remaining > 0:
        hard_exit_px = float(bars_view.iloc[-1]["close"])
        unit_pnl += ((hard_exit_px - entry_price) if direction == 1 else (entry_price - hard_exit_px)) * remaining
        exit_price = hard_exit_px
        exit_time = pd.to_datetime(int(bars_view.iloc[-1]["close_time"]), unit="ms", utc=True)
        exit_reason = "hard_exit"

    r_multiple = unit_pnl / stop_distance if stop_distance > 0 else np.nan
    return {
        "asset": asset_label,
        "trade_date": trade_date,
        "confirmation_time": confirm_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": "long" if direction == 1 else "short",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "stop_distance": stop_distance,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "tp1_hit": bool(tp1_hit),
        "tp2_hit": bool(tp2_hit),
        "sl_hit": bool(sl_hit),
        "exit_reason": exit_reason,
        "or_high": or_high,
        "or_low": or_low,
        "or_range": or_range,
        "or_range_pct_of_entry": (or_range / entry_price * 100.0) if entry_price > 0 else np.nan,
        "unit_pnl": unit_pnl,
        "r_multiple": r_multiple,
    }


def run_config_single_mode(
    config: dict,
    asset_packs: Dict[str, dict],
    xau_move_map: Dict[date, float],
    volume_threshold: float,
    strategy_mode: str,
) -> pd.DataFrame:
    records: List[dict] = []
    buffer_pct = int(config["buffer_pct"])

    for asset, pack in asset_packs.items():
        if not pack["available"]:
            continue

        sessions = pack["sessions"]
        bars_by_date = pack["bars_by_date"]
        is_metal = bool(pack["is_metal"])

        for row in sessions.itertuples(index=False):
            d = row.date
            if pd.Timestamp(d).weekday() == 6:
                continue
            if not bool(row.or_valid):
                continue

            if config["pboc_filter"] and is_metal:
                xau_move = xau_move_map.get(d, np.nan)
                if pd.isna(xau_move) or abs(xau_move) > 0.3:
                    continue

            if config["volume_filter"]:
                if pd.isna(row.vol_ma5) or row.session_volume < (volume_threshold * row.vol_ma5):
                    continue

            if config["atr_filter"]:
                if pd.isna(row.atr20) or pd.isna(row.atr20_p20) or row.atr20 < row.atr20_p20:
                    continue

            bars = bars_by_date.get(d)
            if bars is None or bars.empty:
                continue

            trade = simulate_trade_for_session(
                asset_label=asset,
                trade_date=d,
                bars=bars,
                or_high=float(row.or_high),
                or_low=float(row.or_low),
                or_range=float(row.or_range),
                buffer_pct=buffer_pct,
                entry_start_minute=int(row.entry_start_minute),
            )
            if trade is None:
                continue

            trade["or_mode"] = strategy_mode
            trade["or_duration_bars"] = int(row.or_duration_bars)
            trade["buffer_pct"] = buffer_pct
            trade["pboc_filter"] = bool(config["pboc_filter"])
            trade["volume_filter"] = bool(config["volume_filter"])
            trade["atr_filter"] = bool(config["atr_filter"])
            trade["sizing_mode"] = config["sizing_mode"]
            records.append(trade)

    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records).sort_values(["entry_time", "asset", "or_mode"]).reset_index(drop=True)
    return out


def run_config(
    config: dict,
    asset_packs_by_mode: Dict[str, Dict[str, dict]],
    xau_move_map: Dict[date, float],
    volume_threshold: float,
) -> pd.DataFrame:
    mode = str(config["strategy_mode"])
    if mode == "asia_combined_5m_15m":
        t5 = run_config_single_mode(
            config=config,
            asset_packs=asset_packs_by_mode["asia_5m"],
            xau_move_map=xau_move_map,
            volume_threshold=volume_threshold,
            strategy_mode="asia_5m",
        )
        t15 = run_config_single_mode(
            config=config,
            asset_packs=asset_packs_by_mode["asia_15m"],
            xau_move_map=xau_move_map,
            volume_threshold=volume_threshold,
            strategy_mode="asia_15m",
        )
        if t5.empty and t15.empty:
            return pd.DataFrame()
        if t5.empty:
            return t15.sort_values(["entry_time", "asset", "or_mode"]).reset_index(drop=True)
        if t15.empty:
            return t5.sort_values(["entry_time", "asset", "or_mode"]).reset_index(drop=True)
        return pd.concat([t5, t15], ignore_index=True).sort_values(
            ["entry_time", "asset", "or_mode"]
        ).reset_index(drop=True)

    if mode not in asset_packs_by_mode:
        raise ValueError(f"Unknown strategy mode: {mode}")
    return run_config_single_mode(
        config=config,
        asset_packs=asset_packs_by_mode[mode],
        xau_move_map=xau_move_map,
        volume_threshold=volume_threshold,
        strategy_mode=mode,
    )


def size_trades_asset(trades: pd.DataFrame, mode: str, start_equity: float) -> pd.DataFrame:
    if trades.empty:
        return trades.assign(qty=pd.Series(dtype=float), pnl_usd=pd.Series(dtype=float))

    t = trades.sort_values("entry_time").copy()
    equity = float(start_equity)
    qty_list: List[float] = []
    pnl_list: List[float] = []
    eq_after: List[float] = []

    for row in t.itertuples(index=False):
        if mode == "risk_1pct":
            risk_usd = equity * 0.01
            qty = risk_usd / max(float(row.stop_distance), 1e-12)
        elif mode == "fixed_1000":
            qty = 1000.0 / max(float(row.entry_price), 1e-12)
        else:
            raise ValueError(f"Unknown sizing mode: {mode}")

        pnl = float(row.unit_pnl) * qty
        equity += pnl
        qty_list.append(qty)
        pnl_list.append(pnl)
        eq_after.append(equity)

    t["qty"] = qty_list
    t["pnl_usd"] = pnl_list
    t["equity_after_trade"] = eq_after
    return t


def size_trades_portfolio(trades: pd.DataFrame, mode: str, start_equity: float) -> pd.DataFrame:
    if trades.empty:
        return trades.assign(qty=pd.Series(dtype=float), pnl_usd=pd.Series(dtype=float))

    t = trades.sort_values(["entry_time", "asset"]).copy().reset_index(drop=True)
    t["trade_id"] = np.arange(len(t))

    equity = float(start_equity)
    exit_heap: List[Tuple[pd.Timestamp, int, float]] = []
    qty_map: Dict[int, float] = {}
    pnl_map: Dict[int, float] = {}
    eq_entry_map: Dict[int, float] = {}

    for row in t.itertuples(index=False):
        while exit_heap and exit_heap[0][0] <= row.entry_time:
            _, _, realized_pnl = heapq.heappop(exit_heap)
            equity += realized_pnl

        eq_entry_map[row.trade_id] = equity
        if mode == "risk_1pct":
            risk_usd = equity * 0.01
            qty = risk_usd / max(float(row.stop_distance), 1e-12)
        elif mode == "fixed_1000":
            qty = 1000.0 / max(float(row.entry_price), 1e-12)
        else:
            raise ValueError(f"Unknown sizing mode: {mode}")

        pnl = float(row.unit_pnl) * qty
        qty_map[row.trade_id] = qty
        pnl_map[row.trade_id] = pnl
        heapq.heappush(exit_heap, (row.exit_time, int(row.trade_id), pnl))

    while exit_heap:
        _, _, realized_pnl = heapq.heappop(exit_heap)
        equity += realized_pnl

    t["qty"] = t["trade_id"].map(qty_map)
    t["pnl_usd"] = t["trade_id"].map(pnl_map)
    t["equity_at_entry"] = t["trade_id"].map(eq_entry_map)
    t = t.drop(columns=["trade_id"])
    return t


def build_daily_equity_curve(
    dates: List[date],
    trades: pd.DataFrame,
    start_equity: float,
) -> pd.DataFrame:
    daily = pd.DataFrame({"date": dates})
    if trades.empty:
        daily["pnl_usd"] = 0.0
        daily["equity"] = float(start_equity)
        return daily

    pnl_by_day = trades.groupby(trades["exit_time"].dt.date)["pnl_usd"].sum()
    daily["pnl_usd"] = daily["date"].map(pnl_by_day).fillna(0.0)
    daily["equity"] = float(start_equity) + daily["pnl_usd"].cumsum()
    return daily


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())


def summarize_metrics(trades: pd.DataFrame, daily_eq: pd.DataFrame, start_equity: float) -> dict:
    total_trades = int(len(trades))
    gross_profit = float(trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum()) if total_trades else 0.0
    gross_loss = float(trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].sum()) if total_trades else 0.0
    winners = trades[trades["pnl_usd"] > 0] if total_trades else pd.DataFrame()
    losers = trades[trades["pnl_usd"] < 0] if total_trades else pd.DataFrame()

    win_rate = (len(winners) / total_trades * 100.0) if total_trades else 0.0
    avg_winner = float(winners["pnl_usd"].mean()) if not winners.empty else 0.0
    avg_loser = float(losers["pnl_usd"].mean()) if not losers.empty else 0.0
    pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else (math.inf if gross_profit > 0 else 0.0)

    rets = daily_eq["equity"].pct_change().dropna()
    if rets.empty or float(rets.std(ddof=0)) == 0.0:
        sharpe = 0.0
    else:
        sharpe = float((rets.mean() / rets.std(ddof=0)) * np.sqrt(365.0))

    max_dd = compute_max_drawdown(daily_eq["equity"])
    ending_equity = float(daily_eq["equity"].iloc[-1]) if not daily_eq.empty else float(start_equity)
    total_return_pct = ((ending_equity / float(start_equity)) - 1.0) * 100.0

    if trades.empty:
        best_month = ""
        worst_month = ""
        best_month_pnl = 0.0
        worst_month_pnl = 0.0
    else:
        monthly = trades.groupby(trades["exit_time"].dt.strftime("%Y-%m"))["pnl_usd"].sum().sort_index()
        best_month = str(monthly.idxmax()) if not monthly.empty else ""
        worst_month = str(monthly.idxmin()) if not monthly.empty else ""
        best_month_pnl = float(monthly.max()) if not monthly.empty else 0.0
        worst_month_pnl = float(monthly.min()) if not monthly.empty else 0.0

    return {
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "avg_winner_usd": avg_winner,
        "avg_loser_usd": avg_loser,
        "profit_factor": pf,
        "sharpe_365d": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
        "ending_equity": ending_equity,
        "total_return_pct": total_return_pct,
        "best_month": best_month,
        "best_month_pnl_usd": best_month_pnl,
        "worst_month": worst_month,
        "worst_month_pnl_usd": worst_month_pnl,
    }


def day_of_week_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["day_of_week", "trades", "win_rate_pct", "total_pnl_usd", "avg_pnl_usd"])
    g = (
        trades.assign(day_of_week=trades["entry_time"].dt.day_name())
        .groupby("day_of_week")
        .agg(
            trades=("pnl_usd", "size"),
            win_rate_pct=("pnl_usd", lambda x: (x.gt(0).mean() * 100.0) if len(x) else 0.0),
            total_pnl_usd=("pnl_usd", "sum"),
            avg_pnl_usd=("pnl_usd", "mean"),
        )
        .reset_index()
    )
    order = pd.Categorical(g["day_of_week"], categories=DAY_ORDER, ordered=True)
    g = g.assign(_order=order).sort_values("_order").drop(columns="_order")
    return g


def monthly_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["month", "trades", "total_pnl_usd", "win_rate_pct"])
    g = (
        trades.assign(month=trades["exit_time"].dt.strftime("%Y-%m"))
        .groupby("month")
        .agg(
            trades=("pnl_usd", "size"),
            total_pnl_usd=("pnl_usd", "sum"),
            win_rate_pct=("pnl_usd", lambda x: x.gt(0).mean() * 100.0),
        )
        .reset_index()
        .sort_values("month")
    )
    return g


def filter_on_off_comparison(metrics_df: pd.DataFrame, filter_col: str) -> pd.DataFrame:
    rows = []
    for state in [False, True]:
        subset = metrics_df[metrics_df[filter_col] == state]
        rows.append(
            {
                "filter_name": filter_col,
                "filter_state": "on" if state else "off",
                "config_count": int(len(subset)),
                "avg_total_return_pct": float(subset["total_return_pct"].mean()) if not subset.empty else np.nan,
                "median_total_return_pct": float(subset["total_return_pct"].median()) if not subset.empty else np.nan,
                "avg_sharpe_365d": float(subset["sharpe_365d"].mean()) if not subset.empty else np.nan,
                "avg_profit_factor": float(subset["profit_factor"].replace(np.inf, np.nan).mean())
                if not subset.empty
                else np.nan,
                "avg_total_trades": float(subset["total_trades"].mean()) if not subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_configs(strategy_mode: str) -> List[dict]:
    cfgs = []
    for buffer_pct, sizing_mode, pboc, volf, atrf in itertools.product(
        BUFFER_PCTS, SIZING_MODES, [False, True], [False, True], [False, True]
    ):
        cfg = {
            "strategy_mode": strategy_mode,
            "buffer_pct": int(buffer_pct),
            "sizing_mode": sizing_mode,
            "pboc_filter": bool(pboc),
            "volume_filter": bool(volf),
            "atr_filter": bool(atrf),
        }
        cfg["config_id"] = (
            f"mode-{strategy_mode}_buf{buffer_pct}_size-{sizing_mode}_pboc-{int(pboc)}_vol-{int(volf)}_atr-{int(atrf)}"
        )
        cfgs.append(cfg)
    return cfgs


def save_equity_plot(equity_curves: pd.DataFrame, portfolio_metrics: pd.DataFrame, out_path: Path) -> None:
    if equity_curves.empty or portfolio_metrics.empty:
        return
    top_configs = (
        portfolio_metrics.sort_values("ending_equity", ascending=False).head(6)["config_id"].tolist()
    )
    plt.figure(figsize=(13, 7))
    for cfg in top_configs:
        sub = equity_curves[equity_curves["config_id"] == cfg].sort_values("date")
        plt.plot(sub["date"], sub["equity"], label=cfg, linewidth=1.4)
    plt.title("Asia ORB Portfolio Equity Curves (Top 6 Configs)")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_or_range_plot(or_summary: pd.DataFrame, out_path: Path) -> None:
    if or_summary.empty:
        return
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    x = np.arange(len(or_summary))

    ax1.bar(x, or_summary["avg_r_multiple"], color="#1f77b4", alpha=0.8, label="Avg R")
    ax2.plot(x, or_summary["win_rate_pct"], color="#d62728", marker="o", linewidth=1.8, label="Win Rate")

    ax1.set_xticks(x)
    ax1.set_xticklabels(or_summary["or_range_bin"].astype(str), rotation=20, ha="right")
    ax1.set_ylabel("Avg R-multiple")
    ax2.set_ylabel("Win Rate (%)")
    ax1.set_title("OR Range Size vs Outcome (All Config Trades)")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start_d = date.fromisoformat(args.start_date)
    end_d = date.fromisoformat(args.end_date)
    if end_d < start_d:
        raise ValueError("end-date must be >= start-date")

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Resolving Binance symbols...")
    resolved_assets, notes = resolve_assets(ASSET_SPECS)
    for n in notes:
        print(f"  - {n}")

    raw_data_by_asset: Dict[str, pd.DataFrame] = {}
    print("Loading 5m candles and building session features...")
    for asset_label, meta in resolved_assets.items():
        print(f"  [{asset_label}] requested={meta['requested_symbol']} resolved={meta['symbol']}")
        if not meta["available"]:
            continue

        raw = load_or_fetch_5m(
            symbol=meta["symbol"],
            start_d=start_d,
            end_d=end_d,
            cache_dir=cache_dir,
            refresh=args.refresh_data,
            sleep_sec=args.sleep_sec,
        )
        raw_data_by_asset[asset_label] = raw
        print(f"    bars={len(raw):,}")

    asset_packs_by_mode: Dict[str, Dict[str, dict]] = {mode: {} for mode in OR_BARS_BY_MODE}
    print(f"Building session packs for mode: {args.strategy_mode}")
    for mode, or_bars in OR_BARS_BY_MODE.items():
        for asset_label, meta in resolved_assets.items():
            if not meta["available"]:
                asset_packs_by_mode[mode][asset_label] = {
                    "available": False,
                    "sessions": pd.DataFrame(),
                    "bars_by_date": {},
                    "symbol": None,
                    "is_metal": bool(meta["is_metal"]),
                }
                continue
            sessions, bars_by_date = build_session_pack(raw_data_by_asset[asset_label], or_duration_bars=or_bars)
            sessions = sessions[(sessions["date"] >= start_d) & (sessions["date"] <= end_d)].reset_index(drop=True)
            asset_packs_by_mode[mode][asset_label] = {
                "available": True,
                "sessions": sessions,
                "bars_by_date": bars_by_date,
                "symbol": meta["symbol"],
                "is_metal": bool(meta["is_metal"]),
            }
            if mode == args.strategy_mode or (
                args.strategy_mode == "asia_combined_5m_15m" and mode in {"asia_5m", "asia_15m"}
            ):
                print(f"    [{mode}] {asset_label}: sessions={len(sessions):,}")

    xau_move_map: Dict[date, float] = {}
    xau_30m = asset_packs_by_mode["asia_30m"].get("XAU")
    if xau_30m is not None and not xau_30m["sessions"].empty:
        xau_df = xau_30m["sessions"]
        xau_move_map = dict(zip(xau_df["date"], xau_df["or_move_pct"]))

    configs = make_configs(args.strategy_mode)
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]

    session_calendar = calendar_dates(start_d, end_d)
    portfolio_metrics_rows: List[dict] = []
    asset_metrics_rows: List[dict] = []
    all_trades: List[pd.DataFrame] = []
    equity_curves_rows: List[pd.DataFrame] = []
    dow_rows: List[pd.DataFrame] = []
    monthly_rows: List[pd.DataFrame] = []

    print(f"Running config sweep: {len(configs)} configurations")
    print(f"Strategy mode: {args.strategy_mode}")
    print(f"Volume filter threshold: {args.volume_threshold:.2f}x")
    for i, cfg in enumerate(configs, 1):
        print(
            f"  [{i}/{len(configs)}] {cfg['config_id']}",
            flush=True,
        )
        base_trades = run_config(
            cfg,
            asset_packs_by_mode=asset_packs_by_mode,
            xau_move_map=xau_move_map,
            volume_threshold=args.volume_threshold,
        )
        if base_trades.empty:
            sized_port = pd.DataFrame(columns=list(base_trades.columns) + ["qty", "pnl_usd"])
        else:
            sized_port = size_trades_portfolio(base_trades, mode=cfg["sizing_mode"], start_equity=args.start_equity)

        daily_port = build_daily_equity_curve(session_calendar, sized_port, start_equity=args.start_equity)
        m_port = summarize_metrics(sized_port, daily_port, start_equity=args.start_equity)
        m_port.update(cfg)
        m_port["volume_threshold"] = float(args.volume_threshold)
        portfolio_metrics_rows.append(m_port)

        ec = daily_port.copy()
        ec["config_id"] = cfg["config_id"]
        equity_curves_rows.append(ec)

        dow_port = day_of_week_performance(sized_port)
        if not dow_port.empty:
            dow_port["scope"] = "portfolio"
            dow_port["asset"] = "ALL"
            for k, v in cfg.items():
                dow_port[k] = v
            dow_port["volume_threshold"] = float(args.volume_threshold)
            dow_rows.append(dow_port)

        monthly_port = monthly_performance(sized_port)
        if not monthly_port.empty:
            monthly_port["scope"] = "portfolio"
            monthly_port["asset"] = "ALL"
            for k, v in cfg.items():
                monthly_port[k] = v
            monthly_port["volume_threshold"] = float(args.volume_threshold)
            monthly_rows.append(monthly_port)

        for asset_label in [a.label for a in ASSET_SPECS]:
            if sized_port.empty:
                asset_trades = pd.DataFrame(columns=sized_port.columns)
            else:
                asset_trades = sized_port[sized_port["asset"] == asset_label].copy()

            if asset_trades.empty and not base_trades.empty:
                base_asset = base_trades[base_trades["asset"] == asset_label].copy()
                if not base_asset.empty:
                    asset_trades = size_trades_asset(
                        base_asset, mode=cfg["sizing_mode"], start_equity=args.start_equity
                    )
            elif not asset_trades.empty:
                asset_trades = size_trades_asset(
                    base_trades[base_trades["asset"] == asset_label].copy(),
                    mode=cfg["sizing_mode"],
                    start_equity=args.start_equity,
                )

            daily_asset = build_daily_equity_curve(session_calendar, asset_trades, start_equity=args.start_equity)
            m_asset = summarize_metrics(asset_trades, daily_asset, start_equity=args.start_equity)
            m_asset.update(cfg)
            m_asset["volume_threshold"] = float(args.volume_threshold)
            m_asset["asset"] = asset_label
            m_asset["available_on_binance"] = bool(resolved_assets.get(asset_label, {}).get("available", False))
            m_asset["resolved_symbol"] = resolved_assets.get(asset_label, {}).get("symbol")
            asset_metrics_rows.append(m_asset)

            dow_asset = day_of_week_performance(asset_trades)
            if not dow_asset.empty:
                dow_asset["scope"] = "asset"
                dow_asset["asset"] = asset_label
                for k, v in cfg.items():
                    dow_asset[k] = v
                dow_asset["volume_threshold"] = float(args.volume_threshold)
                dow_rows.append(dow_asset)

            monthly_asset = monthly_performance(asset_trades)
            if not monthly_asset.empty:
                monthly_asset["scope"] = "asset"
                monthly_asset["asset"] = asset_label
                for k, v in cfg.items():
                    monthly_asset[k] = v
                monthly_asset["volume_threshold"] = float(args.volume_threshold)
                monthly_rows.append(monthly_asset)

        if not sized_port.empty:
            tlog = sized_port.copy()
            for k, v in cfg.items():
                tlog[k] = v
            tlog["volume_threshold"] = float(args.volume_threshold)
            tlog["config_id"] = cfg["config_id"]
            all_trades.append(tlog)

    portfolio_metrics = pd.DataFrame(portfolio_metrics_rows)
    asset_metrics = pd.DataFrame(asset_metrics_rows)
    trade_log = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_curves = pd.concat(equity_curves_rows, ignore_index=True) if equity_curves_rows else pd.DataFrame()
    dow_perf = pd.concat(dow_rows, ignore_index=True) if dow_rows else pd.DataFrame()
    monthly_perf_df = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()

    if not portfolio_metrics.empty:
        portfolio_metrics["config_id"] = portfolio_metrics.apply(
            lambda r: (
                f"mode-{r['strategy_mode']}_buf{int(r['buffer_pct'])}_size-{r['sizing_mode']}_pboc-{int(bool(r['pboc_filter']))}"
                f"_vol-{int(bool(r['volume_filter']))}_atr-{int(bool(r['atr_filter']))}"
            ),
            axis=1,
        )
    if not asset_metrics.empty:
        asset_metrics["config_id"] = asset_metrics.apply(
            lambda r: (
                f"mode-{r['strategy_mode']}_buf{int(r['buffer_pct'])}_size-{r['sizing_mode']}_pboc-{int(bool(r['pboc_filter']))}"
                f"_vol-{int(bool(r['volume_filter']))}_atr-{int(bool(r['atr_filter']))}"
            ),
            axis=1,
        )

    filter_cmp_frames = []
    if not portfolio_metrics.empty:
        for col in ["pboc_filter", "volume_filter", "atr_filter"]:
            filter_cmp_frames.append(filter_on_off_comparison(portfolio_metrics, col))
    filter_comparison = pd.concat(filter_cmp_frames, ignore_index=True) if filter_cmp_frames else pd.DataFrame()

    or_range_summary = pd.DataFrame()
    if not trade_log.empty:
        tmp = trade_log[["or_range_pct_of_entry", "r_multiple"]].dropna().copy()
        if len(tmp) >= 20:
            tmp["or_range_bin"] = pd.qcut(tmp["or_range_pct_of_entry"], q=5, duplicates="drop")
            or_range_summary = (
                tmp.groupby("or_range_bin", observed=False)
                .agg(
                    trades=("r_multiple", "size"),
                    win_rate_pct=("r_multiple", lambda x: x.gt(0).mean() * 100.0),
                    avg_r_multiple=("r_multiple", "mean"),
                    median_r_multiple=("r_multiple", "median"),
                )
                .reset_index()
            )

    portfolio_metrics.to_csv(output_dir / "portfolio_metrics.csv", index=False)
    asset_metrics.to_csv(output_dir / "asset_metrics.csv", index=False)
    trade_log.to_csv(output_dir / "trade_log.csv", index=False)
    dow_perf.to_csv(output_dir / "day_of_week_performance.csv", index=False)
    monthly_perf_df.to_csv(output_dir / "monthly_performance.csv", index=False)
    filter_comparison.to_csv(output_dir / "filter_on_off_comparison.csv", index=False)
    or_range_summary.to_csv(output_dir / "or_range_distribution_summary.csv", index=False)
    equity_curves.to_csv(output_dir / "portfolio_equity_curves.csv", index=False)

    if not args.no_plots:
        save_equity_plot(
            equity_curves=equity_curves,
            portfolio_metrics=portfolio_metrics,
            out_path=output_dir / "equity_curve_top_configs.png",
        )
        save_or_range_plot(or_range_summary, out_path=output_dir / "or_range_vs_outcome.png")

    print("\nDone.")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Portfolio metrics rows: {len(portfolio_metrics):,}")
    print(f"Asset metrics rows: {len(asset_metrics):,}")
    print(f"Trade log rows: {len(trade_log):,}")
    if not portfolio_metrics.empty:
        best = portfolio_metrics.sort_values("ending_equity", ascending=False).iloc[0]
        worst = portfolio_metrics.sort_values("ending_equity", ascending=True).iloc[0]
        print(
            f"Best config: {best['config_id']}  ending_equity={best['ending_equity']:.2f}  "
            f"return={best['total_return_pct']:.2f}%"
        )
        print(
            f"Worst config: {worst['config_id']}  ending_equity={worst['ending_equity']:.2f}  "
            f"return={worst['total_return_pct']:.2f}%"
        )


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
