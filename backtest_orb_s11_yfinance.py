#!/usr/bin/env python3
"""
ORB BACKTEST S11 — yfinance Primary + Combined Shared-Wallet (5min + 15min)
============================================================================
Motivation (from S10):
  - Binance tokenised equities only have 10-17d of 5m history → SMA20
    cannot compute → equities never trade.  yfinance v1.2.0 fixes this (60d).
  - Data source switched to yfinance Ticker.history() for all assets except SOL.

New in S11: CombinedUS_5and15 — shared wallet running BOTH OR durations live:
  - At 9:35 ET : fire best 5min OR breakout (asset A)
  - At 9:45+ ET: fire best 15min OR breakout from a DIFFERENT ASSET (asset B)
  - Both positions draw from the same balance simultaneously.
  - Risk is split: each trade uses daily_risk_budget / 2 (parallel model).
  - Asset-class exclusion prevents e.g. 2 equity positions on the same day.

Param sets compared:
  S9  : min_score=8, vol_score_mult=2.0, risk_pct=5.0%  (high conviction)
  S10 : min_score=7, vol_score_mult=1.5, risk_pct=7.0%  (high frequency)
  S12 : min_score=8, vol_score_mult=1.3, risk_pct=5.0%  (relaxed vol gate, same quality filter)

Variants:
  US_5min          — 5min OR at NYSE open
  US_15min         — 15min OR at NYSE open
  CombinedUS_5and15 — shared wallet: 5min pick + 15min pick per day
  Asia_5min        — Tokyo open, metals & SOL only

Assets:
  TradFi equities : TSLA AMZN PLTR INTC HOOD COIN MSTR CRCL (yfinance)
  Metals futures  : GC=F (XAU)  SI=F (XAG)                  (yfinance)
  Crypto          : SOL (Binance USDT perp — native source)

Run:  python backtests/backtest_orb_s11_yfinance.py
"""
import sys, io, os, time as _time
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date as date_type
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

# ====================================================================
# PARAM SETS — S9 (conservative) vs S10 (aggressive)
# ====================================================================
BASE_CFG = dict(
    starting_balance    = 100.0,
    max_leverage        = 10,
    commission_pct      = 0.05,

    tp1_mult            = 1.0,
    tp2_mult            = 2.0,
    tp1_close_frac      = 0.50,

    min_or_range_pct    = 0.10,
    max_or_range_pct    = 15.0,
    max_or_atr_mult     = 2.0,

    max_gap_pct         = 0,          # W4 disabled (24h markets; irrelevant for equities too)
    breakout_vol_mult   = 1.5,        # W2
    entry_cutoff_minutes= 90,         # W7
    breakeven_after_tp1 = True,

    sma_period          = 20,
    max_daily_trades    = 2,
    daily_risk_budget   = 7.5,
    kelly_lookback      = 20,
    kelly_fraction      = 0.5,
    fetch_days          = 60,
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
# VARIANTS  (US_5min, US_15min, Asia_5min)
# ====================================================================
VARIANTS = [
    {
        "name":             "US_5min",
        "description":      "US Market Open — 5min OR (NYSE 9:30 ET)",
        "or_duration_bars": 1,
        "session_type":     "us",
        "force_close_hour": 21,   # 16:00 ET = 21:00 UTC
    },
    {
        "name":             "US_15min",
        "description":      "US Market Open — 15min OR (NYSE 9:30 ET)",
        "or_duration_bars": 3,
        "session_type":     "us",
        "force_close_hour": 21,
    },
    {
        "name":             "Asia_5min",
        "description":      "Tokyo Open — 5min OR — metals & SOL only",
        "or_duration_bars": 1,
        "session_type":     "asia",
        "force_close_hour": 6,
    },
]

# ====================================================================
# ASSET UNIVERSE — yfinance PRIMARY, Binance fallback except SOL
# ====================================================================
ASSETS = [
    # US Equities (yfinance primary, Asia session NOT applicable for real equities)
    dict(label="TSLA", yf_sym="TSLA",  bn_sym="TSLA/USDT:USDT", tier=1, primary="yf", asia_ok=False),
    dict(label="COIN", yf_sym="COIN",  bn_sym="COIN/USDT:USDT", tier=1, primary="yf", asia_ok=False),
    dict(label="HOOD", yf_sym="HOOD",  bn_sym="HOOD/USDT:USDT", tier=1, primary="yf", asia_ok=False),
    dict(label="MSTR", yf_sym="MSTR",  bn_sym="MSTR/USDT:USDT", tier=1, primary="yf", asia_ok=False),
    dict(label="AMZN", yf_sym="AMZN",  bn_sym="AMZN/USDT:USDT", tier=2, primary="yf", asia_ok=False),
    dict(label="PLTR", yf_sym="PLTR",  bn_sym="PLTR/USDT:USDT", tier=2, primary="yf", asia_ok=False),
    dict(label="INTC", yf_sym="INTC",  bn_sym="INTC/USDT:USDT", tier=2, primary="yf", asia_ok=False),
    dict(label="CRCL", yf_sym="CRCL",  bn_sym="CRCL/USDT:USDT", tier=2, primary="yf", asia_ok=False),
    # Metals futures (yfinance primary — GC=F/SI=F; genuine 24h Asia liquidity)
    dict(label="XAU",  yf_sym="GC=F",  bn_sym="XAU/USDT:USDT",  tier=3, primary="yf", asia_ok=True),
    dict(label="XAG",  yf_sym="SI=F",  bn_sym="XAG/USDT:USDT",  tier=3, primary="yf", asia_ok=True),
    # Crypto (Binance native — SOL/USDT perps; yfinance has SOL-USD but Binance preferred)
    dict(label="SOL",  yf_sym="SOL-USD", bn_sym="SOL/USDT:USDT", tier=1, primary="bn", asia_ok=True),
]

ASSET_CLASS = {
    "TSLA": "equity", "COIN": "equity", "HOOD": "equity", "MSTR": "equity",
    "AMZN": "equity", "PLTR": "equity", "INTC": "equity", "CRCL": "equity",
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
# DATA FETCHING — yfinance 1.2.0 Ticker().history() API
# ====================================================================
def fetch_yfinance_5m(yf_sym: str, days: int = 60):
    """Fetch 5m bars via yfinance 1.2.0 Ticker.history() — returns UTC-normalised df."""
    try:
        t   = yf.Ticker(yf_sym)
        raw = t.history(interval="5m", period=f"{days}d")
        if raw is None or raw.empty:
            return None, "yf_empty"
        # Normalise columns
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"open": "o", "high": "h", "low": "l",
                                  "close": "c", "volume": "v"})
        # Ensure UTC
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")
        raw["time"] = raw.index
        raw["ts"]   = raw.index.astype("int64") // 10**6
        raw = raw.reset_index(drop=True)
        raw = raw.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        # Drop non-trading bars (pre/post market) — keep rows with v > 0
        raw = raw[raw["v"] > 0].reset_index(drop=True)
        return raw, "yfinance"
    except Exception as e:
        return None, f"yf_err:{e}"


def fetch_daily_yf(yf_sym: str, sma_period: int = 20):
    """Fetch daily OHLCV via yfinance 1.2.0, compute SMA20 and ATR20.
    Returns (sma_data, atr_data, close_data) keyed by date."""
    sma_data, atr_data, close_data = {}, {}, {}
    try:
        t   = yf.Ticker(yf_sym)
        raw = t.history(interval="1d", period="6mo")
        if raw is None or raw.empty:
            return sma_data, atr_data, close_data
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"open": "open_", "high": "high_", "low": "low_",
                                   "close": "close_"})
        # Normalize to consistent col names
        raw = raw.rename(columns={"open_": "open", "high_": "high",
                                   "low_": "low", "close_": "close"})
        raw["sma"] = raw["close"].rolling(sma_period, min_periods=sma_period).mean()
        raw["prev_close"] = raw["close"].shift(1)
        def _tr(r):
            return max(
                r["high"] - r["low"],
                abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
                abs(r["low"]  - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            )
        raw["tr"]     = raw.apply(_tr, axis=1)
        raw["atr20"]  = raw["tr"].rolling(sma_period, min_periods=5).mean()

        raw["sma_prev"]   = raw["sma"].shift(1)
        raw["atr20_prev"] = raw["atr20"].shift(1)
        raw["close_prev"] = raw["close"].shift(1)

        if raw.index.tz is None:
            dates = raw.index.date
        else:
            dates = raw.index.tz_convert("UTC").date

        for i, (idx, row) in enumerate(raw.iterrows()):
            if pd.isna(row.get("sma_prev")):
                continue
            d = dates[i]
            sma_data[d]   = {"sma": float(row["sma_prev"]), "close": float(row["close_prev"])}
            close_data[d] = float(row["close_prev"])
            if pd.notna(row.get("atr20_prev")):
                atr_data[d] = float(row["atr20_prev"])
    except Exception as e:
        print(f"      [daily_yf WARN] {yf_sym}: {e}")
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


_ex = None
def _get_binance():
    global _ex
    if _ex is None:
        _ex = ccxt.binance({"options": {"defaultType": "future"}})
    return _ex


def fetch_binance_5m(bn_sym: str, days: int = 60):
    ex    = _get_binance()
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms= int(datetime.now(timezone.utc).timestamp() * 1000)
    all_c, retries = [], 0
    while since < now_ms:
        try:
            batch = ex.fetch_ohlcv(bn_sym, "5m", since=since, limit=1500)
        except Exception as e:
            retries += 1
            if retries > 3:
                return None, f"bn_err:{e}"
            _time.sleep(1.5)
            continue
        if not batch:
            break
        all_c.extend(batch)
        since = batch[-1][0] + 1
        _time.sleep(0.08)
    if not all_c:
        return None, "bn_empty"
    df = pd.DataFrame(all_c, columns=["ts", "o", "h", "l", "c", "v"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df, "binance"


def fetch_5m(asset, days: int = 60):
    """Primary routing: yfinance for equities/metals, Binance for SOL; fallback on error."""
    if asset.get("primary") == "bn":
        # SOL: Binance native preferred
        df, src = fetch_binance_5m(asset["bn_sym"], days)
        if df is not None and len(df) > 50:
            return df, src
        # Fallback to yfinance
        df, src = fetch_yfinance_5m(asset["yf_sym"], days)
        return df, src
    else:
        # Equities + Metals: yfinance preferred (proper 60d with actual market hours)
        df, src = fetch_yfinance_5m(asset["yf_sym"], days)
        if df is not None and len(df) > 50:
            return df, src
        # Fallback to Binance tokenised (may be thin)
        df, src = fetch_binance_5m(asset["bn_sym"], days)
        return df, src

# ====================================================================
# SESSION BUILDER
# ====================================================================
def build_sessions_variant(df, asset, atr_data, close_data, variant):
    or_duration_bars = variant["or_duration_bars"]
    session_type     = variant["session_type"]
    force_close_hour = variant["force_close_hour"]

    df["date"] = df["time"].dt.date

    daily_vol = df.groupby("date")["v"].sum().reset_index()
    daily_vol.columns = ["date", "day_vol"]
    daily_vol["vol_ma"] = daily_vol["day_vol"].rolling(20, min_periods=3).mean()
    vol_map  = dict(zip(daily_vol["date"],
                        zip(daily_vol["day_vol"], daily_vol["vol_ma"])))

    sessions       = []
    prev_or_range  = None
    prev_close_5m  = None
    monday_only    = variant.get("monday_only", False)

    for d, grp in df.groupby("date"):
        dow = pd.Timestamp(d).dayofweek
        if dow >= 5:
            continue
        if monday_only and dow != 0:
            continue

        # US equities: only US session; metals/crypto: Asia eligible
        if session_type == "asia" and not asset.get("asia_ok", False):
            continue

        if session_type == "us":
            day_slots = [("us", _us_market_open_utc_hour(d), 30, force_close_hour)]
        elif session_type == "asia":
            day_slots = [("asia", 0, 0, force_close_hour)]
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

            dv, dv_ma = vol_map.get(d, (0, None))
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
                "vol_above_avg":      (dv_ma is not None and dv_ma > 0),  # computed with vol_score_mult in score
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
# SCORING  (unchanged from S9/S10)
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

    # +2 Volume above average (uses per-cfg vol_score_mult)
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
            fee  = tr.tp1 * tr.size_full * frac * (comm)
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
            if cfg["breakout_vol_mult"] > 0 and idx >= 5:
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

        picks         = []
        picked_assets = set()
        picked_classes= set()

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
# COMBINED SHARED WALLET — 5min OR + 15min OR on same day, same capital
# ====================================================================
def _best_candidate(sessions_for_day, sma_data_all, cfg, exclude_labels=None, exclude_classes=None):
    """Score all sessions for one day, return best qualifying (score, dir, sess, reasons)."""
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
    """
    Shared wallet: fire best qualifying 5min OR trade at ~9:35 ET, then fire
    best qualifying 15min OR trade from a DIFFERENT ASSET CLASS at ~9:45+ ET.
    Both positions draw risk from the same capital (parallel model):
      pnl_day = pnl_5min_trade + pnl_15min_trade
    Each trade risks per_trade_risk = daily_risk_budget / 2 when both fire,
    or full daily_risk_budget when only one fires.
    """
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

        # --- Pick 5min candidate -------------------------------------------
        pick5 = _best_candidate(sess5, sma_data_all, cfg)

        # --- Pick 15min candidate: must be different asset (and class) -------
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
        base_balance  = balance   # both trades start from the same base
        day_pnl       = 0.0

        for sc, dr, ss, rs in picks:
            # Parallel model: each trade draws from base_balance independently
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
        "final_balance":float(eq[-1]),
        "by_reason":    by_reason,
        "by_asset":     {k: len(v) for k, v in by_asset.items()},
        "avg_score":    float(np.mean([t.score for t in trades])),
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
          f"PF: {pf_s}  Sharpe: {stats['sharpe']:.2f}")
    print(f"    MaxDD: {stats['max_drawdown']:.1f}%  Return: {stats['total_return']:+.1f}%  "
          f"$100 -> ${stats['final_balance']:.2f}")
    reasons = "  ".join(f"{k}:{v}" for k, v in sorted(stats["by_reason"].items()))
    print(f"    Exits: {reasons}")
    if stats["by_asset"]:
        assets_str = ", ".join(f"{a}({n})" for a, n in sorted(stats["by_asset"].items(), key=lambda x: -x[1]))
        print(f"    Assets: {assets_str}")
    fs = result["filter_stats"]
    print(f"    ATR-filtered: {fs['atr']}  Score-skipped: {result['skipped_score']}d  No-setup: {result['no_setup_days']}d")


def print_per_asset_table(per_asset_results, sma_counts, tag):
    print(f"  [{tag}] Per-Asset Breakdown")
    print(f"    {'Asset':<7} {'SMAd':>5} {'Sess':>4} {'Trd':>3} {'WR':>6} {'PF':>6} {'Ret':>7} {'Final$':>7}")
    sec(72)
    for label in sorted(per_asset_results.keys()):
        data   = per_asset_results[label]
        trades = data["trades"]
        sma_d  = sma_counts.get(label, 0)
        if not trades:
            print(f"    {label:<7} {sma_d:>5} {data['sessions']:>4}  --    --     --      --    ${data['balance']:.2f}")
            continue
        s   = compute_stats(trades, data["equity_curve"], 100.0)
        pfs = f"{s['profit_factor']:.2f}" if s["profit_factor"] < 1e6 else "INF"
        print(f"    {label:<7} {sma_d:>5} {data['sessions']:>4} {s['trades']:>3} "
              f"{s['win_rate']:>5.1f}% {pfs:>6} {s['total_return']:>+6.1f}%  ${s['final_balance']:>6.2f}")

# ====================================================================
# MAIN
# ====================================================================
def main():
    bar()
    print("  ORB BACKTEST S11 — yfinance Primary: Full TradFi Equity Coverage")
    print("  Data: yfinance 1.2.0 (Ticker.history) — 60d 5m for equities & metals")
    print("  Param sets: S9 (min_score=8, vol=2.0x, risk=5%)  vs  S10 (min_score=7, vol=1.5x, risk=7%)")
    print("  Assets: TSLA AMZN PLTR INTC HOOD COIN MSTR CRCL | XAU XAG | SOL(Binance)")
    bar()
    print()

    # ── Phase 1: Fetch data ───────────────────────────────────────
    print("  Phase 1: Fetching 5m + daily SMA/ATR data...")
    sec()
    all_data       = {}
    sma_data_all   = {}
    atr_data_all   = {}
    close_data_all = {}
    sma_counts     = {}

    for asset in ASSETS:
        label  = asset["label"]
        yf_sym = asset["yf_sym"]
        print(f"    {label:<6}  5m...", end="", flush=True)
        df, src = fetch_5m(asset, BASE_CFG["fetch_days"])
        n = len(df) if df is not None else 0
        print(f" {n:>6,} [{src:<9}]  daily...", end="", flush=True)
        sma, atr, closes = fetch_daily_yf(yf_sym, BASE_CFG["sma_period"])
        if len(sma) == 0 and df is not None and len(df) > 50:
            sma, atr, closes = compute_daily_from_5m(df, BASE_CFG["sma_period"])
            print(f" sma={len(sma)}d  atr={len(atr)}d [5m fallback]")
        else:
            print(f" sma={len(sma)}d  atr={len(atr)}d")
        all_data[label]       = {"df": df, "source": src, "asset": asset}
        sma_data_all[label]   = sma
        atr_data_all[label]   = atr
        close_data_all[label] = closes
        sma_counts[label]     = len(sma)
        _time.sleep(0.15)
    print()

    # ── Phase 2: Build sessions & run variants × param sets ───────
    # Master results: results[pset_name][variant_name] = {...}
    all_results = {pname: {} for pname in PARAM_SETS}
    # Cache 5min and 15min US sessions for the combined wallet run
    _us5_by_date  = defaultdict(list)  # US_5min  sessions by date
    _us15_by_date = defaultdict(list)  # US_15min sessions by date

    for variant in VARIANTS:
        vname = variant["name"]
        print(f"  Building sessions: {vname}...")
        sec()

        all_sessions_by_date  = defaultdict(list)
        all_sessions_by_label = defaultdict(list)

        for label, info in all_data.items():
            df    = info["df"]
            asset = info["asset"]
            if df is None or len(df) < 50:
                print(f"    {label:<6}  SKIP (no data)"); continue
            if variant["session_type"] == "asia" and not asset.get("asia_ok", False):
                continue

            sessions = build_sessions_variant(
                df, asset, atr_data_all.get(label, {}),
                close_data_all.get(label, {}), variant
            )
            dates = set()
            for s in sessions:
                all_sessions_by_date[s["date"]].append(s)
                all_sessions_by_label[label].append(s)
                dates.add(s["date"])
                # Cache for combined wallet variant
                if vname == "US_5min":
                    _us5_by_date[s["date"]].append(s)
                elif vname == "US_15min":
                    _us15_by_date[s["date"]].append(s)
            if dates:
                ds, de = min(dates), max(dates)
                print(f"    {label:<6} {len(sessions):>3} sessions  {ds} -> {de}")

        print(f"    Total trading days: {len(all_sessions_by_date)}")
        print()

        for pname, cfg in PARAM_SETS.items():
            tag = f"{vname}/{pname}"
            print(f"    Running concentrated + per-asset: {tag}...")
            conc_result = run_concentrated(
                all_sessions_by_date, sma_data_all, atr_data_all,
                force_close_hour=variant["force_close_hour"], cfg=cfg
            )
            conc_stats = compute_stats(
                conc_result["trades"], conc_result["equity_curve"], cfg["starting_balance"]
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
        print()

    # ── Phase 2b: Combined shared-wallet (5min + 15min) ──────────
    print("  Building sessions: CombinedUS_5and15 (shared wallet)...")
    sec()
    n5  = sum(len(v) for v in _us5_by_date.values())
    n15 = sum(len(v) for v in _us15_by_date.values())
    print(f"    5min sessions cached : {n5}  across {len(_us5_by_date)} dates")
    print(f"    15min sessions cached: {n15}  across {len(_us15_by_date)} dates")
    print(f"    Parallel model: each trade risks daily_budget/2 from SAME base balance")
    print(f"    Asset-class exclusion: 5min pick blocks its class from 15min pool")
    print()
    for pname, cfg in PARAM_SETS.items():
        tag = f"CombinedUS_5and15/{pname}"
        print(f"    Running combined wallet: {tag}...")
        comb_result = run_combined_us_5_15(
            _us5_by_date, _us15_by_date, sma_data_all, cfg=cfg
        )
        comb_stats = compute_stats(
            comb_result["trades"], comb_result["equity_curve"], cfg["starting_balance"]
        )
        all_results[pname]["CombinedUS_5and15"] = {
            "variant": {"name": "CombinedUS_5and15",
                        "description": "Shared wallet: 5min + 15min OR same day",
                        "session_type": "combined_us", "force_close_hour": 21},
            "stats":   comb_stats,
            "result":  comb_result,
            "per_asset": {},
        }
    print()

    # ── Phase 3: Output ───────────────────────────────────────────
    bar()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║          S11 RESULTS — CONCENTRATED PORTFOLIO               ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    bar()
    print()

    for pname, pdata in all_results.items():
        cfg = PARAM_SETS[pname]
        bar()
        print(f"  PARAM SET: {cfg['label']}")
        bar()
        for vname, data in pdata.items():
            tag = f"{vname}/{pname}"
            print_concentrated_summary(data["stats"], data["result"], tag)
            print()

    # ── Per-asset breakdowns ─────────────────────────────────────
    bar()
    print("  PER-ASSET BREAKDOWN — US_5min variant")
    bar()
    for pname, pdata in all_results.items():
        if "US_5min" not in pdata:
            continue
        tag = f"US_5min/{pname}"
        print()
        print_per_asset_table(pdata["US_5min"]["per_asset"], sma_counts, tag)
        print()

    # ── Comparison table ─────────────────────────────────────────
    bar()
    print("  COMPARISON TABLE — ALL VARIANTS × BOTH PARAM SETS")
    bar()
    print(f"  {'Tag':<26} {'T':>4} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'Ret':>7} {'$End':>7}")
    sec()
    # S10 reference from prior run (Binance-only data)
    print(f"  {'S10ref/US_5min':<26} {'30':>4} {'43.3%':>6} {'2.24':>6} {'2.68':>7} {'-11.9%':>7} {'+35.5%':>7} {'$135.51':>7}")
    print(f"  {'S9ref/US_5min':<26} {'8':>4}  {'75.0%':>6} {'9.77':>6} {'5.05':>7} {'-5.8%':>7}  {'+37.5%':>7} {'$137.53':>7}")
    sec()
    ORDER = ["US_5min", "US_15min", "CombinedUS_5and15", "Asia_5min"]
    for pname, pdata in all_results.items():
        for vname in ORDER:
            if vname not in pdata:
                continue
            tag   = f"{pname}/{vname}"
            stats = pdata[vname]["stats"]
            if stats is None:
                print(f"  {tag:<26} {'0':>4}  {'—':>6} {'—':>6} {'—':>7} {'—':>7} {'—':>7} {'$100.00':>7}")
                continue
            pf_s = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 1e6 else "INF"
            marker = " <-- COMBINED" if "Combined" in vname else ""
            print(f"  {tag:<26} {stats['trades']:>4} {stats['win_rate']:>5.1f}% {pf_s:>6} "
                  f"{stats['sharpe']:>7.2f} {stats['max_drawdown']:>+6.1f}% "
                  f"{stats['total_return']:>+6.1f}% ${stats['final_balance']:>6.2f}{marker}")

    # ── Combined wallet trade log ─────────────────────────────────
    bar()
    print("  COMBINED WALLET TRADE LOG (5min + 15min OR, same day, same capital)")
    sec()
    for pname, pdata in all_results.items():
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
                  f"{e['exit_reason']:<4}  R={e['r_mult']:+.2f}  pnl={e['pnl_usd']:+.2f}")
        print()

    # ── Summary key findings ──────────────────────────────────────
    bar()
    print("  KEY FINDINGS — S11 SUMMARY")
    sec()
    for pname, pdata in all_results.items():
        for vname in ["US_5min", "US_15min", "CombinedUS_5and15"]:
            if vname not in pdata:
                continue
            stats = pdata[vname]["stats"]
            tag   = f"{pname}/{vname}"
            if stats:
                pf_s = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 1e6 else "INF"
                marker = "  *** COMBINED WALLET" if "Combined" in vname else ""
                print(f"  {tag:<26}: {stats['trades']}T  WR {stats['win_rate']:.1f}%  "
                      f"PF {pf_s}  MaxDD {stats['max_drawdown']:.1f}%  Ret {stats['total_return']:+.1f}%{marker}")
            else:
                print(f"  {tag:<26}: NO TRADES")
    print()
    bar()
    print("  S11 RUN COMPLETE")
    bar()


if __name__ == "__main__":
    main()
