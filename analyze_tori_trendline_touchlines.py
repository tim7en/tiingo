#!/usr/bin/env python3
"""Systematize Tori Trade's playbook with straight touchpoint trendlines.

This version follows the playbook more closely than the regression-envelope
proxy. It constructs static straight trendlines from swing highs/lows, looks for
2- or 3-touch bounce and break setups, and uses those same straight lines for
entries, invalidation, and chart overlays.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results" / "tori_trendline_touchlines"
NY_TZ = ZoneInfo("America/New_York")

ASSET_PATHS = {
    "XAU": CACHE_DIR / "XAU_5m.parquet",
    "XAG": CACHE_DIR / "XAG_5m.parquet",
    "COPPER": CACHE_DIR / "COPPER_5m.parquet",
    "UUP": CACHE_DIR / "UUP_5m.parquet",
}


@dataclass(frozen=True)
class SweepConfig:
    asset: str
    bar_minutes: int
    recent_pivots: int
    min_touches: int
    slope_threshold: float
    stop_buffer: float
    take_profit_r: float
    touch_tolerance: float
    break_buffer: float
    max_break_risk_pct: float
    use_take_profit: bool


def parse_csv_list(raw: str, cast):
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default="XAU,UUP")
    parser.add_argument("--bar-minutes", type=int, default=240)
    parser.add_argument(
        "--recent-pivots",
        default="8,12",
        help="How many recent pivot highs/lows to consider when fitting trendlines.",
    )
    parser.add_argument("--touch-options", default="2,3")
    parser.add_argument("--slope-thresholds", default="0.0002,0.0004,0.0006")
    parser.add_argument("--stop-buffers", default="0.0,0.25,0.5")
    parser.add_argument("--take-profits", default="1.5,2.0,3.0,4.0")
    parser.add_argument(
        "--use-take-profit",
        action="store_true",
        help="Enable fixed R-multiple take profits. Default behavior is trail-only on the safety line.",
    )
    parser.add_argument("--touch-tolerances", default="0.25,0.35")
    parser.add_argument("--break-buffer", type=float, default=0.15)
    parser.add_argument("--max-break-risk-pct", type=float, default=0.035)
    parser.add_argument(
        "--line-violation-buffer",
        type=float,
        default=0.75,
        help="How far a pivot may violate a line, in ATR units, before that line is rejected.",
    )
    parser.add_argument("--max-candidates-per-bar", type=int, default=8)
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--trade-chart-limit", type=int, default=0)
    parser.add_argument("--skip-trade-charts", action="store_true")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


def load_asset_5m(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["o", "h", "l", "c", "v", "time"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df.sort_values("time").reset_index(drop=True)


def resample_session_bars(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    if bar_minutes % 5 != 0:
        raise ValueError("--bar-minutes must be divisible by 5")

    bars_per_bucket = bar_minutes // 5
    out = df.copy()
    out["time_et"] = out["time"].dt.tz_convert(NY_TZ)
    out["date_et"] = out["time_et"].dt.date
    out["bar_in_day"] = out.groupby("date_et").cumcount()
    out["bucket"] = out["bar_in_day"] // bars_per_bucket

    grouped = (
        out.groupby(["date_et", "bucket"], as_index=False)
        .agg(
            time=("time", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            raw_bars=("time", "size"),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    grouped["time_et"] = grouped["time"].dt.tz_convert(NY_TZ)
    grouped["date_et"] = grouped["time_et"].dt.date
    grouped["weekday"] = grouped["time_et"].dt.day_name()
    prev_close = grouped["close"].shift(1)
    tr = pd.concat(
        [
            grouped["high"] - grouped["low"],
            (grouped["high"] - prev_close).abs(),
            (grouped["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    grouped["tr"] = tr
    grouped["atr"] = grouped["tr"].rolling(5, min_periods=1).mean()
    grouped["atr_safe"] = (
        grouped["atr"]
        .replace(0.0, np.nan)
        .ffill()
        .fillna(grouped["close"] * 0.0025)
        .clip(lower=1e-6)
    )
    return grouped


def local_extrema_flags(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    swing_high = np.zeros(len(high), dtype=bool)
    swing_low = np.zeros(len(low), dtype=bool)
    if len(high) < 3:
        return swing_high, swing_low

    swing_high[1:-1] = (high[1:-1] >= high[:-2]) & (high[1:-1] > high[2:])
    swing_low[1:-1] = (low[1:-1] <= low[:-2]) & (low[1:-1] < low[2:])
    return swing_high, swing_low


def line_value(anchor_idx: int, anchor_price: float, slope: float, idx: int | np.ndarray) -> np.ndarray:
    return anchor_price + slope * (np.asarray(idx) - anchor_idx)


def make_candidate(
    kind: str,
    current_idx: int,
    anchor1_idx: int,
    anchor2_idx: int,
    anchor1_price: float,
    slope: float,
    pivot_idx: np.ndarray,
    pivot_price: np.ndarray,
    atr: np.ndarray,
    max_touch_tolerance: float,
    violation_buffer: float,
) -> dict | None:
    if anchor2_idx <= anchor1_idx:
        return None

    relevant_mask = (pivot_idx >= anchor1_idx) & (pivot_idx < current_idx)
    test_idx = pivot_idx[relevant_mask]
    test_price = pivot_price[relevant_mask]
    if len(test_idx) < 2:
        return None

    line_at_touches = line_value(anchor1_idx, anchor1_price, slope, test_idx)
    atr_at_touches = atr[test_idx]
    violations = (test_price - line_at_touches) / atr_at_touches

    if kind == "support":
        if violations.min(initial=0.0) < -(max_touch_tolerance + violation_buffer):
            return None
    else:
        if violations.max(initial=0.0) > (max_touch_tolerance + violation_buffer):
            return None

    touch_error = np.abs(test_price - line_at_touches) / atr_at_touches
    touch_mask = touch_error <= max_touch_tolerance
    if touch_mask.sum() < 2:
        return None

    touch_idx = test_idx[touch_mask]
    line_now = float(line_value(anchor1_idx, anchor1_price, slope, current_idx))
    if not np.isfinite(line_now) or line_now <= 0:
        return None

    return {
        "kind": kind,
        "anchor1_idx": int(anchor1_idx),
        "anchor2_idx": int(anchor2_idx),
        "anchor1_price": float(anchor1_price),
        "anchor2_price": float(line_value(anchor1_idx, anchor1_price, slope, anchor2_idx)),
        "slope": float(slope),
        "slope_pct": float(slope / line_now),
        "line_now": line_now,
        "touch_count": int(touch_mask.sum()),
        "first_touch_idx": int(touch_idx[0]),
        "last_touch_idx": int(touch_idx[-1]),
        "age_bars": int(current_idx - touch_idx[0]),
        "span_bars": int(anchor2_idx - anchor1_idx),
        "max_touch_error": float(touch_error[touch_mask].max(initial=0.0)),
    }


def build_candidates_for_bar(
    current_idx: int,
    recent_idx: np.ndarray,
    price_array: np.ndarray,
    pivot_idx: np.ndarray,
    pivot_price: np.ndarray,
    atr: np.ndarray,
    kind: str,
    max_touch_tolerance: float,
    violation_buffer: float,
    max_candidates: int,
) -> list[dict]:
    if len(recent_idx) < 2:
        return []

    candidates = []
    for pos_a in range(len(recent_idx) - 1):
        a_idx = int(recent_idx[pos_a])
        a_price = float(price_array[a_idx])
        for pos_b in range(pos_a + 1, len(recent_idx)):
            b_idx = int(recent_idx[pos_b])
            b_price = float(price_array[b_idx])
            slope = (b_price - a_price) / (b_idx - a_idx)
            candidate = make_candidate(
                kind=kind,
                current_idx=current_idx,
                anchor1_idx=a_idx,
                anchor2_idx=b_idx,
                anchor1_price=a_price,
                slope=slope,
                pivot_idx=pivot_idx,
                pivot_price=pivot_price,
                atr=atr,
                max_touch_tolerance=max_touch_tolerance,
                violation_buffer=violation_buffer,
            )
            if candidate is not None:
                candidates.append(candidate)

    candidates.sort(
        key=lambda c: (
            c["touch_count"],
            c["age_bars"],
            c["span_bars"],
            -c["max_touch_error"],
            abs(c["slope_pct"]),
        ),
        reverse=True,
    )
    return candidates[:max_candidates]


def precompute_touchline_model(
    df: pd.DataFrame,
    recent_pivots: int,
    max_touch_tolerance: float,
    violation_buffer: float,
    max_candidates_per_bar: int,
) -> dict:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr_safe"].to_numpy(dtype=float)
    swing_high, swing_low = local_extrema_flags(high, low)
    pivot_low_idx = np.flatnonzero(swing_low)
    pivot_high_idx = np.flatnonzero(swing_high)

    support_candidates: list[list[dict]] = [[] for _ in range(len(df))]
    resistance_candidates: list[list[dict]] = [[] for _ in range(len(df))]

    for i in range(len(df)):
        recent_lows = pivot_low_idx[pivot_low_idx < i][-recent_pivots:]
        recent_highs = pivot_high_idx[pivot_high_idx < i][-recent_pivots:]
        support_candidates[i] = build_candidates_for_bar(
            current_idx=i,
            recent_idx=recent_lows,
            price_array=low,
            pivot_idx=pivot_low_idx,
            pivot_price=low[pivot_low_idx],
            atr=atr,
            kind="support",
            max_touch_tolerance=max_touch_tolerance,
            violation_buffer=violation_buffer,
            max_candidates=max_candidates_per_bar,
        )
        resistance_candidates[i] = build_candidates_for_bar(
            current_idx=i,
            recent_idx=recent_highs,
            price_array=high,
            pivot_idx=pivot_high_idx,
            pivot_price=high[pivot_high_idx],
            atr=atr,
            kind="resistance",
            max_touch_tolerance=max_touch_tolerance,
            violation_buffer=violation_buffer,
            max_candidates=max_candidates_per_bar,
        )

    bars_per_day = max(1.0, float(df.groupby("date_et").size().median()))
    min_week_bars = max(2, int(round(bars_per_day * 5.0)))
    return {
        "support_candidates": support_candidates,
        "resistance_candidates": resistance_candidates,
        "bars_per_day": bars_per_day,
        "min_week_bars": min_week_bars,
        "atr": atr,
        "close": close,
        "open": df["open"].to_numpy(dtype=float),
        "high": high,
        "low": low,
        "time": df["time"].to_numpy(),
        "time_et": df["time_et"].to_numpy(),
        "weekday": df["weekday"].to_numpy(),
        "date_et": pd.Series(df["date_et"]).astype(str).to_numpy(),
    }


def qualify_bounce_candidate(
    cand: dict,
    side: int,
    i: int,
    arrays: dict,
    cfg: SweepConfig,
    min_week_bars: int,
) -> tuple[bool, float]:
    atr = arrays["atr"]
    low = arrays["low"]
    high = arrays["high"]
    close = arrays["close"]
    open_ = arrays["open"]
    line_now = cand["line_now"]
    if cand["touch_count"] < cfg.min_touches or cand["age_bars"] < min_week_bars:
        return False, float("inf")

    if side == 1:
        if cand["slope_pct"] < cfg.slope_threshold:
            return False, float("inf")
        if low[i] > line_now + cfg.touch_tolerance * atr[i]:
            return False, float("inf")
        if close[i] < line_now - 0.10 * atr[i]:
            return False, float("inf")
        if close[i] < open_[i]:
            return False, float("inf")
        distance = abs(low[i] - line_now) / atr[i]
        return True, float(distance)

    if cand["slope_pct"] > -cfg.slope_threshold:
        return False, float("inf")
    if high[i] < line_now - cfg.touch_tolerance * atr[i]:
        return False, float("inf")
    if close[i] > line_now + 0.10 * atr[i]:
        return False, float("inf")
    if close[i] > open_[i]:
        return False, float("inf")
    distance = abs(high[i] - line_now) / atr[i]
    return True, float(distance)


def pick_best_support_for_long_break(support_candidates: list[dict], price: float, slope_threshold: float) -> dict | None:
    eligible = [
        cand
        for cand in support_candidates
        if cand["touch_count"] >= 2 and cand["line_now"] < price and cand["slope_pct"] >= -(slope_threshold / 2.0)
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda c: (c["touch_count"], c["line_now"], c["age_bars"]), reverse=True)
    return eligible[0]


def pick_best_resistance_for_short_break(resistance_candidates: list[dict], price: float, slope_threshold: float) -> dict | None:
    eligible = [
        cand
        for cand in resistance_candidates
        if cand["touch_count"] >= 2 and cand["line_now"] > price and cand["slope_pct"] <= (slope_threshold / 2.0)
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda c: (c["touch_count"], c["line_now"], c["age_bars"]), reverse=True)
    return eligible[0]


def build_signals_for_bar(i: int, model: dict, cfg: SweepConfig) -> list[dict]:
    support_candidates = model["support_candidates"][i]
    resistance_candidates = model["resistance_candidates"][i]
    atr = model["atr"]
    close = model["close"]
    min_week_bars = int(model["min_week_bars"])

    signals = []

    for cand in support_candidates:
        ok, distance = qualify_bounce_candidate(cand, side=1, i=i, arrays=model, cfg=cfg, min_week_bars=min_week_bars)
        if ok:
            signals.append(
                {
                    "setup": "bounce",
                    "side": 1,
                    "action": cand,
                    "safety": cand,
                    "touch_count": cand["touch_count"],
                    "current_distance": distance,
                    "priority": 3,
                }
            )

    for cand in resistance_candidates:
        ok, distance = qualify_bounce_candidate(cand, side=-1, i=i, arrays=model, cfg=cfg, min_week_bars=min_week_bars)
        if ok:
            signals.append(
                {
                    "setup": "bounce",
                    "side": -1,
                    "action": cand,
                    "safety": cand,
                    "touch_count": cand["touch_count"],
                    "current_distance": distance,
                    "priority": 3,
                }
            )

    for action in resistance_candidates:
        if action["touch_count"] < cfg.min_touches or action["age_bars"] < min_week_bars:
            continue
        if action["slope_pct"] > -cfg.slope_threshold:
            continue
        if close[i] <= action["line_now"] + cfg.break_buffer * atr[i]:
            continue
        safety = pick_best_support_for_long_break(support_candidates, float(close[i]), cfg.slope_threshold)
        if safety is None:
            continue
        initial_stop = safety["line_now"] - cfg.stop_buffer * atr[i]
        initial_risk = close[i] - initial_stop
        if initial_risk <= 0:
            continue
        risk_pct = initial_risk / close[i]
        if risk_pct > cfg.max_break_risk_pct:
            continue
        signals.append(
            {
                "setup": "break",
                "side": 1,
                "action": action,
                "safety": safety,
                "touch_count": action["touch_count"],
                "current_distance": risk_pct,
                "priority": 2,
            }
        )

    for action in support_candidates:
        if action["touch_count"] < cfg.min_touches or action["age_bars"] < min_week_bars:
            continue
        if action["slope_pct"] < cfg.slope_threshold:
            continue
        if close[i] >= action["line_now"] - cfg.break_buffer * atr[i]:
            continue
        safety = pick_best_resistance_for_short_break(resistance_candidates, float(close[i]), cfg.slope_threshold)
        if safety is None:
            continue
        initial_stop = safety["line_now"] + cfg.stop_buffer * atr[i]
        initial_risk = initial_stop - close[i]
        if initial_risk <= 0:
            continue
        risk_pct = initial_risk / close[i]
        if risk_pct > cfg.max_break_risk_pct:
            continue
        signals.append(
            {
                "setup": "break",
                "side": -1,
                "action": action,
                "safety": safety,
                "touch_count": action["touch_count"],
                "current_distance": risk_pct,
                "priority": 2,
            }
        )

    signals.sort(
        key=lambda s: (
            s["priority"],
            s["touch_count"],
            -s["current_distance"],
            s["action"]["age_bars"],
        ),
        reverse=True,
    )
    return signals


def line_from_trade(trade: pd.Series, prefix: str, idx: int | np.ndarray) -> np.ndarray:
    return line_value(
        anchor_idx=int(trade[f"{prefix}_anchor1_idx"]),
        anchor_price=float(trade[f"{prefix}_anchor1_price"]),
        slope=float(trade[f"{prefix}_slope"]),
        idx=idx,
    )


def simulate_config(df: pd.DataFrame, model: dict, cfg: SweepConfig) -> pd.DataFrame:
    close = model["close"]
    high = model["high"]
    low = model["low"]
    atr = model["atr"]
    time = model["time"]
    time_et = model["time_et"]
    weekday = model["weekday"]
    date_et = model["date_et"]
    bars_per_day = float(model["bars_per_day"])

    trades = []
    warmup = int(model["min_week_bars"])
    i = warmup
    while i < len(df) - 1:
        signals = build_signals_for_bar(i, model, cfg)
        if not signals:
            i += 1
            continue

        signal = signals[0]
        action = signal["action"]
        safety = signal["safety"]
        side = int(signal["side"])
        entry_idx = i
        entry_price = float(close[entry_idx])

        if side == 1:
            initial_stop = float(line_value(safety["anchor1_idx"], safety["anchor1_price"], safety["slope"], entry_idx) - cfg.stop_buffer * atr[entry_idx])
            initial_risk = entry_price - initial_stop
            take_profit = entry_price + cfg.take_profit_r * initial_risk if cfg.use_take_profit and cfg.take_profit_r > 0 else np.nan
        else:
            initial_stop = float(line_value(safety["anchor1_idx"], safety["anchor1_price"], safety["slope"], entry_idx) + cfg.stop_buffer * atr[entry_idx])
            initial_risk = initial_stop - entry_price
            take_profit = entry_price - cfg.take_profit_r * initial_risk if cfg.use_take_profit and cfg.take_profit_r > 0 else np.nan

        if initial_risk <= 0 or not np.isfinite(initial_risk):
            i += 1
            continue

        exit_idx = None
        exit_reason = "end_of_data"
        exit_price = float(close[-1])
        peak_favorable = 0.0
        peak_adverse = 0.0

        j = entry_idx + 1
        while j < len(df):
            safety_line_now = float(line_value(safety["anchor1_idx"], safety["anchor1_price"], safety["slope"], j))
            if side == 1:
                stop_level = safety_line_now - cfg.stop_buffer * atr[j]
                move_favorable = (high[j] - entry_price) / initial_risk
                move_adverse = (low[j] - entry_price) / initial_risk
                peak_favorable = max(peak_favorable, move_favorable)
                peak_adverse = min(peak_adverse, move_adverse)
                stop_hit = close[j] <= stop_level
                target_hit = np.isfinite(take_profit) and high[j] >= take_profit
            else:
                stop_level = safety_line_now + cfg.stop_buffer * atr[j]
                move_favorable = (entry_price - low[j]) / initial_risk
                move_adverse = (entry_price - high[j]) / initial_risk
                peak_favorable = max(peak_favorable, move_favorable)
                peak_adverse = min(peak_adverse, move_adverse)
                stop_hit = close[j] >= stop_level
                target_hit = np.isfinite(take_profit) and low[j] <= take_profit

            if stop_hit and target_hit:
                exit_price = float(close[j])
                exit_reason = "safety_break"
                exit_idx = j
                break
            if stop_hit:
                exit_price = float(close[j])
                exit_reason = "safety_break"
                exit_idx = j
                break
            if target_hit:
                exit_price = float(take_profit)
                exit_reason = "take_profit"
                exit_idx = j
                break
            j += 1

        if exit_idx is None:
            exit_idx = len(df) - 1

        return_pct = side * (exit_price / entry_price - 1.0)
        r_multiple = side * (exit_price - entry_price) / initial_risk

        trades.append(
            {
                **asdict(cfg),
                "entry_time_utc": pd.Timestamp(time[entry_idx]),
                "exit_time_utc": pd.Timestamp(time[exit_idx]),
                "entry_time_et": pd.Timestamp(time_et[entry_idx]),
                "exit_time_et": pd.Timestamp(time_et[exit_idx]),
                "entry_date_et": date_et[entry_idx],
                "exit_date_et": date_et[exit_idx],
                "entry_weekday": weekday[entry_idx],
                "setup": signal["setup"],
                "side": "long" if side == 1 else "short",
                "touch_count": int(action["touch_count"]),
                "touch_age_bars": int(action["age_bars"]),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "initial_stop": initial_stop,
                "take_profit_price": take_profit,
                "entry_bar_idx": int(entry_idx),
                "exit_bar_idx": int(exit_idx),
                "initial_risk": initial_risk,
                "risk_pct_of_entry": initial_risk / entry_price,
                "return_pct": return_pct,
                "r_multiple": r_multiple,
                "exit_reason": exit_reason,
                "hold_bars": int(exit_idx - entry_idx),
                "hold_days_est": float((exit_idx - entry_idx) / max(bars_per_day, 1.0)),
                "action_kind": action["kind"],
                "action_anchor1_idx": int(action["anchor1_idx"]),
                "action_anchor2_idx": int(action["anchor2_idx"]),
                "action_anchor1_price": float(action["anchor1_price"]),
                "action_anchor2_price": float(action["anchor2_price"]),
                "action_slope": float(action["slope"]),
                "action_slope_pct": float(action["slope_pct"]),
                "action_line_entry": float(action["line_now"]),
                "action_touch_count": int(action["touch_count"]),
                "safety_kind": safety["kind"],
                "safety_anchor1_idx": int(safety["anchor1_idx"]),
                "safety_anchor2_idx": int(safety["anchor2_idx"]),
                "safety_anchor1_price": float(safety["anchor1_price"]),
                "safety_anchor2_price": float(safety["anchor2_price"]),
                "safety_slope": float(safety["slope"]),
                "safety_slope_pct": float(safety["slope_pct"]),
                "safety_line_entry": float(line_value(safety["anchor1_idx"], safety["anchor1_price"], safety["slope"], entry_idx)),
                "safety_touch_count": int(safety["touch_count"]),
                "mfe_r": float(peak_favorable),
                "mae_r": float(peak_adverse),
            }
        )
        i = exit_idx + 1

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame, buy_hold_return: float, min_trades: int) -> dict:
    if trades.empty:
        return {
            "trade_count": 0,
            "long_count": 0,
            "short_count": 0,
            "bounce_count": 0,
            "break_count": 0,
            "win_rate": np.nan,
            "avg_return_pct": np.nan,
            "median_return_pct": np.nan,
            "avg_r_multiple": np.nan,
            "profit_factor": np.nan,
            "total_return": np.nan,
            "cagr": np.nan,
            "max_drawdown": np.nan,
            "trades_per_year": np.nan,
            "avg_hold_days": np.nan,
            "tp_rate": np.nan,
            "buy_hold_return": buy_hold_return,
            "outperformance_vs_hold": np.nan,
            "quality_score": -np.inf,
        }

    trades = trades.sort_values("exit_time_utc").reset_index(drop=True)
    equity = (1.0 + trades["return_pct"]).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    running_max = equity.cummax()
    max_drawdown = float((equity / running_max - 1.0).min())
    gross_profit = float(trades.loc[trades["return_pct"] > 0, "return_pct"].sum())
    gross_loss = float(trades.loc[trades["return_pct"] < 0, "return_pct"].sum())
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
    years = max(
        (pd.Timestamp(trades["exit_time_utc"].iloc[-1]) - pd.Timestamp(trades["entry_time_utc"].iloc[0])).days / 365.25,
        1.0 / 365.25,
    )
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    outperformance = total_return - buy_hold_return
    trade_count = int(len(trades))
    quality_score = (
        total_return
        + 0.25 * np.tanh(profit_factor if np.isfinite(profit_factor) else 5.0)
        + 0.10 * float(trades["r_multiple"].mean())
        - 0.60 * abs(max_drawdown)
    )
    if trade_count < min_trades:
        quality_score -= 1.0

    return {
        "trade_count": trade_count,
        "long_count": int((trades["side"] == "long").sum()),
        "short_count": int((trades["side"] == "short").sum()),
        "bounce_count": int((trades["setup"] == "bounce").sum()),
        "break_count": int((trades["setup"] == "break").sum()),
        "win_rate": float((trades["return_pct"] > 0).mean()),
        "avg_return_pct": float(trades["return_pct"].mean()),
        "median_return_pct": float(trades["return_pct"].median()),
        "avg_r_multiple": float(trades["r_multiple"].mean()),
        "profit_factor": float(profit_factor),
        "total_return": total_return,
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
        "trades_per_year": float(trade_count / years),
        "avg_hold_days": float(trades["hold_days_est"].mean()),
        "tp_rate": float((trades["exit_reason"] == "take_profit").mean()),
        "buy_hold_return": float(buy_hold_return),
        "outperformance_vs_hold": float(outperformance),
        "quality_score": float(quality_score),
    }


def add_trend_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    abs_slope = out["action_slope_pct"].abs()
    if abs_slope.nunique() < 3:
        out["trend_bucket"] = "mixed"
        return out
    out["trend_bucket"] = pd.qcut(abs_slope, q=3, labels=["soft", "steady", "strong"], duplicates="drop")
    return out


def build_position_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    enriched = add_trend_buckets(trades)
    grouped = (
        enriched.groupby(["setup", "side", "trend_bucket"], dropna=False, observed=True)
        .agg(
            trades=("return_pct", "size"),
            win_rate=("return_pct", lambda s: float((s > 0).mean())),
            avg_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            avg_r_multiple=("r_multiple", "mean"),
            total_return=("return_pct", lambda s: float((1.0 + s).prod() - 1.0)),
            avg_touch_count=("touch_count", "mean"),
            avg_hold_days=("hold_days_est", "mean"),
        )
        .reset_index()
        .sort_values(["total_return", "win_rate", "trades"], ascending=[False, False, False])
    )
    return grouped


def build_yearly_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    yearly = trades.copy()
    yearly["entry_year"] = pd.to_datetime(yearly["entry_time_et"]).dt.year
    grouped = (
        yearly.groupby(["entry_year", "setup", "side"])
        .agg(
            trades=("return_pct", "size"),
            win_rate=("return_pct", lambda s: float((s > 0).mean())),
            avg_return_pct=("return_pct", "mean"),
            total_return=("return_pct", lambda s: float((1.0 + s).prod() - 1.0)),
        )
        .reset_index()
        .sort_values(["entry_year", "total_return"], ascending=[True, False])
    )
    return grouped


def draw_candlesticks(ax: plt.Axes, frame: pd.DataFrame):
    x = frame["x"].to_numpy(dtype=float)
    width = 0.68
    for xi, open_, high, low, close in zip(x, frame["open"], frame["high"], frame["low"], frame["close"]):
        color = "#16a34a" if close >= open_ else "#dc2626"
        ax.vlines(xi, low, high, color=color, linewidth=1.1, alpha=0.9, zorder=2)
        body_low = min(open_, close)
        body_height = abs(close - open_)
        if body_height < 1e-9:
            ax.hlines(close, xi - width / 2.0, xi + width / 2.0, color=color, linewidth=1.8, zorder=3)
        else:
            ax.add_patch(
                Rectangle(
                    (xi - width / 2.0, body_low),
                    width,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=3,
                )
            )


def plot_trade_overview(asset: str, df: pd.DataFrame, trades: pd.DataFrame, out_path: Path):
    if trades.empty:
        return

    view = df[["time_et", "close"]].copy()
    view["time_plot"] = pd.to_datetime(view["time_et"]).dt.tz_localize(None)
    trades = trades.copy()
    trades["entry_plot"] = pd.to_datetime(trades["entry_time_et"]).dt.tz_localize(None)
    trades["exit_plot"] = pd.to_datetime(trades["exit_time_et"]).dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(view["time_plot"], view["close"], color="#475569", linewidth=1.05, alpha=0.95, label=f"{asset} close")

    longs = trades[trades["side"] == "long"]
    shorts = trades[trades["side"] == "short"]
    ax.scatter(longs["entry_plot"], longs["entry_price"], marker="^", s=55, color="#2563eb", label="Long entry", zorder=5)
    ax.scatter(shorts["entry_plot"], shorts["entry_price"], marker="v", s=55, color="#7c3aed", label="Short entry", zorder=5)
    ax.scatter(trades["exit_plot"], trades["exit_price"], marker="X", s=45, color="#111827", label="Exit", zorder=5)

    ax.set_title(f"{asset} straight-trendline trade overview")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", ncol=4, frameon=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_trade_detail(asset: str, df: pd.DataFrame, trade: pd.Series, out_path: Path):
    entry_idx = int(trade["entry_bar_idx"])
    exit_idx = int(trade["exit_bar_idx"])
    context = 12
    post_bars = 6
    start_idx = max(0, min(int(trade["action_anchor1_idx"]), int(trade["safety_anchor1_idx"]), entry_idx) - context)
    end_idx = min(len(df) - 1, exit_idx + post_bars)

    frame = df.iloc[start_idx : end_idx + 1][["time_et", "open", "high", "low", "close"]].copy()
    frame["bar_idx"] = np.arange(start_idx, end_idx + 1)
    frame["time_plot"] = pd.to_datetime(frame["time_et"]).dt.tz_localize(None)
    frame["x"] = np.arange(len(frame), dtype=float)
    frame["action_line"] = line_from_trade(trade, "action", frame["bar_idx"].to_numpy())
    frame["safety_line"] = line_from_trade(trade, "safety", frame["bar_idx"].to_numpy())
    frame["take_profit"] = float(trade["take_profit_price"])

    fig, ax = plt.subplots(figsize=(15, 7))
    draw_candlesticks(ax, frame)
    ax.plot(frame["x"], frame["action_line"], color="#dc2626", linewidth=2.0, label="Action line", zorder=4)
    ax.plot(frame["x"], frame["safety_line"], color="#16a34a", linewidth=2.0, label="Safety line", zorder=4)
    if np.isfinite(float(trade["take_profit_price"])):
        ax.plot(frame["x"], frame["take_profit"], color="#2563eb", linewidth=1.4, linestyle=":", label="Take profit", zorder=3)

    entry_x = float(entry_idx - start_idx)
    exit_x = float(exit_idx - start_idx)
    is_long = str(trade["side"]) == "long"
    entry_marker = "^" if is_long else "v"

    ax.scatter([entry_x], [float(trade["entry_price"])], marker=entry_marker, s=120, color="#1d4ed8", edgecolor="white", linewidth=0.8, zorder=6, label="Entry")
    ax.scatter([exit_x], [float(trade["exit_price"])], marker="X", s=110, color="#111827", edgecolor="white", linewidth=0.8, zorder=6, label="Exit")

    for prefix, color, marker in [("action", "#dc2626", "o"), ("safety", "#16a34a", "s")]:
        for anchor_num in [1, 2]:
            anchor_idx = int(trade[f"{prefix}_anchor{anchor_num}_idx"])
            if start_idx <= anchor_idx <= end_idx:
                anchor_x = float(anchor_idx - start_idx)
                anchor_price = float(trade[f"{prefix}_anchor{anchor_num}_price"])
                ax.scatter([anchor_x], [anchor_price], marker=marker, s=40, color=color, edgecolor="white", linewidth=0.6, zorder=6)

    ax.axvspan(entry_x, exit_x, color="#93c5fd", alpha=0.10, zorder=1)

    ax.annotate("Entry", (entry_x, float(trade["entry_price"])), xytext=(0, 12 if is_long else -20), textcoords="offset points", ha="center", color="#1d4ed8", fontsize=9, fontweight="bold")
    ax.annotate(f"Exit: {trade['exit_reason']}", (exit_x, float(trade["exit_price"])), xytext=(0, -18), textcoords="offset points", ha="center", color="#111827", fontsize=9, fontweight="bold")

    details = (
        f"{asset} | trade {int(trade['trade_id'])} | {trade['setup']} {trade['side']}\n"
        f"entry {float(trade['entry_price']):.3f} | exit {float(trade['exit_price']):.3f} | "
        f"ret {float(trade['return_pct']) * 100:.2f}% | R {float(trade['r_multiple']):.2f} | "
        f"touches {int(trade['touch_count'])}"
    )
    ax.text(
        0.99,
        0.98,
        details,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "boxstyle": "round,pad=0.4"},
    )

    ax.set_title(f"{asset} trade detail")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", ncol=3, frameon=False)
    tick_count = min(7, len(frame))
    tick_idx = np.linspace(0, len(frame) - 1, num=tick_count, dtype=int)
    tick_labels = [frame.iloc[idx]["time_plot"].strftime("%Y-%m-%d") for idx in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_trade_charts(asset: str, df: pd.DataFrame, trades: pd.DataFrame, out_dir: Path, chart_limit: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_trade_overview(asset, df, trades, out_dir / f"{asset.lower()}_trade_overview.png")
    selected = trades.head(chart_limit) if chart_limit > 0 else trades
    for _, trade in selected.iterrows():
        label = f"trade_{int(trade['trade_id']):03d}_{trade['setup']}_{trade['side']}_{pd.to_datetime(trade['entry_time_et']).strftime('%Y%m%d_%H%M')}.png"
        plot_trade_detail(asset, df, trade, out_dir / label)


def iter_configs(args, assets: Iterable[str]) -> list[SweepConfig]:
    recent_pivots = parse_csv_list(args.recent_pivots, int)
    touch_options = parse_csv_list(args.touch_options, int)
    slope_thresholds = parse_csv_list(args.slope_thresholds, float)
    stop_buffers = parse_csv_list(args.stop_buffers, float)
    take_profits = parse_csv_list(args.take_profits, float) if args.use_take_profit else [0.0]
    touch_tolerances = parse_csv_list(args.touch_tolerances, float)

    configs = []
    for asset, recent, min_touches, slope_th, stop_buffer, take_profit, touch_tol in itertools.product(
        assets,
        recent_pivots,
        touch_options,
        slope_thresholds,
        stop_buffers,
        take_profits,
        touch_tolerances,
    ):
        configs.append(
            SweepConfig(
                asset=asset,
                bar_minutes=args.bar_minutes,
                recent_pivots=recent,
                min_touches=min_touches,
                slope_threshold=slope_th,
                stop_buffer=stop_buffer,
                take_profit_r=take_profit,
                touch_tolerance=touch_tol,
                break_buffer=args.break_buffer,
                max_break_risk_pct=args.max_break_risk_pct,
                use_take_profit=bool(args.use_take_profit),
            )
        )
    return configs


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = [asset.strip().upper() for asset in args.assets.split(",") if asset.strip()]
    unknown = [asset for asset in assets if asset not in ASSET_PATHS]
    if unknown:
        raise ValueError(f"Unsupported assets: {unknown}. Supported: {sorted(ASSET_PATHS)}")

    max_touch_tolerance = max(parse_csv_list(args.touch_tolerances, float))
    recent_pivots_values = sorted(set(parse_csv_list(args.recent_pivots, int)))

    prepared_assets: dict[str, pd.DataFrame] = {}
    models: dict[tuple[str, int], dict] = {}
    baseline_rows = []

    for asset in assets:
        df = resample_session_bars(load_asset_5m(ASSET_PATHS[asset]), args.bar_minutes)
        prepared_assets[asset] = df
        baseline_rows.append(
            {
                "asset": asset,
                "bar_minutes": args.bar_minutes,
                "bars": len(df),
                "start_utc": df["time"].iloc[0],
                "end_utc": df["time"].iloc[-1],
                "buy_hold_return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0),
            }
        )
        for recent in recent_pivots_values:
            models[(asset, recent)] = precompute_touchline_model(
                df=df,
                recent_pivots=recent,
                max_touch_tolerance=max_touch_tolerance,
                violation_buffer=args.line_violation_buffer,
                max_candidates_per_bar=args.max_candidates_per_bar,
            )

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(out_dir / "asset_baselines.csv", index=False)

    summary_rows = []
    best_trade_logs: dict[str, pd.DataFrame] = {}
    best_cfgs: dict[str, SweepConfig] = {}
    best_scores: dict[str, float] = {asset: -np.inf for asset in assets}

    configs = iter_configs(args, assets)
    for idx, cfg in enumerate(configs, start=1):
        model = models[(cfg.asset, cfg.recent_pivots)]
        trades = simulate_config(prepared_assets[cfg.asset], model, cfg)
        buy_hold_return = float(
            baseline_df.loc[baseline_df["asset"] == cfg.asset, "buy_hold_return"].iloc[0]
        )
        summary = summarize_trades(trades, buy_hold_return, args.min_trades)
        row = {**asdict(cfg), **summary}
        summary_rows.append(row)

        if summary["quality_score"] > best_scores[cfg.asset]:
            best_scores[cfg.asset] = summary["quality_score"]
            best_trade_logs[cfg.asset] = trades
            best_cfgs[cfg.asset] = cfg

        if idx % 50 == 0:
            print(f"[{idx}/{len(configs)}] finished {cfg.asset} pivots={cfg.recent_pivots} touches={cfg.min_touches}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["quality_score", "total_return", "trade_count", "profit_factor"],
        ascending=[False, False, False, False],
    )
    summary_df.to_csv(out_dir / "sweep_summary.csv", index=False)

    best_rows = []
    for asset in assets:
        asset_summary = summary_df[summary_df["asset"] == asset].copy()
        if asset_summary.empty:
            continue
        best = asset_summary.head(args.top_k)
        best.to_csv(out_dir / f"{asset.lower()}_top_configs.csv", index=False)
        best_rows.append(best.iloc[0].to_dict())

        trades = best_trade_logs.get(asset, pd.DataFrame())
        if trades.empty:
            continue
        trades = trades.sort_values("entry_time_utc").reset_index(drop=True)
        trades.insert(0, "trade_id", np.arange(1, len(trades) + 1))
        trades.to_csv(out_dir / f"{asset.lower()}_best_trade_log.csv", index=False)
        build_position_stats(trades).to_csv(out_dir / f"{asset.lower()}_best_position_stats.csv", index=False)
        build_yearly_stats(trades).to_csv(out_dir / f"{asset.lower()}_best_yearly_stats.csv", index=False)
        if not args.skip_trade_charts:
            save_trade_charts(
                asset=asset,
                df=prepared_assets[asset],
                trades=trades,
                out_dir=out_dir / f"{asset.lower()}_trade_charts",
                chart_limit=args.trade_chart_limit,
            )

    if best_rows:
        pd.DataFrame(best_rows).sort_values(
            ["quality_score", "total_return"], ascending=[False, False]
        ).to_csv(out_dir / "best_configs.csv", index=False)

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
