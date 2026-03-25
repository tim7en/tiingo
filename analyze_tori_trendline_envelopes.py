#!/usr/bin/env python3
"""Systematize Tori Trade's trendline playbook with regression envelopes.

The PDF playbook is discretionary and indicator-free: entries come from
manually drawn trendlines, while the "action line" and "safety line" guide
entries, invalidation, and trailing stops. To backtest it on our local XAU/UUP
cache, this script uses rolling linear-regression envelopes as a mechanical
proxy for those lines:

- Trendline bounce:
  In an uptrend, the lower envelope acts as the action/safety line for longs.
  In a downtrend, the upper envelope acts as the action/safety line for shorts.
- Trendline break:
  A close through the envelope is the action-line break.
  The opposite envelope becomes the safety line.

The sweep ranks envelope widths, touchpoint requirements, stop buffers, and
take-profit multiples for each asset separately.
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
RESULTS_DIR = ROOT / "results" / "tori_trendline_envelopes"
NY_TZ = ZoneInfo("America/New_York")

ASSET_PATHS = {
    "XAU": CACHE_DIR / "XAU_5m.parquet",
    "UUP": CACHE_DIR / "UUP_5m.parquet",
}


@dataclass(frozen=True)
class SweepConfig:
    asset: str
    bar_minutes: int
    lookback: int
    envelope_width: float
    min_touches: int
    slope_threshold: float
    stop_buffer: float
    take_profit_r: float
    touch_tolerance: float
    break_buffer: float
    max_break_risk_pct: float


def parse_csv_list(raw: str, cast):
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        default="XAU,UUP",
        help="Comma-separated asset labels from the local cache.",
    )
    parser.add_argument(
        "--bar-minutes",
        type=int,
        default=240,
        help="Resample size in minutes. The playbook defaults to 4H, so 240 is the default.",
    )
    parser.add_argument("--lookbacks", default="14,20,28")
    parser.add_argument("--envelope-widths", default="0.9,1.2,1.5,1.8")
    parser.add_argument("--touch-options", default="2,3")
    parser.add_argument("--slope-thresholds", default="0.0002,0.0004,0.0006")
    parser.add_argument("--stop-buffers", default="0.0,0.25,0.5")
    parser.add_argument("--take-profits", default="1.5,2.0,3.0,4.0")
    parser.add_argument("--touch-tolerance", type=float, default=0.35)
    parser.add_argument("--break-buffer", type=float, default=0.15)
    parser.add_argument(
        "--max-break-risk-pct",
        type=float,
        default=0.035,
        help="Skip break entries whose distance to the safety line is too large.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=15,
        help="Minimum trades preferred by the ranking model. Lower counts are penalized as sparse evidence.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--trade-chart-limit",
        type=int,
        default=0,
        help="How many detailed trade charts to save for each asset's best config. 0 means all trades.",
    )
    parser.add_argument(
        "--skip-trade-charts",
        action="store_true",
        help="Skip generating trade chart PNGs.",
    )
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
    grouped["close_ret"] = grouped["close"].pct_change()
    return grouped


def build_regression_cache(close: np.ndarray, lookback: int) -> dict[str, np.ndarray]:
    n = len(close)
    pred = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    resid = np.full(n, np.nan)

    x = np.arange(lookback, dtype=float)
    x_mean = float(x.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0.0:
        raise ValueError(f"Invalid lookback for regression: {lookback}")

    for i in range(lookback - 1, n):
        y = close[i - lookback + 1 : i + 1]
        y_mean = float(y.mean())
        b = float(((x - x_mean) * (y - y_mean)).sum() / denom)
        a = y_mean - b * x_mean
        y_hat = a + b * x
        pred[i] = y_hat[-1]
        slope[i] = b
        resid[i] = math.sqrt(float(np.mean((y - y_hat) ** 2)))

    slope_pct = slope / np.where(np.abs(pred) < 1e-9, np.nan, pred)
    return {
        "pred": pred,
        "slope": slope,
        "slope_pct": slope_pct,
        "resid": resid,
    }


def build_model_arrays(
    df: pd.DataFrame,
    regression: dict[str, np.ndarray],
    cfg: SweepConfig,
) -> dict[str, np.ndarray | float | int]:
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    time = df["time"].to_numpy()
    time_et = df["time_et"].to_numpy()
    weekday = df["weekday"].to_numpy()
    date_et = pd.Series(df["date_et"]).astype(str).to_numpy()
    atr = df["atr"].to_numpy(dtype=float)

    pred = regression["pred"]
    slope_pct = regression["slope_pct"]
    resid = regression["resid"]
    macro_pred = regression["macro_pred"]
    macro_slope_pct = regression["macro_slope_pct"]
    macro_resid = regression["macro_resid"]

    risk_unit = np.maximum(np.maximum(resid, macro_resid * 0.5), atr)
    lower = pred - cfg.envelope_width * risk_unit
    upper = pred + cfg.envelope_width * risk_unit

    swing_high, swing_low = local_extrema_flags(high, low)
    support_touch = swing_low & (np.abs(low - lower) <= cfg.touch_tolerance * risk_unit)
    resistance_touch = swing_high & (np.abs(high - upper) <= cfg.touch_tolerance * risk_unit)

    bars_per_day = max(1.0, float(df.groupby("date_et").size().median()))
    min_week_bars = max(2, int(round(bars_per_day * 5.0)))

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "time": time,
        "time_et": time_et,
        "weekday": weekday,
        "date_et": date_et,
        "atr": atr,
        "pred": pred,
        "slope_pct": slope_pct,
        "resid": resid,
        "macro_pred": macro_pred,
        "macro_slope_pct": macro_slope_pct,
        "macro_resid": macro_resid,
        "risk_unit": risk_unit,
        "lower": lower,
        "upper": upper,
        "support_touch": support_touch,
        "resistance_touch": resistance_touch,
        "bars_per_day": bars_per_day,
        "min_week_bars": min_week_bars,
    }


def local_extrema_flags(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    swing_high = np.zeros(len(high), dtype=bool)
    swing_low = np.zeros(len(low), dtype=bool)
    if len(high) < 3:
        return swing_high, swing_low

    swing_high[1:-1] = (high[1:-1] >= high[:-2]) & (high[1:-1] > high[2:])
    swing_low[1:-1] = (low[1:-1] <= low[:-2]) & (low[1:-1] < low[2:])
    return swing_high, swing_low


def first_touch_age(indices: np.ndarray, current_idx: int) -> int | None:
    if len(indices) == 0:
        return None
    return int(current_idx - indices[0])


def build_signal_candidates(
    i: int,
    arrays: dict[str, np.ndarray],
    cfg: SweepConfig,
    min_week_bars: int,
) -> list[dict]:
    close = arrays["close"]
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    slope_pct = arrays["slope_pct"]
    macro_slope_pct = arrays["macro_slope_pct"]
    lower = arrays["lower"]
    upper = arrays["upper"]
    risk_unit = arrays["risk_unit"]
    support_touch = arrays["support_touch"]
    resistance_touch = arrays["resistance_touch"]

    if (
        np.isnan(lower[i])
        or np.isnan(upper[i])
        or np.isnan(slope_pct[i])
        or np.isnan(macro_slope_pct[i])
        or np.isnan(risk_unit[i])
    ):
        return []

    threshold = cfg.slope_threshold
    up_trend = slope_pct[i] > threshold and macro_slope_pct[i] > 0
    down_trend = slope_pct[i] < -threshold and macro_slope_pct[i] < 0

    window_start = max(0, i - cfg.lookback + 1)
    support_idx = np.flatnonzero(support_touch[window_start:i]) + window_start
    resistance_idx = np.flatnonzero(resistance_touch[window_start:i]) + window_start
    support_age = first_touch_age(support_idx, i)
    resistance_age = first_touch_age(resistance_idx, i)

    long_risk_pct = (close[i] - (lower[i] - cfg.stop_buffer * risk_unit[i])) / close[i]
    short_risk_pct = ((upper[i] + cfg.stop_buffer * risk_unit[i]) - close[i]) / close[i]

    candidates: list[dict] = []

    if (
        up_trend
        and len(support_idx) >= cfg.min_touches
        and support_age is not None
        and support_age >= min_week_bars
        and low[i] <= lower[i] + cfg.touch_tolerance * risk_unit[i]
        and close[i] >= lower[i]
        and close[i] >= open_[i]
    ):
        candidates.append(
            {
                "setup": "bounce",
                "side": 1,
                "touch_count": int(len(support_idx)),
                "touch_age_bars": support_age,
                "trend_strength": abs(float(slope_pct[i])),
                "priority": 1,
            }
        )

    if (
        down_trend
        and len(resistance_idx) >= cfg.min_touches
        and resistance_age is not None
        and resistance_age >= min_week_bars
        and high[i] >= upper[i] - cfg.touch_tolerance * risk_unit[i]
        and close[i] <= upper[i]
        and close[i] <= open_[i]
    ):
        candidates.append(
            {
                "setup": "bounce",
                "side": -1,
                "touch_count": int(len(resistance_idx)),
                "touch_age_bars": resistance_age,
                "trend_strength": abs(float(slope_pct[i])),
                "priority": 1,
            }
        )

    if (
        down_trend
        and len(resistance_idx) >= cfg.min_touches
        and resistance_age is not None
        and resistance_age >= min_week_bars
        and close[i] > upper[i] + cfg.break_buffer * risk_unit[i]
        and long_risk_pct <= cfg.max_break_risk_pct
    ):
        candidates.append(
            {
                "setup": "break",
                "side": 1,
                "touch_count": int(len(resistance_idx)),
                "touch_age_bars": resistance_age,
                "trend_strength": abs(float(slope_pct[i])),
                "priority": 2,
            }
        )

    if (
        up_trend
        and len(support_idx) >= cfg.min_touches
        and support_age is not None
        and support_age >= min_week_bars
        and close[i] < lower[i] - cfg.break_buffer * risk_unit[i]
        and short_risk_pct <= cfg.max_break_risk_pct
    ):
        candidates.append(
            {
                "setup": "break",
                "side": -1,
                "touch_count": int(len(support_idx)),
                "touch_age_bars": support_age,
                "trend_strength": abs(float(slope_pct[i])),
                "priority": 2,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["priority"],
            item["touch_count"],
            item["trend_strength"],
        ),
        reverse=True,
    )
    return candidates


def simulate_config(
    df: pd.DataFrame,
    regression: dict[str, np.ndarray],
    cfg: SweepConfig,
) -> pd.DataFrame:
    arrays = build_model_arrays(df, regression, cfg)
    close = arrays["close"]
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    time = arrays["time"]
    time_et = arrays["time_et"]
    weekday = arrays["weekday"]
    date_et = arrays["date_et"]
    risk_unit = arrays["risk_unit"]
    lower = arrays["lower"]
    upper = arrays["upper"]
    slope_pct = arrays["slope_pct"]
    macro_slope_pct = arrays["macro_slope_pct"]
    pred = arrays["pred"]
    bars_per_day = float(arrays["bars_per_day"])
    min_week_bars = int(arrays["min_week_bars"])

    warmup = max(cfg.lookback - 1, regression["macro_lookback"] - 1, min_week_bars)
    trades = []
    i = warmup
    while i < len(df) - 1:
        candidates = build_signal_candidates(i, arrays, cfg, min_week_bars)
        if not candidates:
            i += 1
            continue

        signal = candidates[0]
        side = int(signal["side"])
        entry_idx = i
        entry_price = close[entry_idx]

        if side == 1:
            base_line = lower[entry_idx]
            trailing_stop = base_line - cfg.stop_buffer * risk_unit[entry_idx]
            initial_risk = entry_price - trailing_stop
            take_profit = entry_price + cfg.take_profit_r * initial_risk
        else:
            base_line = upper[entry_idx]
            trailing_stop = base_line + cfg.stop_buffer * risk_unit[entry_idx]
            initial_risk = trailing_stop - entry_price
            take_profit = entry_price - cfg.take_profit_r * initial_risk

        if initial_risk <= 0 or not np.isfinite(initial_risk):
            i += 1
            continue

        exit_idx = None
        exit_reason = "end_of_data"
        exit_price = close[-1]
        peak_favorable = 0.0
        peak_adverse = 0.0
        stop_start = trailing_stop

        j = entry_idx + 1
        while j < len(df):
            if side == 1:
                trailing_stop = max(trailing_stop, lower[j] - cfg.stop_buffer * risk_unit[j])
                move_favorable = (high[j] - entry_price) / initial_risk
                move_adverse = (low[j] - entry_price) / initial_risk
                peak_favorable = max(peak_favorable, move_favorable)
                peak_adverse = min(peak_adverse, move_adverse)
                stop_hit = close[j] <= trailing_stop
                target_hit = high[j] >= take_profit
                if stop_hit and target_hit:
                    exit_price = close[j]
                    exit_reason = "safety_break"
                    exit_idx = j
                    break
                if stop_hit:
                    exit_price = close[j]
                    exit_reason = "safety_break"
                    exit_idx = j
                    break
                if target_hit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                    exit_idx = j
                    break
            else:
                trailing_stop = min(trailing_stop, upper[j] + cfg.stop_buffer * risk_unit[j])
                move_favorable = (entry_price - low[j]) / initial_risk
                move_adverse = (entry_price - high[j]) / initial_risk
                peak_favorable = max(peak_favorable, move_favorable)
                peak_adverse = min(peak_adverse, move_adverse)
                stop_hit = close[j] >= trailing_stop
                target_hit = low[j] <= take_profit
                if stop_hit and target_hit:
                    exit_price = close[j]
                    exit_reason = "safety_break"
                    exit_idx = j
                    break
                if stop_hit:
                    exit_price = close[j]
                    exit_reason = "safety_break"
                    exit_idx = j
                    break
                if target_hit:
                    exit_price = take_profit
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
                "touch_count": signal["touch_count"],
                "touch_age_bars": signal["touch_age_bars"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "initial_stop": stop_start,
                "final_stop": trailing_stop,
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
                "trend_slope_pct": float(slope_pct[entry_idx]),
                "macro_slope_pct": float(macro_slope_pct[entry_idx]),
                "envelope_mid": float(pred[entry_idx]),
                "envelope_lower": float(lower[entry_idx]),
                "envelope_upper": float(upper[entry_idx]),
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
    abs_slope = out["trend_slope_pct"].abs()
    if abs_slope.nunique() < 3:
        out["trend_bucket"] = "mixed"
        return out
    out["trend_bucket"] = pd.qcut(
        abs_slope,
        q=3,
        labels=["soft", "steady", "strong"],
        duplicates="drop",
    )
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


def build_trade_line_series(
    model_arrays: dict[str, np.ndarray | float | int],
    trade: pd.Series,
    cfg: SweepConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(model_arrays["lower"], dtype=float)
    upper = np.asarray(model_arrays["upper"], dtype=float)
    risk_unit = np.asarray(model_arrays["risk_unit"], dtype=float)

    entry_idx = int(trade["entry_bar_idx"])
    exit_idx = int(trade["exit_bar_idx"])
    is_long = str(trade["side"]) == "long"
    is_bounce = str(trade["setup"]) == "bounce"

    if is_bounce:
        action_line = lower.copy() if is_long else upper.copy()
    else:
        action_line = upper.copy() if is_long else lower.copy()

    raw_safety = lower - cfg.stop_buffer * risk_unit if is_long else upper + cfg.stop_buffer * risk_unit
    safety_line = raw_safety.copy()
    if is_long:
        safety_line[entry_idx:] = np.maximum.accumulate(raw_safety[entry_idx:])
    else:
        safety_line[entry_idx:] = np.minimum.accumulate(raw_safety[entry_idx:])

    take_profit = np.full(len(action_line), np.nan)
    take_profit[entry_idx : exit_idx + 1] = float(trade["take_profit_price"])
    return action_line, safety_line, take_profit


def draw_candlesticks(
    ax: plt.Axes,
    frame: pd.DataFrame,
    up_color: str = "#16a34a",
    down_color: str = "#dc2626",
):
    x = mdates.date2num(pd.to_datetime(frame["time_plot"]).tolist())
    if len(x) > 1:
        width = (x[1] - x[0]) * 0.68
    else:
        width = 0.18

    for xi, open_, high, low, close in zip(x, frame["open"], frame["high"], frame["low"], frame["close"]):
        color = up_color if close >= open_ else down_color
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
    return x


def plot_trade_overview(asset: str, df: pd.DataFrame, trades: pd.DataFrame, out_path: Path):
    if trades.empty:
        return

    view = df[["time_et", "close"]].copy()
    view["time_plot"] = pd.to_datetime(view["time_et"]).dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(view["time_plot"], view["close"], color="#475569", linewidth=1.1, alpha=0.95, label=f"{asset} close")

    trades = trades.copy()
    trades["entry_plot"] = pd.to_datetime(trades["entry_time_et"]).dt.tz_localize(None)
    trades["exit_plot"] = pd.to_datetime(trades["exit_time_et"]).dt.tz_localize(None)

    longs = trades[trades["side"] == "long"]
    shorts = trades[trades["side"] == "short"]
    ax.scatter(longs["entry_plot"], longs["entry_price"], marker="^", s=55, color="#2563eb", label="Long entry", zorder=5)
    ax.scatter(shorts["entry_plot"], shorts["entry_price"], marker="v", s=55, color="#7c3aed", label="Short entry", zorder=5)
    ax.scatter(trades["exit_plot"], trades["exit_price"], marker="X", s=45, color="#111827", label="Exit", zorder=5)

    ax.set_title(f"{asset} best-config trade overview")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", ncol=4, frameon=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_trade_detail(
    asset: str,
    df: pd.DataFrame,
    model_arrays: dict[str, np.ndarray | float | int],
    trade: pd.Series,
    cfg: SweepConfig,
    out_path: Path,
):
    entry_idx = int(trade["entry_bar_idx"])
    exit_idx = int(trade["exit_bar_idx"])
    context = max(cfg.lookback, 10)
    post_bars = max(4, cfg.lookback // 3)
    start_idx = max(0, entry_idx - context)
    end_idx = min(len(df) - 1, exit_idx + post_bars)

    frame = df.iloc[start_idx : end_idx + 1][["time_et", "open", "high", "low", "close"]].copy()
    frame["time_plot"] = pd.to_datetime(frame["time_et"]).dt.tz_localize(None)
    action_line, safety_line, take_profit = build_trade_line_series(model_arrays, trade, cfg)
    frame["action_line"] = np.asarray(action_line)[start_idx : end_idx + 1]
    frame["safety_line"] = np.asarray(safety_line)[start_idx : end_idx + 1]
    frame["take_profit"] = np.asarray(take_profit)[start_idx : end_idx + 1]
    frame["mid_line"] = np.asarray(model_arrays["pred"], dtype=float)[start_idx : end_idx + 1]

    fig, ax = plt.subplots(figsize=(15, 7))
    x = draw_candlesticks(ax, frame)

    ax.plot(frame["time_plot"], frame["action_line"], color="#dc2626", linewidth=2.0, label="Action line", zorder=4)
    ax.plot(frame["time_plot"], frame["safety_line"], color="#16a34a", linewidth=2.0, label="Safety line", zorder=4)
    ax.plot(frame["time_plot"], frame["mid_line"], color="#64748b", linewidth=1.2, linestyle="--", alpha=0.8, label="Envelope mid", zorder=3)
    if np.isfinite(frame["take_profit"]).any():
        ax.plot(frame["time_plot"], frame["take_profit"], color="#2563eb", linewidth=1.4, linestyle=":", label="Take profit", zorder=3)

    entry_time = pd.to_datetime(trade["entry_time_et"]).tz_localize(None)
    exit_time = pd.to_datetime(trade["exit_time_et"]).tz_localize(None)
    is_long = str(trade["side"]) == "long"
    entry_marker = "^" if is_long else "v"
    ax.scatter([entry_time], [float(trade["entry_price"])], marker=entry_marker, s=120, color="#1d4ed8", edgecolor="white", linewidth=0.8, zorder=6, label="Entry")
    ax.scatter([exit_time], [float(trade["exit_price"])], marker="X", s=110, color="#111827", edgecolor="white", linewidth=0.8, zorder=6, label="Exit")

    entry_num = mdates.date2num(entry_time)
    exit_num = mdates.date2num(exit_time)
    ax.axvspan(entry_num, exit_num, color="#93c5fd", alpha=0.08, zorder=1)

    ax.annotate(
        "Entry",
        (entry_time, float(trade["entry_price"])),
        xytext=(0, 12 if is_long else -20),
        textcoords="offset points",
        ha="center",
        color="#1d4ed8",
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        f"Exit: {trade['exit_reason']}",
        (exit_time, float(trade["exit_price"])),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        color="#111827",
        fontsize=9,
        fontweight="bold",
    )

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
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9, "boxstyle": "round,pad=0.4"},
    )

    ax.set_title(f"{asset} trade detail")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", ncol=3, frameon=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_trade_charts(
    asset: str,
    df: pd.DataFrame,
    model_arrays: dict[str, np.ndarray | float | int],
    trades: pd.DataFrame,
    cfg: SweepConfig,
    out_dir: Path,
    chart_limit: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_trade_overview(asset, df, trades, out_dir / f"{asset.lower()}_trade_overview.png")

    if chart_limit > 0:
        selected = trades.head(chart_limit)
    else:
        selected = trades

    for _, trade in selected.iterrows():
        label = f"trade_{int(trade['trade_id']):03d}_{trade['setup']}_{trade['side']}_{pd.to_datetime(trade['entry_time_et']).strftime('%Y%m%d_%H%M')}.png"
        plot_trade_detail(asset, df, model_arrays, trade, cfg, out_dir / label)


def iter_configs(args, assets: Iterable[str]) -> list[SweepConfig]:
    lookbacks = parse_csv_list(args.lookbacks, int)
    envelope_widths = parse_csv_list(args.envelope_widths, float)
    touch_options = parse_csv_list(args.touch_options, int)
    slope_thresholds = parse_csv_list(args.slope_thresholds, float)
    stop_buffers = parse_csv_list(args.stop_buffers, float)
    take_profits = parse_csv_list(args.take_profits, float)

    configs = []
    for asset, lookback, width, min_touches, slope_th, stop_buffer, take_profit in itertools.product(
        assets,
        lookbacks,
        envelope_widths,
        touch_options,
        slope_thresholds,
        stop_buffers,
        take_profits,
    ):
        configs.append(
            SweepConfig(
                asset=asset,
                bar_minutes=args.bar_minutes,
                lookback=lookback,
                envelope_width=width,
                min_touches=min_touches,
                slope_threshold=slope_th,
                stop_buffer=stop_buffer,
                take_profit_r=take_profit,
                touch_tolerance=args.touch_tolerance,
                break_buffer=args.break_buffer,
                max_break_risk_pct=args.max_break_risk_pct,
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

    prepared_assets: dict[str, pd.DataFrame] = {}
    regression_caches: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    baseline_rows = []

    unique_lookbacks = sorted(set(parse_csv_list(args.lookbacks, int)))
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
        close = df["close"].to_numpy(dtype=float)
        for lookback in unique_lookbacks:
            base = build_regression_cache(close, lookback)
            macro_lookback = max(lookback * 3, lookback + 6)
            if macro_lookback >= len(df):
                macro_lookback = max(lookback + 2, len(df) // 3)
            macro = build_regression_cache(close, macro_lookback)
            regression_caches[(asset, lookback)] = {
                **base,
                "macro_pred": macro["pred"],
                "macro_slope_pct": macro["slope_pct"],
                "macro_resid": macro["resid"],
                "macro_lookback": macro_lookback,
            }

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(out_dir / "asset_baselines.csv", index=False)

    summary_rows = []
    best_trade_logs: dict[str, pd.DataFrame] = {}
    best_scores: dict[str, float] = {asset: -np.inf for asset in assets}
    best_cfgs: dict[str, SweepConfig] = {}
    best_regressions: dict[str, dict[str, np.ndarray]] = {}

    configs = iter_configs(args, assets)
    for idx, cfg in enumerate(configs, start=1):
        asset_df = prepared_assets[cfg.asset]
        regression = regression_caches[(cfg.asset, cfg.lookback)]
        trades = simulate_config(asset_df, regression, cfg)
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
            best_regressions[cfg.asset] = regression

        if idx % 50 == 0:
            print(f"[{idx}/{len(configs)}] finished {cfg.asset} lookback={cfg.lookback} width={cfg.envelope_width}")

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
        if not args.skip_trade_charts and asset in best_cfgs and asset in best_regressions:
            chart_dir = out_dir / f"{asset.lower()}_trade_charts"
            model_arrays = build_model_arrays(prepared_assets[asset], best_regressions[asset], best_cfgs[asset])
            save_trade_charts(
                asset=asset,
                df=prepared_assets[asset],
                model_arrays=model_arrays,
                trades=trades,
                cfg=best_cfgs[asset],
                out_dir=chart_dir,
                chart_limit=args.trade_chart_limit,
            )

    if best_rows:
        pd.DataFrame(best_rows).sort_values(
            ["quality_score", "total_return"], ascending=[False, False]
        ).to_csv(out_dir / "best_configs.csv", index=False)

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
