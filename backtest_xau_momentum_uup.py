#!/usr/bin/env python3
"""Backtest an XAU momentum strategy with UUP confirmation and leverage stress tests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results" / "xau_momentum_uup"
NY_TZ = ZoneInfo("America/New_York")


@dataclass
class StrategyParams:
    ema_fast: int = 8
    ema_slow: int = 21
    ema_trend: int = 55
    uup_fast: int = 8
    uup_slow: int = 21
    atr_period: int = 14
    breakout_bars: int = 12
    signal_lookback: int = 3
    z_beta_window: int = 144
    z_window: int = 288
    stop_atr_mult: float = 1.35
    stop_floor_pct: float = 0.0025
    take_profit_r: float = 3.0
    trail_after_r: float = 1.0
    trail_atr_mult: float = 1.1
    max_hold_bars: int = 24
    min_score: int = 5
    max_trades_per_day: int = 2
    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    liq_buffer: float = 0.98


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xau-path", default=str(CACHE_DIR / "XAU_5m.parquet"))
    parser.add_argument("--uup-path", default=str(CACHE_DIR / "UUP_5m.parquet"))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--starting-equity", type=float, default=10000.0)
    parser.add_argument("--fixed-risk-pct", type=float, default=0.015)
    return parser.parse_args()


def load_data(xau_path: Path, uup_path: Path) -> pd.DataFrame:
    xau = pd.read_parquet(xau_path).rename(
        columns={"o": "xau_o", "h": "xau_h", "l": "xau_l", "c": "xau_c", "v": "xau_v"}
    )
    uup = pd.read_parquet(uup_path).rename(
        columns={"o": "uup_o", "h": "uup_h", "l": "uup_l", "c": "uup_c", "v": "uup_v"}
    )
    for df in [xau, uup]:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    keep_xau = ["time", "xau_o", "xau_h", "xau_l", "xau_c", "xau_v"]
    keep_uup = ["time", "uup_o", "uup_h", "uup_l", "uup_c", "uup_v"]
    df = pd.merge(xau[keep_xau], uup[keep_uup], on="time", how="inner").sort_values("time").reset_index(drop=True)
    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_et"] = out["time"].dt.tz_convert(NY_TZ)
    out["date_et"] = out["time_et"].dt.date
    out["weekday"] = out["time_et"].dt.day_name()
    out["hour_et"] = out["time_et"].dt.hour
    out["minute_et"] = out["time_et"].dt.minute
    et_minutes = out["hour_et"] * 60 + out["minute_et"]
    out["us_window"] = np.select(
        [
            (et_minutes >= 570) & (et_minutes < 600),
            (et_minutes >= 600) & (et_minutes < 630),
            (et_minutes >= 630) & (et_minutes < 690),
            (et_minutes >= 690) & (et_minutes < 840),
            (et_minutes >= 840) & (et_minutes < 900),
            (et_minutes >= 900) & (et_minutes < 930),
            (et_minutes >= 930) & (et_minutes <= 955),
        ],
        [
            "us_open_30m",
            "us_open_2nd_30m",
            "us_open_hour2",
            "us_midday",
            "power_hour",
            "close_30m",
            "close_5m_tail",
        ],
        default="off_hours",
    )
    return out


def add_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = add_session_features(df)

    out["xau_ret"] = out["xau_c"].pct_change()
    out["uup_ret"] = out["uup_c"].pct_change()
    out["xau_ema_fast"] = out["xau_c"].ewm(span=params.ema_fast, adjust=False).mean()
    out["xau_ema_slow"] = out["xau_c"].ewm(span=params.ema_slow, adjust=False).mean()
    out["xau_ema_trend"] = out["xau_c"].ewm(span=params.ema_trend, adjust=False).mean()
    out["uup_ema_fast"] = out["uup_c"].ewm(span=params.uup_fast, adjust=False).mean()
    out["uup_ema_slow"] = out["uup_c"].ewm(span=params.uup_slow, adjust=False).mean()

    prev_close = out["xau_c"].shift(1)
    tr_parts = pd.concat(
        [
            out["xau_h"] - out["xau_l"],
            (out["xau_h"] - prev_close).abs(),
            (out["xau_l"] - prev_close).abs(),
        ],
        axis=1,
    )
    out["xau_tr"] = tr_parts.max(axis=1)
    out["xau_atr"] = out["xau_tr"].rolling(params.atr_period).mean()
    out["xau_atr_pct"] = out["xau_atr"] / out["xau_c"]
    out["xau_atr_pct_med"] = out["xau_atr_pct"].rolling(60).median()

    out["xau_vol_20"] = out["xau_ret"].rolling(20).std()
    out["xau_vol_20_med"] = out["xau_vol_20"].rolling(120).median()
    out["xau_vol_ratio"] = out["xau_vol_20"] / out["xau_vol_20_med"].replace(0.0, np.nan)
    out["xau_volume_ratio"] = out["xau_v"] / out["xau_v"].rolling(20).median().replace(0.0, np.nan)

    out["rolling_high"] = out["xau_h"].shift(1).rolling(params.breakout_bars).max()
    out["rolling_low"] = out["xau_l"].shift(1).rolling(params.breakout_bars).min()
    out["xau_mom"] = out["xau_c"] / out["xau_c"].shift(params.signal_lookback) - 1.0
    out["uup_mom"] = out["uup_c"] / out["uup_c"].shift(params.signal_lookback) - 1.0

    typical = (out["xau_h"] + out["xau_l"] + out["xau_c"]) / 3.0
    pv = typical * out["xau_v"].fillna(0.0)
    out["xau_vwap"] = pv.groupby(out["date_et"]).cumsum() / out["xau_v"].fillna(0.0).groupby(out["date_et"]).cumsum().replace(0.0, np.nan)

    xau_log = np.log(out["xau_c"])
    uup_log = np.log(out["uup_c"])
    xau_log_ret = xau_log.diff()
    uup_log_ret = uup_log.diff()
    rolling_cov = xau_log_ret.rolling(params.z_beta_window).cov(uup_log_ret)
    rolling_var = uup_log_ret.rolling(params.z_beta_window).var()
    out["pair_beta"] = (rolling_cov / rolling_var.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill().clip(-4.0, 4.0)
    out["pair_spread"] = xau_log - out["pair_beta"] * uup_log
    out["pair_zscore"] = (
        (out["pair_spread"] - out["pair_spread"].rolling(params.z_window).mean())
        / out["pair_spread"].rolling(params.z_window).std().replace(0.0, np.nan)
    )
    return out


def compute_signal_score(row: pd.Series, side: int) -> tuple[int, dict[str, bool]]:
    if side == 1:
        checks = {
            "ema_stack": row["xau_ema_fast"] > row["xau_ema_slow"] > row["xau_ema_trend"],
            "ema_slope": row["xau_ema_fast"] > row["xau_ema_fast_prev"] and row["xau_ema_slow"] > row["xau_ema_slow_prev"],
            "breakout": row["xau_c"] > row["rolling_high"],
            "vwap": row["xau_c"] > row["xau_vwap"],
            "volume": row["xau_volume_ratio"] > 1.1,
            "volatility": row["xau_atr_pct"] > row["xau_atr_pct_med"] and row["xau_vol_ratio"] > 1.0,
            "uup_confirm": row["uup_ema_fast"] < row["uup_ema_slow"] and row["uup_mom"] < 0,
            "pair_support": row["pair_zscore"] < -0.75,
            "session_quality": row["us_window"] in {"us_open_30m", "close_30m", "close_5m_tail"},
        }
        core_ready = checks["ema_stack"] and checks["breakout"] and checks["vwap"]
    else:
        checks = {
            "ema_stack": row["xau_ema_fast"] < row["xau_ema_slow"] < row["xau_ema_trend"],
            "ema_slope": row["xau_ema_fast"] < row["xau_ema_fast_prev"] and row["xau_ema_slow"] < row["xau_ema_slow_prev"],
            "breakout": row["xau_c"] < row["rolling_low"],
            "vwap": row["xau_c"] < row["xau_vwap"],
            "volume": row["xau_volume_ratio"] > 1.1,
            "volatility": row["xau_atr_pct"] > row["xau_atr_pct_med"] and row["xau_vol_ratio"] > 1.0,
            "uup_confirm": row["uup_ema_fast"] > row["uup_ema_slow"] and row["uup_mom"] > 0,
            "pair_support": row["pair_zscore"] > 0.75,
            "session_quality": row["us_window"] in {"us_open_30m", "close_30m", "close_5m_tail"},
        }
        core_ready = checks["ema_stack"] and checks["breakout"] and checks["vwap"]
    score = int(sum(bool(v) for v in checks.values()))
    checks["core_ready"] = core_ready
    return score, checks


def dynamic_cap_from_score(score: int) -> int:
    if score >= 8:
        return 1000
    if score >= 7:
        return 100
    if score >= 6:
        return 50
    return 10


def dynamic_risk_from_score(score: int) -> float:
    if score >= 8:
        return 0.03
    if score >= 7:
        return 0.02
    if score >= 6:
        return 0.015
    return 0.01


def choose_leverage(mode_name: str, score: int, fixed_risk_pct: float) -> tuple[int, float, str]:
    if mode_name == "dynamic_confidence":
        return dynamic_cap_from_score(score), dynamic_risk_from_score(score), "risk_capped"
    if mode_name.startswith("risk_capped_"):
        lev = int(mode_name.split("_")[-1].replace("x", ""))
        return lev, fixed_risk_pct, "risk_capped"
    if mode_name.startswith("full_notional_"):
        lev = int(mode_name.split("_")[-1].replace("x", ""))
        return lev, fixed_risk_pct, "full_notional"
    raise ValueError(f"Unknown mode: {mode_name}")


def apply_entry_slippage(raw_open: float, side: int, bps: float) -> float:
    slip = bps / 10000.0
    return raw_open * (1.0 + slip) if side == 1 else raw_open * (1.0 - slip)


def apply_exit_slippage(raw_px: float, side: int, bps: float) -> float:
    slip = bps / 10000.0
    return raw_px * (1.0 - slip) if side == 1 else raw_px * (1.0 + slip)


def compute_position_size(
    equity: float,
    entry_px: float,
    stop_px: float,
    leverage_cap: int,
    risk_pct: float,
    sizing_mode: str,
) -> tuple[float, float]:
    stop_move = abs(entry_px - stop_px) / entry_px
    if stop_move <= 0:
        return 0.0, 0.0
    if sizing_mode == "full_notional":
        effective_lev = float(leverage_cap)
    else:
        risk_based_lev = risk_pct / stop_move
        effective_lev = float(min(leverage_cap, risk_based_lev))
    notional = equity * effective_lev
    return notional, effective_lev


def raw_trade_return(side: int, entry_px: float, exit_px: float) -> float:
    if side == 1:
        return exit_px / entry_px - 1.0
    return entry_px / exit_px - 1.0


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    side: int,
    entry_px: float,
    stop_px: float,
    target_px: float,
    effective_lev: float,
    params: StrategyParams,
) -> dict:
    entry_row = df.iloc[entry_idx]
    liq_move = params.liq_buffer / max(effective_lev, 1e-9)
    if side == 1:
        liq_px = entry_px * (1.0 - liq_move)
    else:
        liq_px = entry_px * (1.0 + liq_move)

    stop_px_live = stop_px
    max_favorable = 0.0
    max_adverse = 0.0
    exit_idx = entry_idx
    exit_reason = "time_stop"
    exit_px = df.iloc[min(entry_idx + params.max_hold_bars, len(df) - 1)]["xau_c"]
    bars_held = 0

    for i in range(entry_idx, len(df)):
        row = df.iloc[i]
        bars_held = i - entry_idx + 1
        high_px = float(row["xau_h"])
        low_px = float(row["xau_l"])
        close_px = float(row["xau_c"])
        favorable = raw_trade_return(side, entry_px, high_px if side == 1 else low_px)
        adverse = -raw_trade_return(side, entry_px, low_px if side == 1 else high_px)
        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        if side == 1:
            if low_px <= liq_px:
                exit_px = liq_px
                exit_reason = "liquidation"
                exit_idx = i
                break
            if low_px <= stop_px_live:
                exit_px = stop_px_live
                exit_reason = "stop"
                exit_idx = i
                break
            if high_px >= target_px:
                exit_px = target_px
                exit_reason = "target"
                exit_idx = i
                break
        else:
            if high_px >= liq_px:
                exit_px = liq_px
                exit_reason = "liquidation"
                exit_idx = i
                break
            if high_px >= stop_px_live:
                exit_px = stop_px_live
                exit_reason = "stop"
                exit_idx = i
                break
            if low_px <= target_px:
                exit_px = target_px
                exit_reason = "target"
                exit_idx = i
                break

        if favorable >= params.trail_after_r * abs(entry_px - stop_px) / entry_px:
            trail_atr = float(row["xau_atr"])
            if np.isfinite(trail_atr) and trail_atr > 0:
                if side == 1:
                    stop_px_live = max(stop_px_live, close_px - params.trail_atr_mult * trail_atr)
                else:
                    stop_px_live = min(stop_px_live, close_px + params.trail_atr_mult * trail_atr)

        is_session_end = str(row["us_window"]) == "close_5m_tail"
        if bars_held >= params.max_hold_bars or is_session_end:
            exit_px = close_px
            exit_reason = "session_exit" if is_session_end else "time_stop"
            exit_idx = i
            break

    return {
        "exit_idx": exit_idx,
        "exit_time": df.iloc[exit_idx]["time"],
        "exit_price_raw": float(exit_px),
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "mfe_raw": max_favorable,
        "mae_raw": max_adverse,
    }


def compute_max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def build_strategy_frame(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = add_indicators(df, params)
    out["xau_ema_fast_prev"] = out["xau_ema_fast"].shift(1)
    out["xau_ema_slow_prev"] = out["xau_ema_slow"].shift(1)
    return out


def run_mode(
    df: pd.DataFrame,
    params: StrategyParams,
    mode_name: str,
    starting_equity: float,
    fixed_risk_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    equity = starting_equity
    equity_curve = []
    trades = []
    warmup = max(params.ema_trend, params.z_window, 120)
    day_trade_count: dict = {}
    i = warmup

    while i < len(df) - 2:
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        if str(row["us_window"]) == "off_hours":
            i += 1
            continue

        trade_day = row["date_et"]
        trades_today = day_trade_count.get(trade_day, 0)
        if trades_today >= params.max_trades_per_day:
            i += 1
            continue

        chosen: Optional[tuple[int, int, dict[str, bool]]] = None
        for side in [1, -1]:
            score, checks = compute_signal_score(row, side)
            if score < params.min_score or not checks["core_ready"]:
                continue
            if chosen is None or score > chosen[0]:
                chosen = (score, side, checks)

        if chosen is None:
            i += 1
            continue

        score, side, checks = chosen
        leverage_cap, risk_pct, sizing_mode = choose_leverage(mode_name, score, fixed_risk_pct)
        entry_px = apply_entry_slippage(float(next_row["xau_o"]), side, params.slippage_bps)
        atr = float(row["xau_atr"])
        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        stop_dist_abs = max(params.stop_atr_mult * atr, params.stop_floor_pct * entry_px)
        stop_px = entry_px - stop_dist_abs if side == 1 else entry_px + stop_dist_abs
        target_px = entry_px + params.take_profit_r * stop_dist_abs if side == 1 else entry_px - params.take_profit_r * stop_dist_abs
        notional, effective_lev = compute_position_size(
            equity=equity,
            entry_px=entry_px,
            stop_px=stop_px,
            leverage_cap=leverage_cap,
            risk_pct=risk_pct,
            sizing_mode=sizing_mode,
        )
        if notional <= 0 or effective_lev <= 0:
            i += 1
            continue

        sim = simulate_trade(df, i + 1, side, entry_px, stop_px, target_px, effective_lev, params)
        exit_px = apply_exit_slippage(sim["exit_price_raw"], side, params.slippage_bps)
        trade_ret_raw = raw_trade_return(side, entry_px, exit_px)
        commission_cost = effective_lev * (params.commission_bps / 10000.0) * 2.0
        equity_ret = effective_lev * trade_ret_raw - commission_cost
        equity *= max(0.0, 1.0 + equity_ret)

        trade = {
            "mode": mode_name,
            "entry_time": next_row["time"],
            "exit_time": sim["exit_time"],
            "date_et": str(trade_day),
            "weekday": row["weekday"],
            "us_window": row["us_window"],
            "side": "long" if side == 1 else "short",
            "score": score,
            "entry_price": entry_px,
            "stop_price": stop_px,
            "target_price": target_px,
            "exit_price": exit_px,
            "exit_reason": sim["exit_reason"],
            "bars_held": sim["bars_held"],
            "trade_return_raw": trade_ret_raw,
            "equity_return": equity_ret,
            "equity_after": equity,
            "mfe_raw": sim["mfe_raw"],
            "mae_raw": sim["mae_raw"],
            "effective_leverage": effective_lev,
            "leverage_cap": leverage_cap,
            "risk_pct": risk_pct,
            "sizing_mode": sizing_mode,
            "xau_atr_pct": row["xau_atr_pct"],
            "xau_volume_ratio": row["xau_volume_ratio"],
            "pair_zscore": row["pair_zscore"],
            "uup_mom": row["uup_mom"],
            "signal_checks": ",".join(k for k, v in checks.items() if v),
        }
        trades.append(trade)
        equity_curve.append({"time": sim["exit_time"], "mode": mode_name, "equity": equity})
        day_trade_count[trade_day] = trades_today + 1

        if equity <= starting_equity * 0.05:
            break
        i = max(sim["exit_idx"] + 1, i + 1)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    if equity_df.empty:
        equity_df = pd.DataFrame([{"time": df.iloc[0]["time"], "mode": mode_name, "equity": starting_equity}])

    summary = summarize_mode(mode_name, trades_df, equity_df, starting_equity)
    return trades_df, equity_df, summary


def summarize_mode(mode_name: str, trades_df: pd.DataFrame, equity_df: pd.DataFrame, starting_equity: float) -> dict:
    if trades_df.empty:
        return {
            "mode": mode_name,
            "trades": 0,
            "win_rate": np.nan,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": np.nan,
            "avg_trade_return_pct": np.nan,
            "median_trade_return_pct": np.nan,
            "avg_effective_leverage": np.nan,
            "max_effective_leverage": np.nan,
            "liquidations": 0,
            "stops": 0,
            "targets": 0,
            "session_exits": 0,
            "time_exits": 0,
        }

    wins = trades_df.loc[trades_df["equity_return"] > 0, "equity_return"]
    losses = trades_df.loc[trades_df["equity_return"] < 0, "equity_return"]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    end_equity = float(equity_df["equity"].iloc[-1])
    max_dd = compute_max_drawdown(equity_df["equity"])
    return {
        "mode": mode_name,
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["equity_return"] > 0).mean()),
        "total_return_pct": float(end_equity / starting_equity - 1.0) * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
        "avg_trade_return_pct": float(trades_df["equity_return"].mean()) * 100.0,
        "median_trade_return_pct": float(trades_df["equity_return"].median()) * 100.0,
        "avg_effective_leverage": float(trades_df["effective_leverage"].mean()),
        "max_effective_leverage": float(trades_df["effective_leverage"].max()),
        "liquidations": int((trades_df["exit_reason"] == "liquidation").sum()),
        "stops": int((trades_df["exit_reason"] == "stop").sum()),
        "targets": int((trades_df["exit_reason"] == "target").sum()),
        "session_exits": int((trades_df["exit_reason"] == "session_exit").sum()),
        "time_exits": int((trades_df["exit_reason"] == "time_stop").sum()),
    }


def save_outputs(out_dir: Path, summary_df: pd.DataFrame, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    trades_df.to_csv(out_dir / "trade_log.csv", index=False)
    equity_df.to_csv(out_dir / "equity_curve.csv", index=False)


def main():
    args = parse_args()
    params = StrategyParams()
    raw = load_data(Path(args.xau_path), Path(args.uup_path))
    frame = build_strategy_frame(raw, params)

    modes = [
        "dynamic_confidence",
        "risk_capped_10x",
        "risk_capped_50x",
        "risk_capped_100x",
        "risk_capped_1000x",
        "full_notional_10x",
        "full_notional_50x",
        "full_notional_100x",
        "full_notional_1000x",
    ]

    summaries = []
    all_trades = []
    all_equity = []
    for mode in modes:
        trades_df, equity_df, summary = run_mode(
            df=frame,
            params=params,
            mode_name=mode,
            starting_equity=args.starting_equity,
            fixed_risk_pct=args.fixed_risk_pct,
        )
        summaries.append(summary)
        all_trades.append(trades_df)
        all_equity.append(equity_df)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["total_return_pct", "profit_factor", "win_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_out = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    out_dir = Path(args.out_dir)
    save_outputs(out_dir, summary_df, trades_out, equity_out)

    print(f"Saved outputs to {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
