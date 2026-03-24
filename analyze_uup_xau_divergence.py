#!/usr/bin/env python3
"""Analyze 5-minute divergence/mean-reversion patterns between UUP and XAU."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results" / "uup_xau_divergence"
NY_TZ = ZoneInfo("America/New_York")


@dataclass
class EventResult:
    event_id: int
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    entry_time: Optional[pd.Timestamp]
    normalize_time: pd.Timestamp
    direction: str
    trade_side: Optional[str]
    move_context: str
    weekday: str
    hour_utc: int
    hour_et: int
    session_et: str
    start_z: float
    peak_z: float
    entry_z: Optional[float]
    normalize_z: float
    beta_at_entry: Optional[float]
    bars_start_to_normalize: int
    minutes_start_to_normalize: int
    bars_entry_to_normalize: Optional[int]
    minutes_entry_to_normalize: Optional[int]
    xau_window_return: float
    uup_window_return: float
    pair_return_to_normalize: Optional[float]
    mfe_return: Optional[float]
    mae_return: Optional[float]
    realized_rr: Optional[float]
    best_rr: Optional[float]
    profitable_at_normalize: Optional[bool]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uup-path", default=str(CACHE_DIR / "UUP_5m.parquet"))
    parser.add_argument("--xau-path", default=str(CACHE_DIR / "XAU_5m.parquet"))
    parser.add_argument("--beta-window", type=int, default=288, help="Rolling beta window in bars.")
    parser.add_argument("--z-window", type=int, default=288, help="Rolling z-score window in bars.")
    parser.add_argument("--move-window", type=int, default=12, help="Return-context window in bars.")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Divergence threshold.")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Normalization threshold.")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


def load_pair(uup_path: Path, xau_path: Path) -> pd.DataFrame:
    uup = pd.read_parquet(uup_path)[["time", "c"]].rename(columns={"c": "uup_close"})
    xau = pd.read_parquet(xau_path)[["time", "c"]].rename(columns={"c": "xau_close"})
    uup["time"] = pd.to_datetime(uup["time"], utc=True)
    xau["time"] = pd.to_datetime(xau["time"], utc=True)
    df = pd.merge(xau, uup, on="time", how="inner").sort_values("time").reset_index(drop=True)
    return df


def build_features(
    df: pd.DataFrame,
    beta_window: int,
    z_window: int,
    move_window: int,
) -> pd.DataFrame:
    out = df.copy()
    out["xau_log"] = np.log(out["xau_close"])
    out["uup_log"] = np.log(out["uup_close"])
    out["xau_ret"] = out["xau_log"].diff()
    out["uup_ret"] = out["uup_log"].diff()

    rolling_cov = out["xau_ret"].rolling(beta_window).cov(out["uup_ret"])
    rolling_var = out["uup_ret"].rolling(beta_window).var()
    out["beta"] = rolling_cov / rolling_var.replace(0.0, np.nan)
    out["beta"] = out["beta"].replace([np.inf, -np.inf], np.nan)
    out["beta"] = out["beta"].ffill()
    out["beta"] = out["beta"].clip(-4.0, 4.0)

    out["spread"] = out["xau_log"] - out["beta"] * out["uup_log"]
    out["spread_mean"] = out["spread"].rolling(z_window).mean()
    out["spread_std"] = out["spread"].rolling(z_window).std()
    out["zscore"] = (out["spread"] - out["spread_mean"]) / out["spread_std"].replace(0.0, np.nan)

    out["xau_window_return"] = out["xau_close"].pct_change(move_window)
    out["uup_window_return"] = out["uup_close"].pct_change(move_window)
    out["time_et"] = out["time"].dt.tz_convert(NY_TZ)
    out["weekday"] = out["time_et"].dt.day_name()
    out["hour_utc"] = out["time"].dt.hour
    out["hour_et"] = out["time_et"].dt.hour
    out["session_et"] = np.select(
        [
            out["hour_et"] < 9,
            (out["hour_et"] >= 9) & (out["hour_et"] < 11),
            (out["hour_et"] >= 11) & (out["hour_et"] < 14),
            (out["hour_et"] >= 14) & (out["hour_et"] < 16),
        ],
        [
            "pre_us",
            "us_open",
            "us_midday",
            "us_close",
        ],
        default="post_us",
    )
    return out


def classify_move_context(xau_ret: float, uup_ret: float) -> str:
    x_sign = "up" if xau_ret >= 0 else "down"
    u_sign = "up" if uup_ret >= 0 else "down"
    return f"xau_{x_sign}_uup_{u_sign}"


def rr_from_excursions(reward: Optional[float], risk: Optional[float]) -> Optional[float]:
    if reward is None or risk is None:
        return None
    if risk == 0:
        if reward > 0:
            return float("inf")
        return None
    return reward / risk


def build_trade_return_path(path: pd.DataFrame, entry: pd.Series, sign: float) -> pd.Series:
    hedge_abs = float(np.clip(abs(entry["beta"]), 0.25, 4.0))
    xau_weight = 1.0 / (1.0 + hedge_abs)
    uup_weight = hedge_abs / (1.0 + hedge_abs)

    if sign > 0:
        xau_side = -1.0
        uup_side = 1.0
    else:
        xau_side = 1.0
        uup_side = -1.0

    xau_leg = xau_side * (path["xau_close"] / float(entry["xau_close"]) - 1.0)
    uup_leg = uup_side * (path["uup_close"] / float(entry["uup_close"]) - 1.0)
    return xau_weight * xau_leg + uup_weight * uup_leg


def detect_events(df: pd.DataFrame, entry_z: float, exit_z: float) -> pd.DataFrame:
    usable = df.dropna(subset=["zscore", "beta", "xau_window_return", "uup_window_return"]).reset_index()
    if usable.empty:
        return pd.DataFrame()

    rows = []
    event_id = 1
    i = 1
    while i < len(usable):
        z_prev = usable.at[i - 1, "zscore"]
        z_now = usable.at[i, "zscore"]
        if abs(z_prev) < entry_z <= abs(z_now):
            start_i = i
            sign = 1.0 if z_now > 0 else -1.0
            peak_i = i
            peak_abs = abs(z_now)
            entry_i = None
            normalize_i = None

            j = i + 1
            while j < len(usable):
                z_j = usable.at[j, "zscore"]
                if np.sign(z_j) == sign and abs(z_j) > peak_abs:
                    peak_abs = abs(z_j)
                    peak_i = j

                if entry_i is None and j > peak_i:
                    prev_abs = abs(usable.at[j - 1, "zscore"])
                    if np.sign(z_j) == sign and abs(z_j) < prev_abs:
                        entry_i = j

                if abs(z_j) <= exit_z:
                    normalize_i = j
                    break
                j += 1

            if normalize_i is None:
                break

            start = usable.loc[start_i]
            peak = usable.loc[peak_i]
            normalize = usable.loc[normalize_i]
            entry = usable.loc[entry_i] if entry_i is not None else None

            direction = "xau_rich_uup_cheap" if sign > 0 else "xau_cheap_uup_rich"
            trade_side = "short_xau_long_uup" if sign > 0 else "long_xau_short_uup"
            move_context = classify_move_context(start["xau_window_return"], start["uup_window_return"])

            pair_return_to_normalize = None
            mfe_return = None
            mae_return = None
            realized_rr = None
            best_rr = None
            profitable_at_normalize = None
            bars_entry_to_normalize = None
            minutes_entry_to_normalize = None
            entry_z_value = None
            beta_at_entry = None

            if entry is not None:
                path = usable.loc[entry_i:normalize_i].copy()
                beta_at_entry = float(entry["beta"])
                trade_returns = build_trade_return_path(path, entry, sign)
                pair_return_to_normalize = float(trade_returns.iloc[-1])
                mfe_return = float(trade_returns.max())
                mae_return = float(max(-trade_returns.min(), 0.0))
                realized_rr = rr_from_excursions(pair_return_to_normalize, mae_return)
                best_rr = rr_from_excursions(mfe_return, mae_return)
                profitable_at_normalize = pair_return_to_normalize > 0
                bars_entry_to_normalize = int(normalize_i - entry_i)
                minutes_entry_to_normalize = bars_entry_to_normalize * 5
                entry_z_value = float(entry["zscore"])

            rows.append(
                EventResult(
                    event_id=event_id,
                    start_time=start["time"],
                    peak_time=peak["time"],
                    entry_time=entry["time"] if entry is not None else None,
                    normalize_time=normalize["time"],
                    direction=direction,
                    trade_side=trade_side if entry is not None else None,
                    move_context=move_context,
                    weekday=str(start["weekday"]),
                    hour_utc=int(start["hour_utc"]),
                    hour_et=int(start["hour_et"]),
                    session_et=str(start["session_et"]),
                    start_z=float(start["zscore"]),
                    peak_z=float(peak["zscore"]),
                    entry_z=entry_z_value,
                    normalize_z=float(normalize["zscore"]),
                    beta_at_entry=beta_at_entry,
                    bars_start_to_normalize=int(normalize_i - start_i),
                    minutes_start_to_normalize=int(normalize_i - start_i) * 5,
                    bars_entry_to_normalize=bars_entry_to_normalize,
                    minutes_entry_to_normalize=minutes_entry_to_normalize,
                    xau_window_return=float(start["xau_window_return"]),
                    uup_window_return=float(start["uup_window_return"]),
                    pair_return_to_normalize=pair_return_to_normalize,
                    mfe_return=mfe_return,
                    mae_return=mae_return,
                    realized_rr=realized_rr,
                    best_rr=best_rr,
                    profitable_at_normalize=profitable_at_normalize,
                ).__dict__
            )
            event_id += 1
            i = normalize_i + 1
            continue
        i += 1

    return pd.DataFrame(rows)


def summarize_events(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if events.empty:
        return {
            "summary": pd.DataFrame(),
            "by_direction": pd.DataFrame(),
            "by_move_context": pd.DataFrame(),
            "by_weekday": pd.DataFrame(),
            "by_hour_utc": pd.DataFrame(),
            "by_hour_et": pd.DataFrame(),
            "by_session_et": pd.DataFrame(),
        }

    traded = events.dropna(subset=["entry_time"]).copy()

    summary = pd.DataFrame(
        [
            {
                "events": int(len(events)),
                "events_with_entry": int(len(traded)),
                "entry_rate": float(len(traded) / len(events)),
                "median_minutes_to_normalize": float(events["minutes_start_to_normalize"].median()),
                "median_minutes_entry_to_normalize": float(traded["minutes_entry_to_normalize"].median()) if not traded.empty else np.nan,
                "mean_pair_return_to_normalize": float(traded["pair_return_to_normalize"].mean()) if not traded.empty else np.nan,
                "win_rate_to_normalize": float(traded["profitable_at_normalize"].mean()) if not traded.empty else np.nan,
                "median_realized_rr": float(traded["realized_rr"].replace([np.inf, -np.inf], np.nan).median()) if not traded.empty else np.nan,
                "median_best_rr": float(traded["best_rr"].replace([np.inf, -np.inf], np.nan).median()) if not traded.empty else np.nan,
            }
        ]
    )

    def grouped(group_cols: list[str]) -> pd.DataFrame:
        agg = (
            events.groupby(group_cols, dropna=False)
            .agg(
                events=("event_id", "count"),
                median_minutes_to_normalize=("minutes_start_to_normalize", "median"),
                median_peak_z=("peak_z", "median"),
            )
            .reset_index()
        )
        if traded.empty:
            return agg

        traded_agg = (
            traded.groupby(group_cols, dropna=False)
            .agg(
                entries=("event_id", "count"),
                win_rate=("profitable_at_normalize", "mean"),
                mean_pair_return=("pair_return_to_normalize", "mean"),
                median_realized_rr=("realized_rr", lambda s: s.replace([np.inf, -np.inf], np.nan).median()),
                median_best_rr=("best_rr", lambda s: s.replace([np.inf, -np.inf], np.nan).median()),
                median_minutes_entry_to_normalize=("minutes_entry_to_normalize", "median"),
            )
            .reset_index()
        )
        return agg.merge(traded_agg, on=group_cols, how="left")

    return {
        "summary": summary,
        "by_direction": grouped(["direction"]),
        "by_move_context": grouped(["move_context"]),
        "by_weekday": grouped(["weekday"]),
        "by_hour_utc": grouped(["hour_utc"]),
        "by_hour_et": grouped(["hour_et"]),
        "by_session_et": grouped(["session_et"]),
    }


def save_outputs(df: pd.DataFrame, events: pd.DataFrame, summaries: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "merged_features.parquet", index=False)
    events.to_csv(out_dir / "divergence_events.csv", index=False)
    for name, summary_df in summaries.items():
        summary_df.to_csv(out_dir / f"{name}.csv", index=False)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    df = load_pair(Path(args.uup_path), Path(args.xau_path))
    features = build_features(
        df=df,
        beta_window=args.beta_window,
        z_window=args.z_window,
        move_window=args.move_window,
    )
    events = detect_events(features, entry_z=args.entry_z, exit_z=args.exit_z)
    summaries = summarize_events(events)
    save_outputs(features, events, summaries, out_dir)

    print(f"Saved outputs to {out_dir}")
    if events.empty:
        print("No divergence events found with the current thresholds.")
        return

    traded = events.dropna(subset=["entry_time"])
    print(f"Events detected: {len(events)}")
    print(f"Events with entry trigger: {len(traded)}")
    print(f"Median minutes to normalize: {events['minutes_start_to_normalize'].median():.1f}")
    if not traded.empty:
        print(f"Win rate to normalization: {traded['profitable_at_normalize'].mean():.2%}")
        print(f"Mean pair return to normalization: {traded['pair_return_to_normalize'].mean():.4%}")
        print(
            "Median realized R:R: "
            f"{traded['realized_rr'].replace([np.inf, -np.inf], np.nan).median():.2f}"
        )


if __name__ == "__main__":
    main()
