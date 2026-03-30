#!/usr/bin/env python3
"""Sweep divergence parameters for UUP/XAU and rank more consistent regimes."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_uup_xau_divergence as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "uup_xau_divergence_sweep"


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uup-path", default=str(base.CACHE_DIR / "UUP_5m.parquet"))
    parser.add_argument("--xau-path", default=str(base.CACHE_DIR / "XAU_5m.parquet"))
    parser.add_argument("--beta-windows", default="144,288,576")
    parser.add_argument("--z-windows", default="144,288,576")
    parser.add_argument("--move-windows", default="6,12,24")
    parser.add_argument("--entry-zs", default="1.75,2.0,2.25,2.5,3.0")
    parser.add_argument("--exit-zs", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--min-overall-entries", type=int, default=40)
    parser.add_argument("--min-segment-entries", type=int, default=20)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def safe_median(series: pd.Series) -> float:
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return np.nan
    return float(cleaned.median())


def evaluate_subset(events: pd.DataFrame, min_entries: int) -> dict:
    traded = events.dropna(subset=["entry_time"]).copy()
    if len(traded) < min_entries:
        return {}

    mean_ret = float(traded["pair_return_to_normalize"].mean())
    std_ret = float(traded["pair_return_to_normalize"].std(ddof=1)) if len(traded) > 1 else np.nan
    downside = traded.loc[traded["pair_return_to_normalize"] < 0, "pair_return_to_normalize"]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    t_score = np.nan
    if std_ret and not np.isnan(std_ret) and std_ret > 0:
        t_score = float(np.sqrt(len(traded)) * mean_ret / std_ret)
    sortino_like = np.nan
    if downside_std and not np.isnan(downside_std) and downside_std > 0:
        sortino_like = float(np.sqrt(len(traded)) * mean_ret / downside_std)

    return {
        "entries": int(len(traded)),
        "event_count": int(len(events)),
        "entry_rate": float(len(traded) / len(events)) if len(events) else np.nan,
        "win_rate": float(traded["profitable_at_normalize"].mean()),
        "mean_pair_return": mean_ret,
        "median_pair_return": float(traded["pair_return_to_normalize"].median()),
        "std_pair_return": std_ret,
        "t_score": t_score,
        "sortino_like": sortino_like,
        "median_realized_rr": safe_median(traded["realized_rr"]),
        "median_best_rr": safe_median(traded["best_rr"]),
        "median_minutes_to_normalize": float(events["minutes_start_to_normalize"].median()),
        "median_minutes_entry_to_normalize": float(traded["minutes_entry_to_normalize"].median()),
        "mean_peak_abs_z": float(events["peak_z"].abs().mean()),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beta_windows = parse_int_list(args.beta_windows)
    z_windows = parse_int_list(args.z_windows)
    move_windows = parse_int_list(args.move_windows)
    entry_zs = parse_float_list(args.entry_zs)
    exit_zs = parse_float_list(args.exit_zs)

    pair = base.load_pair(Path(args.uup_path), Path(args.xau_path))

    feature_cache: dict[tuple[int, int, int], pd.DataFrame] = {}
    overall_rows = []
    segment_rows = []

    feature_combos = list(itertools.product(beta_windows, z_windows, move_windows))
    threshold_combos = list(itertools.product(entry_zs, exit_zs))

    for feature_idx, (beta_window, z_window, move_window) in enumerate(feature_combos, start=1):
        print(
            f"[features {feature_idx}/{len(feature_combos)}] "
            f"beta={beta_window} z={z_window} move={move_window}",
            flush=True,
        )
        features = base.build_features(pair, beta_window=beta_window, z_window=z_window, move_window=move_window)
        feature_cache[(beta_window, z_window, move_window)] = features

        for entry_z, exit_z in threshold_combos:
            if exit_z >= entry_z:
                continue

            events = base.detect_events(features, entry_z=entry_z, exit_z=exit_z)
            if events.empty:
                continue

            common = {
                "beta_window": beta_window,
                "z_window": z_window,
                "move_window": move_window,
                "entry_z": entry_z,
                "exit_z": exit_z,
            }

            overall = evaluate_subset(events, min_entries=args.min_overall_entries)
            if overall:
                overall_rows.append({**common, **overall})

            for segment_type in [
                "direction",
                "move_context",
                "weekday",
                "session_et",
                "us_window",
                "global_window",
                "hour_et",
            ]:
                for segment_value, group in events.groupby(segment_type):
                    metrics = evaluate_subset(group, min_entries=args.min_segment_entries)
                    if not metrics:
                        continue
                    segment_rows.append(
                        {
                            **common,
                            "segment_type": segment_type,
                            "segment_value": segment_value,
                            **metrics,
                        }
                    )

    overall_df = pd.DataFrame(overall_rows)
    segment_df = pd.DataFrame(segment_rows)

    if overall_df.empty:
        print("No configurations met the minimum entry threshold.")
        return

    overall_df = overall_df.sort_values(
        ["t_score", "mean_pair_return", "win_rate", "median_realized_rr", "entries"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    segment_df = segment_df.sort_values(
        ["segment_type", "segment_value", "t_score", "mean_pair_return", "win_rate", "entries"],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)

    best_overall = overall_df.head(25)
    best_by_segment = (
        segment_df.groupby(["segment_type", "segment_value"], dropna=False, as_index=False)
        .head(10)
        .reset_index(drop=True)
    )

    overall_df.to_csv(out_dir / "sweep_overall.csv", index=False)
    segment_df.to_csv(out_dir / "sweep_by_segment.csv", index=False)
    best_overall.to_csv(out_dir / "best_overall.csv", index=False)
    best_by_segment.to_csv(out_dir / "best_by_segment.csv", index=False)

    top = best_overall.iloc[0]
    print(f"Saved sweep results to {out_dir}")
    print(
        "Best overall config: "
        f"beta={int(top['beta_window'])}, z={int(top['z_window'])}, move={int(top['move_window'])}, "
        f"entry_z={top['entry_z']}, exit_z={top['exit_z']}"
    )
    print(
        f"Entries={int(top['entries'])}, win_rate={top['win_rate']:.2%}, "
        f"mean_return={top['mean_pair_return']:.4%}, t_score={top['t_score']:.2f}"
    )


if __name__ == "__main__":
    main()
