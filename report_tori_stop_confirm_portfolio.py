#!/usr/bin/env python3
"""Build a leveraged portfolio report for the stop-confirm Tori trendline results."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SIGNAL_DIR = ROOT / "results" / "tori_trendline_touchlines_stop_confirm_fullhist_v2"
DEFAULT_OUT_DIR = ROOT / "results" / "tori_trendline_stop_confirm_portfolio_report"
ASSET_ORDER = ["XAU", "XAG", "COPPER", "UUP"]


@dataclass(frozen=True)
class SimulationConfig:
    start_equity_per_asset: float
    risk_pct: float
    leverage_cap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dir", default=str(DEFAULT_SIGNAL_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--assets", default="XAU,XAG,COPPER,UUP")
    parser.add_argument("--start-equity-per-asset", type=float, default=1000.0)
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=5.0,
        help="Risk budget per trade. Accepts 5 for 5%% or 0.05 for 5%%.",
    )
    parser.add_argument(
        "--leverage-caps",
        default="20,30",
        help="Comma-separated leverage caps to test.",
    )
    parser.add_argument(
        "--min-setup-trades",
        type=int,
        default=8,
        help="Minimum trades for setup ranking tables.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str, cast) -> list:
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def normalize_risk_pct(raw: float) -> float:
    return raw / 100.0 if raw > 1.0 else raw


def add_trend_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    abs_slope = out["action_slope_pct"].abs()
    if abs_slope.nunique() < 3:
        out["trend_bucket"] = "mixed"
        return out
    out["trend_bucket"] = pd.qcut(abs_slope, q=3, labels=["soft", "steady", "strong"], duplicates="drop")
    out["trend_bucket"] = out["trend_bucket"].astype(str)
    return out


def load_trade_logs(signal_dir: Path, assets: list[str]) -> dict[str, pd.DataFrame]:
    trade_logs: dict[str, pd.DataFrame] = {}
    for asset in assets:
        path = signal_dir / f"{asset.lower()}_best_trade_log.csv"
        df = pd.read_csv(
            path,
            parse_dates=["setup_time_utc", "entry_time_utc", "exit_time_utc", "setup_time_et", "entry_time_et", "exit_time_et"],
        )
        df["asset"] = asset
        trade_logs[asset] = add_trend_buckets(df.sort_values("entry_time_utc").reset_index(drop=True))
    return trade_logs


def master_calendar(signal_dir: Path) -> pd.DatetimeIndex:
    baselines = pd.read_csv(signal_dir / "asset_baselines.csv", parse_dates=["start_utc", "end_utc"])
    start = pd.Timestamp(baselines["start_utc"].min()).normalize()
    end = pd.Timestamp(baselines["end_utc"].max()).normalize()
    return pd.date_range(start=start, end=end, freq="D")


def simulate_asset_trades(trades: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    equity = float(cfg.start_equity_per_asset)
    rows: list[dict] = []

    for row in trades.itertuples(index=False):
        if equity <= 0:
            break

        risk_frac = max(float(row.risk_pct_of_entry), 1e-12)
        target_leverage = cfg.risk_pct / risk_frac
        used_leverage = min(float(cfg.leverage_cap), float(target_leverage))
        entry_equity = equity
        notional_usd = entry_equity * used_leverage
        planned_risk_usd = notional_usd * risk_frac
        raw_pnl_usd = notional_usd * float(row.return_pct)
        pnl_usd = max(-entry_equity, raw_pnl_usd)
        equity = max(0.0, entry_equity + pnl_usd)

        rec = row._asdict()
        rec.update(
            {
                "risk_budget_pct": cfg.risk_pct,
                "leverage_cap": cfg.leverage_cap,
                "target_leverage": float(target_leverage),
                "used_leverage": float(used_leverage),
                "entry_equity_usd": float(entry_equity),
                "exit_equity_usd": float(equity),
                "notional_usd": float(notional_usd),
                "planned_risk_usd": float(planned_risk_usd),
                "pnl_usd": float(pnl_usd),
                "return_on_equity_pct": float(pnl_usd / entry_equity) if entry_equity > 0 else 0.0,
                "blown_up": bool(equity <= 0),
                "exit_date": pd.Timestamp(row.exit_time_utc).normalize(),
            }
        )
        rows.append(rec)

    return pd.DataFrame(rows)


def build_daily_equity_curve(
    calendar: pd.DatetimeIndex,
    sized_trades: pd.DataFrame,
    start_equity: float,
    asset: str,
    leverage_cap: float,
) -> pd.DataFrame:
    daily = pd.DataFrame({"date": calendar})
    daily["asset"] = asset
    daily["leverage_cap"] = leverage_cap
    if sized_trades.empty:
        daily["pnl_usd"] = 0.0
        daily["equity"] = float(start_equity)
        return daily

    pnl_by_day = sized_trades.groupby("exit_date")["pnl_usd"].sum()
    daily["pnl_usd"] = daily["date"].map(pnl_by_day).fillna(0.0)
    daily["equity"] = float(start_equity) + daily["pnl_usd"].cumsum()
    daily["equity"] = daily["equity"].clip(lower=0.0)
    return daily


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())


def summarize_metrics(
    sized_trades: pd.DataFrame,
    daily_curve: pd.DataFrame,
    start_equity: float,
    label: str,
    leverage_cap: float,
) -> dict:
    total_trades = int(len(sized_trades))
    gross_profit = float(sized_trades.loc[sized_trades["pnl_usd"] > 0, "pnl_usd"].sum()) if total_trades else 0.0
    gross_loss = float(sized_trades.loc[sized_trades["pnl_usd"] < 0, "pnl_usd"].sum()) if total_trades else 0.0
    win_rate = float((sized_trades["pnl_usd"] > 0).mean()) if total_trades else 0.0
    ending_equity = float(daily_curve["equity"].iloc[-1]) if not daily_curve.empty else float(start_equity)
    total_return = (ending_equity / float(start_equity)) - 1.0
    years = max((daily_curve["date"].iloc[-1] - daily_curve["date"].iloc[0]).days / 365.25, 1.0 / 365.25)
    cagr = (ending_equity / float(start_equity)) ** (1.0 / years) - 1.0 if start_equity > 0 and ending_equity > 0 else -1.0
    max_dd = compute_max_drawdown(daily_curve["equity"])
    rets = daily_curve["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty or float(rets.std(ddof=0)) == 0.0:
        sharpe = 0.0
    else:
        sharpe = float((rets.mean() / rets.std(ddof=0)) * np.sqrt(365.0))
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else (math.inf if gross_profit > 0 else 0.0)

    return {
        "label": label,
        "leverage_cap": leverage_cap,
        "start_equity_usd": float(start_equity),
        "ending_equity_usd": ending_equity,
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe_365d": sharpe,
        "profit_factor": float(pf),
        "trade_count": total_trades,
        "win_rate_pct": win_rate * 100.0,
        "avg_used_leverage": float(sized_trades["used_leverage"].mean()) if total_trades else 0.0,
        "max_used_leverage": float(sized_trades["used_leverage"].max()) if total_trades else 0.0,
        "avg_hold_days": float(sized_trades["hold_days_est"].mean()) if total_trades else 0.0,
        "blow_up_count": int(sized_trades["blown_up"].sum()) if total_trades else 0,
    }


def setup_performance(sized_trades: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if sized_trades.empty:
        return pd.DataFrame()
    grouped = (
        sized_trades.groupby(["asset", "setup", "side", "trend_bucket"], observed=True)
        .agg(
            trades=("pnl_usd", "size"),
            win_rate_pct=("pnl_usd", lambda s: float((s > 0).mean() * 100.0)),
            total_pnl_usd=("pnl_usd", "sum"),
            avg_pnl_usd=("pnl_usd", "mean"),
            total_return_pct=("return_on_equity_pct", lambda s: float(((1.0 + s).prod() - 1.0) * 100.0)),
            avg_hold_days=("hold_days_est", "mean"),
        )
        .reset_index()
        .sort_values(["total_pnl_usd", "win_rate_pct", "trades"], ascending=[False, False, False])
    )
    grouped["tier"] = "C"
    grouped.loc[(grouped["trades"] >= min_trades) & (grouped["total_pnl_usd"] > 250.0) & (grouped["win_rate_pct"] >= 40.0), "tier"] = "A+"
    grouped.loc[(grouped["tier"] == "C") & (grouped["trades"] >= min_trades) & (grouped["total_pnl_usd"] > 0.0), "tier"] = "B"
    return grouped


def save_combined_equity_plot(portfolio_curves: pd.DataFrame, out_path: Path) -> None:
    if portfolio_curves.empty:
        return
    plt.figure(figsize=(13, 7))
    for lev_cap, sub in portfolio_curves.groupby("leverage_cap"):
        plt.plot(sub["date"], sub["equity"], linewidth=2.0, label=f"{int(lev_cap)}x cap")
    plt.title("Stop-Confirm Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_asset_equity_plot(asset_curves: pd.DataFrame, leverage_cap: float, out_path: Path) -> None:
    subset = asset_curves[asset_curves["leverage_cap"] == leverage_cap].copy()
    if subset.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, asset in zip(axes.flat, ASSET_ORDER):
        sub = subset[subset["asset"] == asset].sort_values("date")
        ax.plot(sub["date"], sub["equity"], linewidth=1.8, color="#0f766e")
        ax.set_title(asset)
        ax.grid(alpha=0.20)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Per-Asset Equity Curves ({int(leverage_cap)}x cap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_end_equity_bar(asset_metrics: pd.DataFrame, portfolio_metrics: pd.DataFrame, out_path: Path) -> None:
    if asset_metrics.empty:
        return
    plot_rows = asset_metrics[["label", "leverage_cap", "ending_equity_usd"]].copy()
    plot_rows["label"] = plot_rows["label"].astype(str)
    combined_rows = portfolio_metrics[["label", "leverage_cap", "ending_equity_usd"]].copy()
    combined_rows["label"] = combined_rows["label"].astype(str)
    plot_rows = pd.concat([plot_rows, combined_rows], ignore_index=True)

    labels = plot_rows["label"].unique().tolist()
    x = np.arange(len(labels))
    width = 0.34

    vals20 = plot_rows[plot_rows["leverage_cap"] == 20.0].set_index("label")["ending_equity_usd"].reindex(labels).fillna(0.0)
    vals30 = plot_rows[plot_rows["leverage_cap"] == 30.0].set_index("label")["ending_equity_usd"].reindex(labels).fillna(0.0)

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2.0, vals20, width=width, label="20x cap", color="#2563eb")
    plt.bar(x + width / 2.0, vals30, width=width, label="30x cap", color="#dc2626")
    plt.xticks(x, labels)
    plt.ylabel("Ending Equity (USD)")
    plt.title("Ending Equity by Asset and Combined Portfolio")
    plt.grid(axis="y", alpha=0.20)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_setup_plot(setup_metrics: pd.DataFrame, leverage_cap: float, out_path: Path) -> None:
    subset = setup_metrics[setup_metrics["leverage_cap"] == leverage_cap].copy()
    if subset.empty:
        return
    top = subset.sort_values(["total_pnl_usd", "win_rate_pct"], ascending=[False, False]).head(12).copy()
    top["label"] = top["asset"] + " | " + top["setup"] + " " + top["side"] + " " + top["trend_bucket"]
    colors = top["tier"].map({"A+": "#16a34a", "B": "#2563eb", "C": "#94a3b8"}).fillna("#94a3b8")

    fig, ax1 = plt.subplots(figsize=(12, 7))
    y = np.arange(len(top))
    ax1.barh(y, top["total_pnl_usd"], color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels(top["label"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Total PnL (USD)")
    ax1.set_title(f"Top Setup Buckets ({int(leverage_cap)}x cap)")
    ax1.grid(axis="x", alpha=0.18)

    ax2 = ax1.twiny()
    ax2.scatter(top["win_rate_pct"], y, color="#111827", s=36, zorder=5)
    ax2.set_xlabel("Win Rate (%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_path: Path,
    signal_dir: Path,
    cfg: SimulationConfig,
    asset_metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    setup_tables: dict[float, pd.DataFrame],
) -> None:
    pivot_assets = (
        asset_metrics.pivot(index="label", columns="leverage_cap", values=["ending_equity_usd", "total_return_pct", "win_rate_pct", "max_drawdown_pct"])
        .sort_index()
    )
    pivot_port = portfolio_metrics.pivot(index="label", columns="leverage_cap", values=["ending_equity_usd", "total_return_pct", "win_rate_pct", "max_drawdown_pct"])

    lines = [
        "# Tori Stop-Confirm Leveraged Portfolio Report",
        "",
        "## Assumptions",
        f"- Signal source: `{signal_dir}`",
        f"- Assets: {', '.join(ASSET_ORDER)}",
        f"- Starting capital: `${cfg.start_equity_per_asset:,.2f}` per asset (`${cfg.start_equity_per_asset * len(ASSET_ORDER):,.2f}` total)",
        f"- Risk budget: `{cfg.risk_pct * 100:.1f}%` per trade",
        "- Position size: `min(leverage cap, risk_budget / stop_distance_pct)`",
        "- Accounts are simulated as isolated per asset, then summed into a combined portfolio.",
        "- No slippage, financing, or commissions are included.",
        "",
        "## Asset Metrics",
        pivot_assets.round(2).to_markdown(),
        "",
        "## Combined Portfolio",
        pivot_port.round(2).to_markdown(),
        "",
        "## Charts",
        "- `combined_equity_curves.png`",
        "- `asset_equity_curves_20x.png`",
        "- `asset_equity_curves_30x.png`",
        "- `ending_equity_bar.png`",
        "- `setup_quality_20x.png`",
        "- `setup_quality_30x.png`",
        "",
    ]

    for lev_cap, table in setup_tables.items():
        lines.extend(
            [
                f"## Top Setups ({int(lev_cap)}x cap)",
                table.head(12).round(2).to_markdown(index=False),
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    signal_dir = Path(args.signal_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = [asset.strip().upper() for asset in args.assets.split(",") if asset.strip()]
    risk_pct = normalize_risk_pct(float(args.risk_pct))
    leverage_caps = parse_csv_list(args.leverage_caps, float)
    calendar = master_calendar(signal_dir)
    trade_logs = load_trade_logs(signal_dir, assets)

    asset_metric_rows: list[dict] = []
    portfolio_metric_rows: list[dict] = []
    asset_curve_rows: list[pd.DataFrame] = []
    setup_tables: dict[float, pd.DataFrame] = {}

    for leverage_cap in leverage_caps:
        sim_cfg = SimulationConfig(
            start_equity_per_asset=float(args.start_equity_per_asset),
            risk_pct=float(risk_pct),
            leverage_cap=float(leverage_cap),
        )

        sized_frames: list[pd.DataFrame] = []
        curve_frames: list[pd.DataFrame] = []

        for asset in assets:
            sized = simulate_asset_trades(trade_logs[asset], sim_cfg)
            sized["leverage_cap"] = leverage_cap
            sized_frames.append(sized)

            daily_curve = build_daily_equity_curve(calendar, sized, sim_cfg.start_equity_per_asset, asset, leverage_cap)
            curve_frames.append(daily_curve)
            asset_curve_rows.append(daily_curve)

            asset_metric_rows.append(
                summarize_metrics(
                    sized_trades=sized,
                    daily_curve=daily_curve,
                    start_equity=sim_cfg.start_equity_per_asset,
                    label=asset,
                    leverage_cap=leverage_cap,
                )
            )

        all_sized = pd.concat(sized_frames, ignore_index=True)
        setup_table = setup_performance(all_sized, min_trades=int(args.min_setup_trades))
        setup_table["leverage_cap"] = leverage_cap
        setup_tables[leverage_cap] = setup_table

        combined = pd.concat(curve_frames, ignore_index=True)
        portfolio_curve = (
            combined.groupby(["date", "leverage_cap"], as_index=False)["equity"].sum().assign(asset="COMBINED")
        )
        asset_curve_rows.append(portfolio_curve.rename(columns={"asset": "asset"}))
        portfolio_metric_rows.append(
            summarize_metrics(
                sized_trades=all_sized,
                daily_curve=portfolio_curve,
                start_equity=sim_cfg.start_equity_per_asset * len(assets),
                label="COMBINED",
                leverage_cap=leverage_cap,
            )
        )

        all_sized.to_csv(out_dir / f"trade_log_sized_{int(leverage_cap)}x.csv", index=False)
        setup_table.to_csv(out_dir / f"setup_metrics_{int(leverage_cap)}x.csv", index=False)
        portfolio_curve.to_csv(out_dir / f"combined_equity_curve_{int(leverage_cap)}x.csv", index=False)

    asset_metrics = pd.DataFrame(asset_metric_rows).sort_values(["label", "leverage_cap"])
    portfolio_metrics = pd.DataFrame(portfolio_metric_rows).sort_values("leverage_cap")
    asset_curves = pd.concat(asset_curve_rows, ignore_index=True)

    asset_metrics.to_csv(out_dir / "asset_metrics.csv", index=False)
    portfolio_metrics.to_csv(out_dir / "portfolio_metrics.csv", index=False)
    asset_curves.to_csv(out_dir / "all_equity_curves.csv", index=False)

    combined_curves = asset_curves[asset_curves["asset"] == "COMBINED"].copy()
    save_combined_equity_plot(combined_curves, out_dir / "combined_equity_curves.png")
    for leverage_cap in leverage_caps:
        save_asset_equity_plot(asset_curves[asset_curves["asset"] != "COMBINED"], leverage_cap, out_dir / f"asset_equity_curves_{int(leverage_cap)}x.png")
        save_setup_plot(setup_tables[leverage_cap], leverage_cap, out_dir / f"setup_quality_{int(leverage_cap)}x.png")
    save_end_equity_bar(asset_metrics, portfolio_metrics, out_dir / "ending_equity_bar.png")

    write_report(
        out_path=out_dir / "report.md",
        signal_dir=signal_dir,
        cfg=SimulationConfig(
            start_equity_per_asset=float(args.start_equity_per_asset),
            risk_pct=float(risk_pct),
            leverage_cap=max(leverage_caps),
        ),
        asset_metrics=asset_metrics,
        portfolio_metrics=portfolio_metrics,
        setup_tables=setup_tables,
    )

    print(f"Saved leveraged portfolio report to {out_dir}")


if __name__ == "__main__":
    main()
