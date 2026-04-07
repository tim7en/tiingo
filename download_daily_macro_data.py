#!/usr/bin/env python3
"""Download daily macro market data into the local cache.

Series:
- DXY: Yahoo Finance chart data for DX-Y.NYB (ICE U.S. Dollar Index)
- US2Y/US10Y/US30Y: FRED Treasury constant maturity yields
- WTI: FRED WTI Cushing crude oil spot price
- CPI: FRED CPIAUCSL with derived month-over-month and year-over-year inflation
- UNRATE: FRED civilian unemployment rate
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from typing import Any

import pandas as pd
import requests


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_START = "1999-01-01"

FRED_SERIES = {
    "US2Y": {
        "series_id": "DGS2",
        "name": "2-Year Treasury Constant Maturity Rate",
        "units": "percent",
        "combined_col": "us_2y_yield",
    },
    "US10Y": {
        "series_id": "DGS10",
        "name": "10-Year Treasury Constant Maturity Rate",
        "units": "percent",
        "combined_col": "us_10y_yield",
    },
    "US30Y": {
        "series_id": "DGS30",
        "name": "30-Year Treasury Constant Maturity Rate",
        "units": "percent",
        "combined_col": "us_30y_yield",
    },
    "WTI": {
        "series_id": "DCOILWTICO",
        "name": "WTI Crude Oil Spot Price, Cushing, OK",
        "units": "USD per barrel",
        "combined_col": "wti_usd_per_bbl",
    },
    "CPI": {
        "series_id": "CPIAUCSL",
        "name": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
        "units": "index 1982-1984=100, seasonally adjusted",
        "combined_col": "cpi_all_items_index",
        "derived": "cpi_inflation",
        "derived_combined_cols": {
            "cpi_mom_pct": "cpi_mom_pct",
            "cpi_yoy_pct": "cpi_yoy_pct",
        },
    },
    "UNRATE": {
        "series_id": "UNRATE",
        "name": "Unemployment Rate",
        "units": "percent, seasonally adjusted",
        "combined_col": "unemployment_rate_pct",
    },
}

DXY_TICKER = "DX-Y.NYB"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_CHART_URL = f"https://query1.finance.yahoo.com/v8/finance/chart/{DXY_TICKER}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; daily-macro-data-downloader/1.0)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="Inclusive end date, YYYY-MM-DD",
    )
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Output cache directory")
    return parser.parse_args()


def checked_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def request_text(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    retries: int = 6,
    timeout: int = 45,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, headers=HEADERS, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.text
        except Exception as exc:  # requests raises several transport-specific exception types.
            last_error = exc
            if attempt == retries - 1:
                break
            sleep_seconds = min(2.0 * (attempt + 1), 20.0)
            print(f"  Retry {attempt + 1}/{retries - 1} after {exc}; sleeping {sleep_seconds:.0f}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def fetch_fred_series(
    session: requests.Session,
    label: str,
    config: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    series_id = config["series_id"]
    text = request_text(session, FRED_CSV_URL, {"id": series_id})
    df = pd.read_csv(StringIO(text))
    if "observation_date" not in df.columns or series_id not in df.columns:
        raise RuntimeError(f"Unexpected FRED CSV columns for {series_id}: {list(df.columns)}")

    df = df.rename(columns={"observation_date": "date", series_id: "value"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    if config.get("derived") == "cpi_inflation":
        df["cpi_mom_pct"] = df["value"].pct_change(periods=1, fill_method=None) * 100.0
        df["cpi_yoy_pct"] = df["value"].pct_change(periods=12, fill_method=None) * 100.0
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    df["series_id"] = series_id
    df["label"] = label
    df["name"] = config["name"]
    df["units"] = config["units"]
    output_cols = ["date", "value"]
    output_cols.extend((config.get("derived_combined_cols") or {}).keys())
    output_cols.extend(["series_id", "label", "name", "units"])
    return df[output_cols].reset_index(drop=True)


def fetch_dxy(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    # Yahoo's period2 is exclusive, so request the day after the inclusive end date.
    yahoo_end = end_date + timedelta(days=1)
    end_dt = datetime(yahoo_end.year, yahoo_end.month, yahoo_end.day, tzinfo=timezone.utc)
    text = request_text(
        session,
        YAHOO_CHART_URL,
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        },
    )
    payload = json.loads(text)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {DXY_TICKER}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"Yahoo chart returned no data for {DXY_TICKER}")

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose")

    df = pd.DataFrame({"timestamp": timestamps})
    for col in ["open", "high", "low", "close", "volume"]:
        values = quote.get(col)
        df[col] = values if values is not None else pd.NA
    df["adj_close"] = adjclose if adjclose is not None else pd.NA
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
    df["ticker"] = DXY_TICKER
    df["label"] = "DXY"
    df["source"] = "Yahoo Finance chart API"
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).copy()
    return df[
        ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker", "label", "source"]
    ].reset_index(drop=True)


def save_table(df: pd.DataFrame, path_without_ext: str) -> None:
    df.to_csv(f"{path_without_ext}.csv", index=False)
    df.to_parquet(f"{path_without_ext}.parquet", index=False)


def print_range(label: str, df: pd.DataFrame, value_col: str) -> None:
    first = df["date"].min()
    last = df["date"].max()
    non_null = int(df[value_col].notna().sum())
    print(f"  {label}: {len(df):,} rows, {non_null:,} non-null, {first} -> {last}")


def main() -> None:
    args = parse_args()
    start_date = checked_date(args.start)
    end_date = checked_date(args.end)
    if end_date < start_date:
        raise SystemExit(f"--end {end_date} is before --start {start_date}")

    os.makedirs(args.cache_dir, exist_ok=True)
    session = requests.Session()

    print(f"Downloading daily macro data: {start_date} -> {end_date}")
    dxy = fetch_dxy(session, start_date, end_date)
    save_table(dxy, os.path.join(args.cache_dir, "DXY_daily"))
    print_range("DXY", dxy, "close")

    combined = dxy[["date", "close"]].rename(columns={"close": "dxy_close"})
    series_metadata: dict[str, Any] = {
        "DXY": {
            "ticker": DXY_TICKER,
            "source": "Yahoo Finance chart API",
            "source_url": YAHOO_CHART_URL,
            "combined_col": "dxy_close",
            "units": "index level",
        }
    }

    for label, config in FRED_SERIES.items():
        fred_df = fetch_fred_series(session, label, config, start_date, end_date)
        save_table(fred_df, os.path.join(args.cache_dir, f"{label}_daily"))
        print_range(label, fred_df, "value")
        derived_cols = config.get("derived_combined_cols") or {}
        fred_combined_cols = ["date", "value", *derived_cols.keys()]
        fred_rename = {"value": config["combined_col"], **derived_cols}
        combined = combined.merge(
            fred_df[fred_combined_cols].rename(columns=fred_rename),
            on="date",
            how="outer",
        )
        series_metadata[label] = {
            **config,
            "source": "FRED",
            "source_url": f"https://fred.stlouisfed.org/series/{config['series_id']}",
        }

    combined = combined.sort_values("date").reset_index(drop=True)
    value_columns = ["dxy_close"]
    for config in FRED_SERIES.values():
        value_columns.append(config["combined_col"])
        value_columns.extend((config.get("derived_combined_cols") or {}).values())
    combined = combined.dropna(subset=value_columns, how="all").reset_index(drop=True)
    save_table(combined, os.path.join(args.cache_dir, "macro_daily_1999"))
    print_range("combined", combined, "dxy_close")

    metadata = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "files": {
            "combined_csv": os.path.join(args.cache_dir, "macro_daily_1999.csv"),
            "combined_parquet": os.path.join(args.cache_dir, "macro_daily_1999.parquet"),
        },
        "series": series_metadata,
        "notes": [
            "Combined table uses an outer join on date and does not forward-fill missing observations.",
            "Rows with no values in any combined data column are dropped from the combined table.",
            "Treasury yields, CPI inflation, and unemployment are percent; WTI is USD per barrel.",
            "CPI and unemployment are monthly FRED observations and are not forward-filled.",
            "CPI month-over-month and year-over-year inflation percentages are computed from CPIAUCSL.",
        ],
    }
    with open(os.path.join(args.cache_dir, "macro_daily_1999_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


if __name__ == "__main__":
    main()
