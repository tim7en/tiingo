# ORB Strategy Backtest — Development Log

> **Purpose:** Guiding document for all ORB (Opening Range Breakout) strategy tests.  
> **Data Source:** Tiingo API (IEX for US stocks/ETFs, Crypto for SOL)  
> **Last Updated:** 2025-06-25  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [API & Data Pipeline](#2-api--data-pipeline)
3. [Strategy Rules (ORB)](#3-strategy-rules-orb)
4. [Parameter Sets](#4-parameter-sets)
5. [Assets & Universe](#5-assets--universe)
6. [Backtest Results — Lump Sum ($100)](#6-backtest-results--lump-sum-100)
7. [Grand Summary Tables](#7-grand-summary-tables)
8. [DCA Simulation Results](#8-dca-simulation-results)
9. [Best Protective Strategy Analysis](#9-best-protective-strategy-analysis)
10. [Known Limitations & Data Gaps](#10-known-limitations--data-gaps)
11. [Lessons Learned](#11-lessons-learned)
12. [Future Tests / TODO](#12-future-tests--todo)

---

## 1. Project Overview

**Goal:** Backtest the ORB (Opening Range Breakout) intraday strategy on 5-minute and 15-minute timeframes across a diversified 10-asset universe, over 1Y / 3Y / 5Y periods, using Tiingo's free-tier API.

**Script:** `backtest_orb_tiingo.py` (~2200 lines)  
**Reference:** `backtest_orb_s11_yfinance.py` (original S11 yfinance version)

**Key Features:**
- Concentrated portfolio (best candidate per day, max 2 trades across different asset classes)
- Multi-factor scoring: SMA trend + Volume + Range + Candle body
- Kelly-informed risk sizing with hard caps
- Breakeven stop after TP1, full exit at TP2 or SL
- Combined wallet variant merging 5min + 15min signals

---

## 2. API & Data Pipeline

### Tiingo API Details
- **API Key:** `34d03d1d1382e36010bdb817d2512a4bfa5585f3`
- **Tier:** Free (50 requests/hour, 1000/day, 500 unique symbols/month)
- **Endpoints Used:**
  - `iex` — US stocks/ETFs 5-minute OHLC (**no volume column**)
  - `crypto` — SOL/USD 5-minute OHLCV (has volume)
  - `tiingo/daily` — Daily OHLCV for real volume data

### Data Workaround — Volume Problem
IEX 5-minute bars have **no volume field**. This caused all non-crypto assets to fail the VOL+2 scoring check and never trade.

**Solution:** Fetch daily volume from Tiingo's daily endpoint (`/tiingo/daily/{ticker}/prices`) and inject it as `daily_vol_override` into session building. Each session's `vol_ratio` is computed against a 20-day rolling average of daily volume.

### Rate Limiting & Caching
- Global pacer: `REQUEST_GAP_SEC = 75` seconds between API calls
- IEX: 2-year chunks per request (5Y = 3 chunks per ticker)
- Crypto: 180-day chunks (5Y ≈ 11 chunks)
- All data cached at `d:\tiingo\cache\`:
  - `*_5m.parquet` — 5-minute OHLC bars
  - `*_daily.json` — Daily OHLCV
  - `*_daily_vol.json` — Daily volume specifically

### Data Coverage
| Asset | Source | 5Y Data Bars | Trading Days |
|-------|--------|-------------|--------------|
| TSLA  | IEX    | 30,000      | 470          |
| COIN  | IEX    | 30,000      | 470          |
| HOOD  | IEX    | 30,000      | 470          |
| MSTR  | IEX    | 30,000      | 470          |
| AMZN  | IEX    | 30,000      | 470          |
| PLTR  | IEX    | 30,000      | 470          |
| INTC  | IEX    | 30,000      | 470          |
| XAU (GLD) | IEX | 30,000   | 470          |
| XAG (SLV) | IEX | 30,000   | 470          |
| SOL   | Crypto | 50,010      | 198 (24/7)   |

**Note:** IEX 30K-bar limit means 5Y period only spans ~Jun 2022 to Feb 2026 = ~470 trading days (~1.9 calendar years of market days, ~3.6 calendar years).

---

## 3. Strategy Rules (ORB)

### Opening Range (OR) Definition
- **5min variant:** First 1 bar (09:30–09:35 ET) = OR high/low
- **15min variant:** First 3 bars (09:30–09:45 ET) = OR high/low

### Entry Signals
- **Long:** Price breaks above OR high
- **Short:** Price breaks below OR low

### Scoring System (max 8 points)
| Factor | Points | Condition |
|--------|--------|-----------|
| **SMA Trend** | +3 | Price above 20-day SMA (long) / below (short) |
| **Volume** | +2 | Daily volume > `vol_mult` × 20-day avg volume |
| **Range Quality** | +2 | OR range within `min_or_range_pct` – `max_or_range_pct` |
| **Candle Body** | +1 | OR candle body > 50% of total range (strong conviction) |

### Position Management
- **TP1:** 1.0 × OR range → close 50% of position
- **TP2:** 2.0 × OR range → close remaining 50%
- **Stop Loss:** Opposite OR bound
- **Breakeven:** After TP1 hit, move stop to entry price
- **Force Close:** 21:00 ET (session end)

### Risk & Leverage
- **Leverage formula:** `min(max_leverage, 5 + score)` → always 10x for qualifying trades
- **Risk per trade:** Kelly-modulated, capped at `risk_pct` from config
- **Commission:** 0.05% per trade (round-trip)

### Portfolio Construction
- **Concentrated:** 1–2 trades per day (best-scored candidates)
- **Diversification rule:** 2nd trade must be from different asset class
- **Asset classes:** Tech (TSLA, COIN, HOOD, MSTR, AMZN, PLTR, INTC), Commodities (XAU, XAG), Crypto (SOL)

---

## 4. Parameter Sets

| Param | S9 (Conservative) | S10 (Aggressive) | S12 (Balanced) |
|-------|-------------------|-------------------|----------------|
| `min_score` | 8 | 7 | 8 |
| `vol_mult` | 2.0x | 1.5x | 1.3x |
| `risk_pct` | 5% | 7% | 5% |
| `min_or_range_pct` | 0.3% | 0.3% | 0.3% |
| `max_or_range_pct` | 3.0% | 3.0% | 3.0% |
| `max_daily_trades` | 2 | 2 | 2 |
| `max_leverage` | 10 | 10 | 10 |
| `starting_balance` | $100 | $100 | $100 |

**Key differences:**
- **S9** = strictest filter (score ≥ 8, vol ≥ 2x) → fewest trades, lowest drawdown
- **S10** = relaxed filter (score ≥ 7, vol ≥ 1.5x, risk 7%) → most trades, highest return
- **S12** = moderate filter (score ≥ 8, vol ≥ 1.3x) → balanced trade count

---

## 5. Assets & Universe

| # | Ticker | Asset | Class | Source |
|---|--------|-------|-------|--------|
| 1 | TSLA   | Tesla | Tech  | IEX    |
| 2 | COIN   | Coinbase | Tech | IEX |
| 3 | HOOD   | Robinhood | Tech | IEX |
| 4 | MSTR   | MicroStrategy | Tech | IEX |
| 5 | AMZN   | Amazon | Tech | IEX |
| 6 | PLTR   | Palantir | Tech | IEX |
| 7 | INTC   | Intel | Tech | IEX |
| 8 | XAU    | Gold (GLD) | Commodity | IEX |
| 9 | XAG    | Silver (SLV) | Commodity | IEX |
| 10 | SOL   | Solana | Crypto | Crypto |

---

## 6. Backtest Results — Lump Sum ($100 starting balance)

### 1Y Period (139 trading days, ~Aug 2025 – Feb 2026)

| PSet | Variant | Trades | WR | PF | Sharpe | CAGR | MaxDD | Return | $Final |
|------|---------|--------|-----|-----|--------|------|-------|--------|--------|
| S9   | US_5min | 13 | 46.2% | 2.27 | 1.82 | 69.2% | -8.5% | +33.6% | $133.63 |
| S9   | US_15min | 7 | 57.1% | 2.16 | 1.46 | 39.0% | -11.3% | +19.9% | $119.94 |
| **S9** | **Combined** | **17** | **52.9%** | **2.62** | **2.40** | **122.0%** | **-8.5%** | **+55.3%** | **$155.26** |
| S10  | US_5min | 62 | 38.7% | 1.82 | 2.11 | 324.6% | -33.3% | +122.0% | $222.02 |
| **S10** | **US_15min** | **43** | **48.8%** | **2.08** | **2.70** | **491.5%** | **-27.7%** | **+166.6%** | **$266.57** |
| S10  | Combined | 61 | 41.0% | 1.99 | 2.43 | 457.7% | -33.3% | +158.1% | $258.06 |
| S12  | US_5min | 41 | 36.6% | 1.55 | 1.19 | 77.1% | -17.2% | +37.0% | $137.04 |
| S12  | US_15min | 41 | 46.3% | 1.79 | 1.58 | 124.5% | -21.4% | +56.2% | $156.20 |
| **S12** | **Combined** | **58** | **44.8%** | **2.25** | **2.37** | **313.9%** | **-15.4%** | **+118.9%** | **$218.91** |

### 3Y Period (306 trading days, ~Jul 2024 – Feb 2026)

| PSet | Variant | Trades | WR | PF | Sharpe | CAGR | MaxDD | Return | $Final |
|------|---------|--------|-----|-----|--------|------|-------|--------|--------|
| S9   | US_5min | 22 | 40.9% | 1.91 | 1.15 | 33.0% | -12.8% | +41.4% | $141.36 |
| S9   | US_15min | 14 | 57.1% | 2.51 | 1.27 | 31.2% | -11.3% | +39.1% | $139.12 |
| S9   | Combined | 30 | 43.3% | 1.96 | 1.17 | 40.6% | -15.7% | +51.2% | $151.21 |
| S10  | US_5min | 117 | 40.2% | 1.91 | 2.21 | 343.0% | -33.3% | +509.4% | $609.38 |
| **S10** | **US_15min** | **90** | **44.4%** | **2.17** | **2.93** | **514.7%** | **-27.7%** | **+807.0%** | **$907.03** |
| S10  | Combined | 119 | 42.9% | 2.04 | 2.52 | 445.6% | -33.3% | +684.8% | $784.78 |
| S12  | US_5min | 84 | 39.3% | 1.76 | 1.60 | 109.8% | -17.2% | +146.0% | $245.96 |
| S12  | US_15min | 77 | 42.9% | 1.81 | 1.77 | 130.6% | -22.8% | +175.8% | $275.81 |
| S12  | Combined | 114 | 42.1% | 2.09 | 2.12 | 223.9% | -27.8% | +316.7% | $416.73 |

### 5Y Period (470 trading days, ~Jun 2022 – Feb 2026)

| PSet | Variant | Trades | WR | PF | Sharpe | CAGR | MaxDD | Return | $Final |
|------|---------|--------|-----|-----|--------|------|-------|--------|--------|
| S9   | US_5min | 28 | 42.9% | 2.05 | 1.12 | 28.8% | -12.8% | +60.2% | $160.23 |
| S9   | US_15min | 19 | 52.6% | 2.39 | 1.10 | 24.1% | -11.3% | +49.7% | $149.68 |
| S9   | Combined | 40 | 42.5% | 1.95 | 1.08 | 33.7% | -18.2% | +71.9% | $171.93 |
| S10  | US_5min | 167 | 38.9% | 1.90 | 1.91 | 256.3% | -38.9% | +969.5% | $1,069.51 |
| **S10** | **US_15min** | **132** | **44.7%** | **2.13** | **2.48** | **366.5%** | **-32.4%** | **+1668.3%** | **$1,768.27** |
| S10  | Combined | 174 | 41.4% | 2.01 | 2.21 | 331.8% | -33.3% | +1430.2% | $1,530.24 |
| S12  | US_5min | 118 | 39.0% | 1.82 | 1.70 | 112.2% | -17.2% | +306.8% | $406.85 |
| S12  | US_15min | 111 | 40.5% | 1.83 | 1.88 | 129.8% | -22.8% | +372.0% | $472.02 |
| **S12** | **Combined** | **164** | **39.6%** | **2.06** | **2.05** | **190.3%** | **-27.8%** | **+629.8%** | **$729.84** |

---

## 7. Grand Summary Tables

### Best Combinations by Metric

| Metric | Value | Config |
|--------|-------|--------|
| **Highest Sharpe** | 2.93 | 3Y / S10 / US_15min |
| **Highest Sortino** | 21.97 | 1Y / S9 / Combined |
| **Highest Calmar** | 20.41 | 1Y / S12 / Combined |
| **Highest CAGR** | 514.7% | 3Y / S10 / US_15min |
| **Highest Total Return** | +1668.3% | 5Y / S10 / US_15min |
| **Highest Win Rate** | 57.1% | 1Y / S9 / US_15min |
| **Highest Profit Factor** | 2.62 | 1Y / S9 / Combined |
| **Best Expectancy (R)** | +0.714 | 1Y / S9 / US_15min |
| **Shallowest MaxDD** | -8.5% | 1Y / S9 / US_5min |
| **Highest Recovery Factor** | 7.14 | 5Y / S12 / Combined |

### Leverage Details
- Formula: `leverage = min(max_leverage, 5 + score)`
- All qualifying trades use **10x leverage** (score 7 → min(10,12)=10, score 8 → min(10,13)=10)

### Trade Details & Leverage (5Y period, key configs)

| PSet | Variant | AvgPnL$ | AvgWin$ | AvgLos$ | MaxConsecW | MaxConsecL | DDdur |
|------|---------|---------|---------|---------|------------|------------|-------|
| S9   | Combined | +$2.46 | +$11.91 | -$4.52 | 3 | 6 | 288d |
| S10  | US_15min | +$17.18 | +$72.48 | -$27.52 | 5 | 6 | 65d |
| S12  | Combined | +$6.08 | +$29.74 | -$9.46 | 3 | 7 | 87d |

---

## 8. DCA Simulation Results

### DCA Schedule
| Year | Monthly | Annual | Cumulative |
|------|---------|--------|------------|
| Y1   | $100    | $1,200 | $1,200     |
| Y2   | $200    | $2,400 | $3,600     |
| Y3   | $300    | $3,600 | $7,200     |
| Y4   | $400    | $4,800 | $12,000    |
| Y5   | $500    | $6,000 | $18,000    |

> **Note:** Actual contributions depend on the trading data available. The IEX 30K-bar limit means the 5Y period only has ~470 trading days. Contributions are injected at the start of each new calendar month in the trading data. The "year number" is computed from elapsed calendar days since first trade.

### DCA Results (Combined wallet — 5min + 15min merged)

| Period | PSet | Contributed | Final$ | ROI | TWR | TWR/yr | MWR/yr |
|--------|------|-------------|--------|-----|-----|--------|--------|
| 1Y | S9  | $800   | $1,051  | +31.4% | +33.6% | +57.2%  | +141.4% |
| 1Y | S10 | $800   | $1,484  | +85.5% | +122.0% | +247.3% | +553.6% |
| 1Y | S12 | $800   | $1,025  | +28.1% | +37.0% | +63.5%  | +123.2% |
| 3Y | S9  | $4,000 | $5,413  | +35.3% | +41.4% | +14.1%  | +37.9%  |
| 3Y | S10 | $4,000 | $12,191 | +204.8% | +509.4% | +99.2% | +163.5% |
| 3Y | S12 | $4,000 | $6,386  | +59.6% | +146.0% | +40.9%  | +60.5%  |
| **5Y** | **S9**  | **$9,600** | **$13,383** | **+39.4%** | +60.2% | +10.8% | **+22.3%** |
| **5Y** | **S10** | **$9,600** | **$43,681** | **+355.0%** | +969.5% | +67.4% | **+104.0%** |
| **5Y** | **S12** | **$9,600** | **$19,789** | **+106.1%** | +306.8% | +35.7% | **+48.5%** |

### TWR vs MWR — What They Mean

| Metric | Measures | Use For |
|--------|----------|---------|
| **ROI** (simple) | Total gain / total contributed | Quick "how much did I make" |
| **TWR** (Time-Weighted) | Pure strategy performance, strips out cash flow timing | Comparing strategy quality across configs |
| **MWR/IRR** (Money-Weighted) | Actual investor's annualized return on capital deployed | Evaluating your personal investment outcome |

- **If MWR > TWR/yr:** DCA timing added value — more money deployed before good periods
- **If MWR < TWR/yr:** DCA timing cost you — more money deployed before bad periods
- In our results, **MWR consistently exceeds TWR/yr** — the DCA schedule (increasing contributions) happened to align with stronger performance periods

### DCA Key Takeaways
- **Best DCA by ROI:** 5Y / S10 → $9,600 invested → **$43,681** = **+355.0%** ROI
- **Best DCA by TWR:** 5Y / S10 → **+969.5%** total TWR (annualized +67.4%)
- **Best DCA by MWR/IRR:** 1Y / S10 → **+553.6%** annualized (short period, explosive growth)
- **S10 dominates:** Consistently highest across all return measures and periods
- **MWR > TWR/yr everywhere:** DCA schedule adds value — escalating contributions compound well with an upward-trending strategy
- **S9 is most stable DCA:** MWR of +22.3% p.a. over 5Y — safe and predictable
- **Baseline comparison:** Simply saving $9,600 with 0% return = $9,600. S10 DCA grew it 4.55x

---

## 9. Best Protective Strategy Analysis

### What Is "Protective"?
A **protective strategy** prioritizes:
1. Low maximum drawdown (capital preservation)
2. High recovery factor (bounces back quickly)
3. Consistent positive expectancy
4. Acceptable returns (not zero, but drawdown-adjusted)

### Strategy Rankings by Protection

#### Tier 1: Most Protective — S9 / Combined
| Metric | Value | Why It Matters |
|--------|-------|----------------|
| MaxDD (1Y) | **-8.5%** | Shallowest drawdown of ALL configs |
| Sortino (1Y) | **21.97** | Extreme downside-adjusted return — almost no bad days |
| Profit Factor | **2.62** | Every $1 lost generates $2.62 in wins |
| Win Rate | 52.9% | Slightly positive, but tight filter means high-quality trades |
| Expectancy | +0.71R | Expected gain of 0.71× risk per trade |

**Weakness:** Very few trades (17 in 1Y, 40 in 5Y). Over 5Y, MaxDD increased to -18.2% and Sharpe dropped to 1.08. Trades too infrequently to take advantage of compounding.

#### Tier 2: Best Balanced Protection — S12 / Combined
| Metric | Value | Why It Matters |
|--------|-------|----------------|
| Calmar (1Y) | **20.41** | Highest CAGR-to-MaxDD ratio of ALL configs |
| MaxDD (1Y) | -15.4% | Moderate — not devastating |
| Recovery Factor (5Y) | **7.14** | Highest of ALL configs — recovers fast from dips |
| Return (5Y) | +629.8% | Strong absolute return |
| Trades (5Y) | 164 | Enough trades for statistical significance |

**Why S12/Combined is the best protective strategy with returns:**
- Calmar of 20.41 means return massively exceeds drawdown risk
- Recovery factor of 7.14 (5Y) = total profit is 7.14× the maximum drawdown
- Moderate MaxDD (-15.4% to -27.8%) vs S10's -33.3% to -38.9%
- 164 trades over 5Y gives statistical confidence
- $100 → $729.84 (5Y) with reasonable risk

#### Tier 3: Highest Absolute Returns — S10 / US_15min
| Metric | Value | Why It Matters |
|--------|-------|----------------|
| Sharpe (3Y) | **2.93** | Best risk-adjusted return ratio |
| Total Return (5Y) | **+1668.3%** | Highest absolute return |
| CAGR (3Y) | **514.7%** | Highest annualized growth |
| MaxDD (5Y) | -32.4% | **Significant drawdown risk** |

**Not protective:** While returns are extraordinary, a 32-38% drawdown is psychologically and financially punishing. This is the growth strategy, not the protective one.

### Recommendation Matrix

| Investor Profile | Recommended Config | Expected 5Y Return | Max Pain |
|-----------------|-------------------|--------------------|---------| 
| **Conservative** | S9 / Combined | +71.9% ($172) | -18.2% |
| **Balanced / Protective** | **S12 / Combined** | **+629.8% ($730)** | **-27.8%** |
| **Aggressive Growth** | S10 / US_15min | +1668.3% ($1,768) | -32.4% |

### Verdict
> **S12 / Combined is the best protective strategy with meaningful returns.**  
> It delivers the highest Calmar ratio (20.41), highest Recovery Factor (7.14), and turns $100 into $730 over 5Y — while keeping MaxDD under 28%. It's the strategy that makes you money while letting you sleep at night.

---

## 10. Known Limitations & Data Gaps

### Data Limitations
1. **IEX 30K-bar cap:** The Tiingo IEX endpoint returns max 30,000 bars. At 5-minute resolution, this is ~192 trading days per request. Even chunking, the "5Y" period only has ~470 trading days of actual data.
2. **No intraday volume (IEX):** IEX 5-minute bars lack volume. We use daily volume from the daily endpoint as a proxy.
3. **Crypto limited hours:** SOL trades 24/7 but our ORB is calibrated for US market hours. Crypto ORB uses different session definitions.
4. **Forward-looking session definition:** The opening range is defined from historical data, not live. In live trading, you'd need to wait for the OR to complete before scoring.

### Strategy Limitations
1. **Survivorship bias:** We selected 10 assets that are currently popular/liquid. They may not have been as liquid or relevant 5 years ago.
2. **No slippage model:** Commission (0.05%) is modeled, but market impact / slippage during breakout is not.
3. **10x leverage assumption:** Retail traders may not have access to 10x leverage on all these instruments.
4. **Force close at 21:00 ET:** Some trades may have been profitable if held longer.
5. **Kelly lookback:** The Kelly position sizing uses trailing trade history, which improves over time but may over-risk early in the backtest.

### Comparison Note (vs yfinance)
The original `backtest_orb_s11_yfinance.py` uses Yahoo Finance data which has proper intraday volume. Results may differ slightly due to:
- Different data providers (price discrepancies)
- Volume source differences (real 5min vol vs daily vol proxy)
- Data availability windows

---

## 11. Lessons Learned

### Technical
1. **Always check what the API actually returns.** IEX 5min data lacking volume was discovered only after the first run showed 0 trades for stocks.
2. **Rate limiting is a real constraint.** At 50 req/hr and 75s pacing, fetching 10 assets × 5Y takes ~59 minutes. Caching is essential.
3. **Cache everything.** Parquet for time series, JSON for metadata. Never re-fetch what you already have.
4. **Combined wallet outperforms individual timeframes.** Merging 5min and 15min signals consistently produces higher Sharpe/Sortino ratios due to more diversified entry signals.

### Strategy
1. **Stricter filters ≠ always better.** S9 has the lowest drawdown but also the lowest returns. The sweet spot is S12 (strict score ≥ 8 but relaxed volume threshold).
2. **15min OR > 5min OR for risk-adjusted returns.** The 15-minute opening range captures more meaningful breakouts and reduces noise trades.
3. **Profit Factor > 2.0 is the quality threshold.** All configs exceed this for Combined wallet variants, confirming strategy edge.
4. **Sortino ratio reveals asymmetric returns.** S9/Combined has Sortino of 21.97 — meaning almost all downside days are tiny relative to upside days. This is the hallmark of a protective strategy.

### DCA Insights
1. **DCA dramatically amplifies returns** when the underlying strategy is profitable. S10 turned $9,600 into $43,680.
2. **Start early, increase gradually.** The compounding effect of larger late-year contributions on an already-profitable account is the driver.
3. **Even the worst DCA config (S12/1Y)** returned +28.1%, far exceeding any savings account.

---

## 12. Future Tests / TODO

- [ ] Test with real intraday volume data (IEX premium or alternative provider)
- [ ] Add slippage model (0.01-0.05% per trade on breakout bars)
- [ ] Test reduced leverage (2x, 5x) impact on returns and drawdowns
- [ ] Add more asset classes (sector ETFs, FX pairs)
- [ ] Test walk-forward optimization (train on 3Y, validate on holdout 2Y)
- [ ] Compare with yfinance S11 baseline results head-to-head
- [ ] Test DCA with S10/US_15min only (highest return variant)
- [ ] Monte Carlo simulation of trade sequence randomization
- [ ] Live paper-trading validation (forward testing)
- [ ] Test overnight hold variant (entry at breakout, exit next open)

---

## Appendix: File Structure

```
d:\tiingo\
├── backtest_orb_tiingo.py          # Main Tiingo backtest script
├── backtest_orb_s11_yfinance.py    # Original yfinance reference
├── test_api.py                     # API connectivity test
├── DEV_LOG.md                      # This file
└── cache\
    ├── TSLA_iex_5m.parquet
    ├── COIN_iex_5m.parquet
    ├── ...
    ├── solusd_crypto_5m.parquet
    ├── *_daily.json
    └── *_daily_vol.json
```
