# Tori Stop-Confirm Leveraged Portfolio Report

## Assumptions
- Signal source: `results\tori_trendline_touchlines_stop_confirm_fullhist_v2`
- Assets: XAU, XAG, COPPER, UUP
- Starting capital: `$1,000.00` per asset (`$4,000.00` total)
- Risk budget: `5.0%` per trade
- Position size: `min(leverage cap, risk_budget / stop_distance_pct)`
- Accounts are simulated as isolated per asset, then summed into a combined portfolio.
- No slippage, financing, or commissions are included.

## Asset Metrics
| label   |   ('ending_equity_usd', 20.0) |   ('ending_equity_usd', 30.0) |   ('total_return_pct', 20.0) |   ('total_return_pct', 30.0) |   ('win_rate_pct', 20.0) |   ('win_rate_pct', 30.0) |   ('max_drawdown_pct', 20.0) |   ('max_drawdown_pct', 30.0) |
|:--------|------------------------------:|------------------------------:|-----------------------------:|-----------------------------:|-------------------------:|-------------------------:|-----------------------------:|-----------------------------:|
| COPPER  |                       3296.79 |                       3365.96 |                       229.68 |                       236.6  |                    36.61 |                    36.61 |                       -57.01 |                       -62.81 |
| UUP     |                      18168    |                      25097.3  |                      1716.8  |                      2409.73 |                    38.06 |                    38.06 |                       -54.61 |                       -55.79 |
| XAG     |                     162746    |                     186180    |                     16174.6  |                     18518    |                    45.35 |                    45.35 |                       -32.86 |                       -32.86 |
| XAU     |                      13543.7  |                      14885.1  |                      1254.37 |                      1388.51 |                    36.7  |                    36.7  |                       -83.82 |                       -85.55 |

## Combined Portfolio
| label    |   ('ending_equity_usd', 20.0) |   ('ending_equity_usd', 30.0) |   ('total_return_pct', 20.0) |   ('total_return_pct', 30.0) |   ('win_rate_pct', 20.0) |   ('win_rate_pct', 30.0) |   ('max_drawdown_pct', 20.0) |   ('max_drawdown_pct', 30.0) |
|:---------|------------------------------:|------------------------------:|-----------------------------:|-----------------------------:|-------------------------:|-------------------------:|-----------------------------:|-----------------------------:|
| COMBINED |                        197755 |                        229528 |                      4843.87 |                       5638.2 |                    38.74 |                    38.74 |                       -29.17 |                       -29.52 |

## Charts
- `combined_equity_curves.png`
- `asset_equity_curves_20x.png`
- `asset_equity_curves_30x.png`
- `ending_equity_bar.png`
- `setup_quality_20x.png`
- `setup_quality_30x.png`

## Top Setups (20x cap)
| asset   | setup   | side   | trend_bucket   |   trades |   win_rate_pct |   total_pnl_usd |   avg_pnl_usd |   total_return_pct |   avg_hold_days | tier   |   leverage_cap |
|:--------|:--------|:-------|:---------------|---------:|---------------:|----------------:|--------------:|-------------------:|----------------:|:-------|---------------:|
| XAG     | bounce  | long   | steady         |        9 |          44.44 |       147605    |      16400.5  |            1045.9  |           19.17 | A+     |             20 |
| UUP     | bounce  | long   | steady         |       19 |          52.63 |        13058.2  |        687.28 |             100.51 |            9.24 | A+     |             20 |
| XAU     | bounce  | long   | soft           |       17 |          23.53 |        10795.6  |        635.04 |             196.49 |           42.24 | B      |             20 |
| UUP     | bounce  | long   | soft           |       19 |          26.32 |         4664.33 |        245.49 |             330.76 |           29.26 | B      |             20 |
| XAG     | bounce  | long   | strong         |       15 |          33.33 |         4660.46 |        310.7  |              33.84 |            4.83 | B      |             20 |
| XAG     | bounce  | long   | soft           |       10 |          50    |         3914.99 |        391.5  |             200.87 |           51.75 | A+     |             20 |
| UUP     | bounce  | short  | strong         |       23 |          39.13 |         3786.95 |        164.65 |             193.59 |            5.43 | B      |             20 |
| XAG     | break   | long   | strong         |        7 |          71.43 |         3568.91 |        509.84 |              83.22 |            9.43 | C      |             20 |
| UUP     | bounce  | short  | steady         |       18 |          33.33 |         2765.54 |        153.64 |              25.05 |            6.25 | B      |             20 |
| XAG     | break   | long   | soft           |        4 |          50    |         2579.48 |        644.87 |              49.82 |           13.75 | C      |             20 |
| UUP     | break   | short  | steady         |        5 |          80    |         2386.9  |        477.38 |              14.47 |            6.2  | C      |             20 |
| XAU     | break   | short  | strong         |        6 |          50    |         1176.94 |        196.16 |              42.37 |            6.75 | C      |             20 |

## Top Setups (30x cap)
| asset   | setup   | side   | trend_bucket   |   trades |   win_rate_pct |   total_pnl_usd |   avg_pnl_usd |   total_return_pct |   avg_hold_days | tier   |   leverage_cap |
|:--------|:--------|:-------|:---------------|---------:|---------------:|----------------:|--------------:|-------------------:|----------------:|:-------|---------------:|
| XAG     | bounce  | long   | steady         |        9 |          44.44 |       169946    |      18882.9  |            1203.05 |           19.17 | A+     |             30 |
| UUP     | bounce  | long   | steady         |       19 |          52.63 |        19307.8  |       1016.2  |             127.56 |            9.24 | A+     |             30 |
| XAU     | bounce  | long   | soft           |       17 |          23.53 |        11802.8  |        694.28 |             205.12 |           42.24 | B      |             30 |
| UUP     | bounce  | short  | strong         |       23 |          39.13 |         5714.73 |        248.47 |             246.87 |            5.43 | B      |             30 |
| XAG     | bounce  | long   | strong         |       15 |          33.33 |         4615.42 |        307.69 |              33.84 |            4.83 | B      |             30 |
| XAG     | break   | long   | strong         |        7 |          71.43 |         4605.75 |        657.96 |              93.82 |            9.43 | C      |             30 |
| UUP     | bounce  | long   | soft           |       19 |          26.32 |         4498.33 |        236.75 |             337.63 |           29.26 | B      |             30 |
| XAG     | bounce  | long   | soft           |       10 |          50    |         3897.76 |        389.78 |             200.87 |           51.75 | A+     |             30 |
| UUP     | bounce  | short  | steady         |       18 |          33.33 |         3393.14 |        188.51 |              33.8  |            6.25 | B      |             30 |
| UUP     | break   | short  | steady         |        5 |          80    |         3198.07 |        639.61 |              14.47 |            6.2  | C      |             30 |
| XAG     | break   | long   | soft           |        4 |          50    |         2837.31 |        709.33 |              49.82 |           13.75 | C      |             30 |
| XAU     | break   | short  | strong         |        6 |          50    |         1259.45 |        209.91 |              42.37 |            6.75 | C      |             30 |
