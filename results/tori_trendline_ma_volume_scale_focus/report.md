# MA + Volume Risk Scaling Report

Assumptions:
- Strategy structure fixed to each asset's current best stop-confirm trendline setup.
- Risk overlay uses a lagged daily 200 MA plus a lagged normalized-volume trend.
- Volume trend is a 5-bar rolling mean of normalized volume, shifted by 1 full bar.
- Average setups per month uses the full sample window, including zero-trade months.
- Average monthly return is shown as both realized-month arithmetic mean and geometric monthly growth.

## Best overlays by asset
- XAU: aligned 1.50x, counter 1.00x, high-volume 1.50x, low-volume 1.00x, volume threshold 1.00, total return 496.55%, setups/month 0.42, geometric monthly return 1.63%
- XAG: aligned 1.50x, counter 1.00x, high-volume 1.50x, low-volume 1.00x, volume threshold 1.00, total return 434.23%, setups/month 0.82, geometric monthly return 1.53%
- COPPER: aligned 1.50x, counter 1.00x, high-volume 1.50x, low-volume 1.00x, volume threshold 1.10, total return 197.34%, setups/month 1.01, geometric monthly return 0.99%
- UUP: aligned 1.50x, counter 1.00x, high-volume 1.50x, low-volume 1.00x, volume threshold 1.00, total return 42.45%, setups/month 1.40, geometric monthly return 0.32%

## Equal-weight portfolio summary
- Total return: 319.92%
- Setups per month: 3.65
- Average realized monthly return: 1.63%
- Geometric monthly return: 1.30%