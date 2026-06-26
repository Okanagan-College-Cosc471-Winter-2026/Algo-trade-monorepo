# May 2026 — Data Quality Report

**Prepared:** 2026-06-26
**Datasets:** `top29_15m_may2026.csv`, `macro_15m_may2026.csv`, `treasury_rates_may2026.csv`
**Coverage:** May 1–29 2026 · 15-minute bars · 29 stocks + 8 macro indicators + daily yield curve

---

## Quick Verdict

| Dataset | Status | One-line |
|---|---|---|
| Top-29 Stocks | **CLEAN** | Zero nulls, zero OHLC violations, perfect bar count |
| Macro ETFs | **MOSTLY CLEAN** | SHY unusable at 15-min resolution |
| Commodities | **USE WITH CARE** | Daily 17:00–18:00 ET settlement gap must be handled |
| Treasury Yields | **CLEAN** | Perfect daily series, zero issues |

---

## Section 1 — Top-29 Stocks (`top29_15m_may2026.csv`)

### Overview

| Metric | Value |
|---|---|
| Symbols | 29 |
| Trading days | 20 (May 25 = Memorial Day, correct) |
| Bars per symbol | 520 (26 bars/day × 20 days) |
| Total rows | 15,080 |
| Nulls | 0 |
| OHLC violations | 0 |
| Duplicate timestamps | 0 |
| Zero / negative prices | 0 |

### Issue 1 — Memorial Day Gap (Expected)

All 29 symbols have **zero bars on May 25 2026** (Memorial Day, US market holiday). This is correct. The calendar should permanently mark May 25 as a non-trading day — no forward-fill or imputation needed.

### Issue 2 — PSKY Stale Closes (Low-Liquidity Flag)

`PSKY` has **10 bars** across three dates where the close price was identical for 3+ consecutive 15-min periods:

| Date | Bars | Close | Note |
|---|---|---|---|
| May 13 | 3 bars (13:30–14:00) | $10.44 | Volume non-zero, high≠low |
| May 22 | 4 bars (14:15–14:45) | $10.48 | Volume non-zero, high≠low |
| May 27 | 3 bars (14:45–15:15) | $10.69 | Volume non-zero, high≠low |

This is genuine illiquid micro-cap behavior — the price simply did not tick at the close. It is **not a data error**. However, PSKY should be flagged in feature engineering; lag features may carry stale signal.

### Issue 3 — Large Gap Opens (Real Events, Keep as Signal)

**14 bars** with a single-bar return exceeding ±5%. All but one occur at the 09:30 open (overnight news gap). These are real events and should **not** be clipped.

| Symbol | Date / Time | Return | Context |
|---|---|---|---|
| INTC | May 05 09:30 | +11.1% | Earnings beat |
| MNST | May 08 09:30 | +12.1% | Earnings beat |
| INTC | May 08 12:45 | +8.0% | Mid-day continuation |
| INTC | May 15 09:30 | −6.8% | Reversal |
| INTC | May 20 09:30 | +6.0% | — |
| ADI | May 26 09:30 | +5.3% | — |
| GE | May 06 09:30 | +5.6% | — |
| DVN | May 06 09:30 | −5.9% | — |
| CVNA | May 06 09:30 | +5.1% | — |
| CVNA | May 13 09:30 | −5.1% | — |
| CVNA | May 22 09:30 | +5.2% | — |
| CF | May 06 09:30 | −5.1% | — |
| APA | May 07 09:30 | −6.0% | — |
| IT | May 13 09:30 | −5.6% | — |

---

## Section 2 — Macro Equity ETFs (`macro_15m_may2026.csv`)

Same trading hours as stocks (09:30–15:45 ET), same Memorial Day gap.

| Symbol | Description | Bars | Stale Bars | Issue |
|---|---|---|---|---|
| SPY | S&P 500 ETF | 520 | 0 | None |
| QQQ | Nasdaq-100 ETF | 520 | 0 | None |
| TLT | 20yr Treasury ETF | 520 | 3 | Isolated, benign |
| IEF | 7-10yr Treasury ETF | 520 | 3 | Isolated, benign |
| SHY | 2yr Treasury ETF | 520 | **101** | **Do not use as 15-min feature** |

### Issue — SHY Structurally Unusable at 15-Min Resolution

SHY trades at ~$82 and moves in ~1¢ increments. At 15-min granularity the price genuinely does not change on 101 out of 520 bars (19.4% of the series). This is not a data error — it is the nature of a near-cash short-duration bond ETF.

**Recommendation:** Drop SHY as a 15-min feature. Use the `year2` column from `treasury_rates_may2026.csv` as the 2-year rate proxy instead.

TLT and IEF have only 3 stale bars each (isolated sessions) — these are fine to keep.

---

## Section 3 — Commodities (`macro_15m_may2026.csv`)

Gold, Silver, and Crude Oil trade near 24-hours on CME/COMEX with a daily settlement break.

| Symbol | Description | Bars | Zero-Vol Bars | Intraday Gaps | Spike |
|---|---|---|---|---|---|
| GCUSD | Gold ($/oz) | 1,935 | 4 | 16 | None |
| SIUSD | Silver ($/oz) | 1,932 | 5 | 16 | None |
| CLUSD | Crude Oil ($/bbl) | 1,940 | 9 | 19 | May 24 −5.2% |

### Issue 1 — Daily 17:00–18:00 ET Settlement Gap (Critical for Feature Engineering)

Every single trading day has a **60-minute dead zone** from 17:00 ET to 18:00 ET. This is the standard CME/COMEX end-of-day settlement and daily close. It is **not missing data** — the exchange is genuinely closed for settlement.

**Why this matters:** A naive `pct_change()` or lag feature across this boundary will compute a fake large return (e.g., if gold closes at $4,600 at 17:00 and reopens at $4,650 at 18:00, the computed return is +1.1% on what is actually an overnight move). This will corrupt any return-based features.

**Fix:** Add a `session_boundary` flag. Reset all lag/return features at 18:00 ET each day for commodity series.

### Issue 2 — Zero Volume Bars (18 total, safe to fill)

18 bars across the three commodities with volume = 0, all in low-liquidity overnight sessions. Safe to **forward-fill close price** and **leave volume as 0** (or drop the bars entirely if overnight data is not used).

### Issue 3 — CLUSD Spike May 24 18:00 (Real Event)

Crude oil dropped **−5.15%** at the CME Sunday evening open on May 24. This aligns with macro news over the weekend. Real event, not a data error — keep it.

---

## Section 4 — Treasury Yield Curve (`treasury_rates_may2026.csv`)

Daily end-of-day yields across the full curve. Cleanest dataset in the collection.

| Metric | Value |
|---|---|
| Rows | 20 (one per trading day) |
| Nulls | 0 |
| Missing days | 0 |

### Yield Curve Movement — May 1 vs May 29

| Tenor | May 1 | May 29 | Change | Range |
|---|---|---|---|---|
| 1-Month | 3.71% | 3.72% | +1bp | 3.65%–3.72% |
| 3-Month | 3.68% | 3.69% | +1bp | 3.65%–3.70% |
| 1-Year | 3.73% | 3.79% | +6bp | 3.73%–3.86% |
| 2-Year | 3.88% | 3.98% | +10bp | 3.87%–4.13% |
| 5-Year | 4.02% | 4.13% | +11bp | 3.99%–4.32% |
| 10-Year | 4.39% | 4.45% | +6bp | 4.36%–4.67% |
| 30-Year | 4.97% | 4.99% | +2bp | 4.94%–5.18% |

The curve drifted mildly higher across May (bear move). The largest move was at the 2–5 year belly (+10–11bp). Front end (1M, 3M) was essentially flat. The 2Y–10Y spread ended at +47bp (positive, not inverted at that tenor pair).

---

## Section 5 — Action Items Before Feature Engineering

| Priority | Action |
|---|---|
| **P0** | Add `session_boundary` flag to commodity bars at 17:00 ET. Never compute cross-session returns. |
| **P0** | Mark May 25 as holiday in the trading calendar. Do not impute. |
| **P1** | Remove `SHY` from 15-min feature set. Substitute `treasury_rates.year2` (daily, interpolate to 15-min if needed). |
| **P1** | Forward-fill 18 zero-volume commodity bars (or drop if using market-hours only). |
| **P2** | Flag `PSKY` as low-liquidity in symbol metadata. Monitor feature importance. |
| **P3** | The 14 gap-opens in stocks are real signal — do **not** winsorize or clip them. |

---

## File Index

```
reports/
├── DATA_QUALITY_REPORT.md          ← this file
│
├── top29_eda/
│   ├── top29_15m_may2026.csv       15-min OHLCV, 29 symbols, May 2026
│   ├── summary_stats.csv           Per-symbol: bars, mean/std/min/max close, monthly return
│   ├── top29_sanity_report.pdf     Full sanity PDF
│   ├── 01_close_grid.png           29-panel price chart
│   ├── 02_monthly_returns.png      Monthly return bar chart
│   ├── 03_volume_heatmap.png       Volume by symbol and hour
│   ├── 04_volatility.png           Price std dev by symbol
│   ├── 05_correlation.png          Hourly return correlation matrix
│   ├── per_symbol/                 Per-symbol deep-dive charts
│   └── sanity/
│       ├── coverage_heatmap.png
│       ├── volume_anomaly.png
│       └── outlier_bars.csv        14 flagged gap-open bars
│
└── macro_eda/
    ├── macro_15m_may2026.csv       15-min OHLCV, 8 macro symbols, May 2026
    ├── macro_summary_stats.csv     Per-symbol stats and monthly returns
    ├── treasury_rates_may2026.csv  Daily yield curve, 20 rows, 1M→30Y tenors
    ├── 01_macro_price_grid.png     8-panel price chart
    ├── 02_macro_returns.png        Monthly return bar chart
    ├── 03_macro_normalised.png     All indicators normalised to 100 at May 1
    ├── 04_commodity_price_volume.png  Gold/Silver/Crude with volume
    ├── 05_treasury_curve.png       Yield curve snapshots + 2Y/10Y time series
    └── 06_macro_vs_stocks_corr.png Cross-correlation: 29 stocks × 8 macro indicators
```
