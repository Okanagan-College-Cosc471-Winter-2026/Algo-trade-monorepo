"""
Data sanity check for top-29 15-min OHLCV data (May 2026).
Checks: missing bars, gaps, nulls, duplicates, OHLC integrity, volume anomalies, stale prices.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV  = Path("reports/top29_eda/top29_15m_may2026.csv")
OUT  = Path("reports/top29_eda/sanity")
OUT.mkdir(parents=True, exist_ok=True)

# Expected trading slots per day: 09:30–16:00 ET = 26 slots
# May 2026 trade dates (Mon–Fri, skip May 26 Memorial Day)
EXPECTED_TRADE_DATES = pd.bdate_range("2026-05-01", "2026-05-30", freq="C",
                                       holidays=["2026-05-25"]).date.tolist()
SLOTS_PER_DAY = 26          # 09:30, 09:45, ..., 15:45  (inclusive)
EXPECTED_BARS = len(EXPECTED_TRADE_DATES) * SLOTS_PER_DAY  # per symbol

print(f"Expected trade dates : {len(EXPECTED_TRADE_DATES)}")
print(f"Expected bars/symbol : {EXPECTED_BARS}  ({len(EXPECTED_TRADE_DATES)} days × {SLOTS_PER_DAY} slots)")
print()

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV, parse_dates=["ts"])
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
symbols = sorted(df["symbol"].unique())
print(f"Loaded {len(df):,} rows  |  {len(symbols)} symbols\n")

issues = []   # collect all findings

# ─────────────────────────────────────────────────────────────────────────────
# 1. NULL / NaN check
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. NULL / NaN check")
null_counts = df[["symbol","open","high","low","close","volume"]].groupby("symbol").apply(
    lambda g: g[["open","high","low","close","volume"]].isna().sum()
)
null_total = null_counts.sum(axis=1)
null_bad = null_total[null_total > 0]
if null_bad.empty:
    print("   PASS  No nulls in any OHLCV column")
else:
    print(f"   FAIL  {len(null_bad)} symbols have nulls:")
    print(null_bad)
    issues.append(("nulls", null_bad.to_dict()))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Duplicate timestamp check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Duplicate timestamp check")
dups = df.groupby(["symbol","ts"]).size().reset_index(name="n")
dups_bad = dups[dups["n"] > 1]
if dups_bad.empty:
    print("   PASS  No duplicate (symbol, ts) pairs")
else:
    print(f"   FAIL  {len(dups_bad)} duplicate timestamps:")
    print(dups_bad.head(20))
    issues.append(("duplicates", len(dups_bad)))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Bar count per symbol
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Bar count per symbol  (expected ~520 — FMP returns extended hours too)")
bar_counts = df.groupby("symbol")["ts"].count().sort_values()
print(bar_counts.to_string())
count_bad = bar_counts[bar_counts < EXPECTED_BARS]
if count_bad.empty:
    print("\n   PASS  All symbols have >= expected bars")
else:
    print(f"\n   WARN  {len(count_bad)} symbols below {EXPECTED_BARS} bars:")
    print(count_bad)
    issues.append(("low_bar_count", count_bad.to_dict()))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Missing trade dates per symbol
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Missing trade dates")
df["trade_date"] = df["ts"].dt.date
missing_dates_by_sym = {}
for sym in symbols:
    present = set(df[df["symbol"] == sym]["trade_date"].unique())
    missing = sorted(set(EXPECTED_TRADE_DATES) - present)
    if missing:
        missing_dates_by_sym[sym] = missing

if not missing_dates_by_sym:
    print("   PASS  No missing trade dates")
else:
    print(f"   FAIL  {len(missing_dates_by_sym)} symbols missing dates:")
    for sym, dates in missing_dates_by_sym.items():
        print(f"   {sym}: {[str(d) for d in dates]}")
    issues.append(("missing_dates", {k: [str(d) for d in v] for k, v in missing_dates_by_sym.items()}))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Intraday gap check (consecutive 15-min slots within a day)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Intraday timestamp gaps (> 15 min within same trade date)")
gap_records = []
for sym in symbols:
    sub = df[df["symbol"] == sym].copy()
    for td, grp in sub.groupby("trade_date"):
        grp = grp.sort_values("ts")
        deltas = grp["ts"].diff().dt.total_seconds() / 60
        big = deltas[deltas > 20]  # >20 min = missed slot
        for idx, delta in big.items():
            gap_records.append({
                "symbol": sym,
                "trade_date": str(td),
                "after_ts": str(grp.loc[idx, "ts"]),
                "gap_min": round(delta, 1),
            })

if not gap_records:
    print("   PASS  No intraday gaps > 20 min")
else:
    gap_df = pd.DataFrame(gap_records)
    print(f"   WARN  {len(gap_df)} gaps found:")
    print(gap_df.head(30).to_string(index=False))
    gap_df.to_csv(OUT / "intraday_gaps.csv", index=False)
    issues.append(("intraday_gaps", len(gap_df)))

# ─────────────────────────────────────────────────────────────────────────────
# 6. OHLC integrity (high >= low, high >= open/close, low <= open/close)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. OHLC integrity")
bad_hl  = df[df["high"] < df["low"]]
bad_ho  = df[df["high"] < df["open"]]
bad_hc  = df[df["high"] < df["close"]]
bad_lo  = df[df["low"]  > df["open"]]
bad_lc  = df[df["low"]  > df["close"]]

ohlc_issues = {
    "high < low":   len(bad_hl),
    "high < open":  len(bad_ho),
    "high < close": len(bad_hc),
    "low > open":   len(bad_lo),
    "low > close":  len(bad_lc),
}
any_ohlc = sum(ohlc_issues.values())
if any_ohlc == 0:
    print("   PASS  All OHLC relationships valid")
else:
    for k, v in ohlc_issues.items():
        if v:
            print(f"   FAIL  {v} rows: {k}")
    issues.append(("ohlc_integrity", ohlc_issues))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Zero / negative price or volume
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Zero / negative price or volume")
zero_price = df[(df[["open","high","low","close"]] <= 0).any(axis=1)]
zero_vol   = df[df["volume"] <= 0]
if zero_price.empty and zero_vol.empty:
    print("   PASS  No zero/negative prices or volumes")
else:
    if not zero_price.empty:
        print(f"   FAIL  {len(zero_price)} rows with zero/negative price")
    if not zero_vol.empty:
        print(f"   WARN  {len(zero_vol)} rows with zero volume  (may be normal for illiquid bars)")
        print(f"         Symbols: {sorted(zero_vol['symbol'].unique())}")
    issues.append(("zero_values", {"price_rows": len(zero_price), "vol_rows": len(zero_vol)}))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Stale price detection (same close repeated >= 5 consecutive bars)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. Stale price (close unchanged >= 5 consecutive bars)")
stale_records = []
for sym in symbols:
    sub = df[df["symbol"] == sym].sort_values("ts")
    streak = 1
    prev   = None
    for _, row in sub.iterrows():
        if row["close"] == prev:
            streak += 1
            if streak == 5:
                stale_records.append({"symbol": sym, "ts": str(row["ts"]), "close": row["close"], "streak_start": streak})
        else:
            streak = 1
        prev = row["close"]

if not stale_records:
    print("   PASS  No stale-price streaks >= 5 bars")
else:
    print(f"   WARN  {len(stale_records)} stale-price events:")
    for r in stale_records[:15]:
        print(f"   {r['symbol']}  at {r['ts']}  close={r['close']}")
    issues.append(("stale_prices", len(stale_records)))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Outlier bar returns (close-to-close > 5% in one 15-min bar)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Outlier bar returns (|Δclose| > 5% in a single 15-min bar)")
df2 = df.copy()
df2["ret"] = df2.groupby("symbol")["close"].pct_change()
outliers = df2[df2["ret"].abs() > 0.05].dropna()
if outliers.empty:
    print("   PASS  No single-bar returns > 5%")
else:
    print(f"   WARN  {len(outliers)} outlier bars:")
    print(outliers[["symbol","ts","close","ret"]].to_string(index=False))
    outliers.to_csv(OUT / "outlier_bars.csv", index=False)
    issues.append(("outlier_bars", len(outliers)))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Missing bars heatmap chart
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. Generating missing-bars heatmap ...")

# Build coverage grid: symbol × trade_date → bar count
pivot_data = df.groupby(["symbol","trade_date"])["ts"].count().reset_index()
pivot_data.columns = ["symbol","trade_date","bar_count"]
pivot = pivot_data.pivot(index="symbol", columns="trade_date", values="bar_count").fillna(0)

fig, ax = plt.subplots(figsize=(max(14, len(EXPECTED_TRADE_DATES)*0.6), 8))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=SLOTS_PER_DAY)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([str(d) for d in pivot.columns], rotation=90, fontsize=7)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8)
plt.colorbar(im, ax=ax, label="Bars per day")
ax.set_title("Bar Coverage Heatmap — Top-29 Stocks, May 2026\n(green=full, red=missing)")
plt.tight_layout()
fig.savefig(OUT / "coverage_heatmap.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("   saved coverage_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Volume anomaly chart (daily volume vs median)
# ─────────────────────────────────────────────────────────────────────────────
daily_vol = df.groupby(["symbol","trade_date"])["volume"].sum().reset_index()
daily_vol["median_vol"] = daily_vol.groupby("symbol")["volume"].transform("median")
daily_vol["vol_ratio"]  = daily_vol["volume"] / daily_vol["median_vol"]

fig, ax = plt.subplots(figsize=(14, 6))
for sym in symbols:
    sub = daily_vol[daily_vol["symbol"] == sym].sort_values("trade_date")
    ax.plot(sub["trade_date"].astype(str), sub["vol_ratio"], alpha=0.5, linewidth=0.8)
ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="median")
ax.axhline(3.0, color="red",   linewidth=1, linestyle=":",  label="3× spike")
ax.set_xlabel("Trade Date")
ax.set_ylabel("Volume / Median Volume")
ax.set_title("Daily Volume Ratio vs Median — All 29 Symbols, May 2026")
plt.xticks(rotation=45, ha="right", fontsize=7)
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "volume_anomaly.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("   saved volume_anomaly.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SANITY SUMMARY")
print("=" * 60)
total_bars = len(df)
total_expected = len(symbols) * EXPECTED_BARS
coverage_pct = total_bars / total_expected * 100
print(f"  Total bars fetched   : {total_bars:,}")
print(f"  Total bars expected  : {total_expected:,}  (incl. extended hours — will exceed 100%)")
print(f"  Coverage ratio       : {coverage_pct:.1f}%")
print(f"  Issues found         : {len(issues)}")
for name, detail in issues:
    print(f"    - {name}: {detail}")
if not issues:
    print("  ALL CHECKS PASSED")

print(f"\nOutputs: {OUT}/")
