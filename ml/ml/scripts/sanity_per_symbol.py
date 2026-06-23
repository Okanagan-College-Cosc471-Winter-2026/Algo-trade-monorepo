"""Per-symbol sanity cards with individual charts."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CSV = Path("reports/top29_eda/top29_15m_may2026.csv")
OUT = Path("reports/top29_eda/per_symbol")
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_TRADE_DATES = pd.bdate_range("2026-05-01", "2026-05-30", freq="C",
                                       holidays=["2026-05-25"]).date.tolist()
SLOTS_PER_DAY = 26

df = pd.read_csv(CSV, parse_dates=["ts"])
df["ts"]         = pd.to_datetime(df["ts"])
df["trade_date"] = df["ts"].dt.date
df = df.sort_values(["symbol","ts"]).reset_index(drop=True)
symbols = sorted(df["symbol"].unique())

# ── summary table ─────────────────────────────────────────────────────────────
rows = []
for sym in symbols:
    sub = df[df["symbol"] == sym].sort_values("ts").copy()

    # basics
    total_bars  = len(sub)
    dates_present = set(sub["trade_date"].unique())
    missing_dates = sorted(set(EXPECTED_TRADE_DATES) - dates_present)

    # nulls
    nulls = sub[["open","high","low","close","volume"]].isna().sum().sum()

    # duplicates
    dups = sub.duplicated(subset=["ts"]).sum()

    # gaps > 20 min intraday
    gaps = 0
    for td, grp in sub.groupby("trade_date"):
        deltas = grp.sort_values("ts")["ts"].diff().dt.total_seconds() / 60
        gaps  += (deltas > 20).sum()

    # OHLC integrity violations
    ohlc_bad = (
        (sub["high"] < sub["low"]) |
        (sub["high"] < sub["open"]) |
        (sub["high"] < sub["close"]) |
        (sub["low"]  > sub["open"]) |
        (sub["low"]  > sub["close"])
    ).sum()

    # zero volume bars
    zero_vol = (sub["volume"] <= 0).sum()

    # stale close (same value >= 5 consecutive bars)
    stale_events = 0
    streak, prev = 1, None
    for c in sub["close"]:
        if c == prev:
            streak += 1
            if streak == 5:
                stale_events += 1
        else:
            streak = 1
        prev = c

    # outlier bars > 5%
    sub["ret"] = sub["close"].pct_change()
    outliers = (sub["ret"].abs() > 0.05).sum()

    # monthly return
    monthly_ret = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100

    status = "CLEAN" if (nulls == 0 and dups == 0 and gaps == 0 and
                         ohlc_bad == 0 and not missing_dates) else "ISSUES"

    rows.append(dict(
        symbol=sym, status=status, bars=total_bars,
        missing_dates=len(missing_dates), intraday_gaps=int(gaps),
        nulls=int(nulls), duplicates=int(dups),
        ohlc_violations=int(ohlc_bad), zero_vol_bars=int(zero_vol),
        stale_events=int(stale_events), outlier_bars=int(outliers),
        monthly_return_pct=round(monthly_ret, 2),
    ))

summary = pd.DataFrame(rows).sort_values("monthly_return_pct", ascending=False).reset_index(drop=True)
summary_path = OUT / "per_symbol_sanity.csv"
summary.to_csv(summary_path, index=False)

print("=" * 72)
print(f"{'SYM':<6} {'STATUS':<8} {'BARS':>5} {'MISS_D':>6} {'GAPS':>5} "
      f"{'NULLS':>5} {'DUPS':>5} {'OHLC':>5} {'0VOL':>6} {'STALE':>6} {'OUT':>4} {'RET%':>7}")
print("-" * 72)
for _, r in summary.iterrows():
    flag = "" if r["status"] == "CLEAN" else " <--"
    print(f"{r['symbol']:<6} {r['status']:<8} {r['bars']:>5} {r['missing_dates']:>6} "
          f"{r['intraday_gaps']:>5} {r['nulls']:>5} {r['duplicates']:>5} "
          f"{r['ohlc_violations']:>5} {r['zero_vol_bars']:>6} {r['stale_events']:>6} "
          f"{r['outlier_bars']:>4} {r['monthly_return_pct']:>7.2f}%{flag}")
print("=" * 72)

issues_count = (summary["status"] == "ISSUES").sum()
print(f"\nClean: {len(symbols)-issues_count}/{len(symbols)}   Issues: {issues_count}/{len(symbols)}")
print(f"Saved: {summary_path}")

# ── Per-symbol 4-panel chart ───────────────────────────────────────────────────
print("\nGenerating per-symbol charts ...")
for sym in symbols:
    sub = df[df["symbol"] == sym].sort_values("ts").copy()
    sub["ret"]     = sub["close"].pct_change()
    sub["vol_med"] = sub["volume"].median()

    daily = sub.groupby("trade_date").agg(
        close=("close","last"),
        volume=("volume","sum"),
        bars=("ts","count"),
    ).reset_index()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{sym} — Data Sanity Card  (May 2026)", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: 15-min close
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(sub["ts"], sub["close"], linewidth=0.7, color="#2c7bb6")
    ax1.set_title("15-min Close Price")
    ax1.set_ylabel("Close ($)")
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))

    # Panel 2: bar count per day (should be flat at 26)
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ["#e74c3c" if b < SLOTS_PER_DAY else "#2ecc71" for b in daily["bars"]]
    ax2.bar(daily["trade_date"].astype(str), daily["bars"], color=colors)
    ax2.axhline(SLOTS_PER_DAY, color="black", linewidth=1, linestyle="--", label=f"expected {SLOTS_PER_DAY}")
    ax2.set_title("Bars per Trade Date")
    ax2.set_ylabel("# bars")
    ax2.tick_params(axis="x", rotation=90, labelsize=6)
    ax2.legend(fontsize=7)

    # Panel 3: daily volume vs median
    med_daily = daily["volume"].median()
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(daily["trade_date"].astype(str), daily["volume"], color="#3498db", alpha=0.7)
    ax3.axhline(med_daily, color="orange", linewidth=1.5, linestyle="--", label="median")
    ax3.axhline(med_daily * 3, color="red", linewidth=1, linestyle=":", label="3× spike")
    ax3.set_title("Daily Volume")
    ax3.set_ylabel("Volume")
    ax3.tick_params(axis="x", rotation=90, labelsize=6)
    ax3.legend(fontsize=7)
    ax3.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # Panel 4: 15-min return distribution
    ax4 = fig.add_subplot(gs[2, 0])
    rets = sub["ret"].dropna()
    ax4.hist(rets, bins=60, color="#9b59b6", edgecolor="none", alpha=0.8)
    ax4.axvline(0, color="black", linewidth=1)
    ax4.axvline( 0.05, color="red", linewidth=1, linestyle="--", label="±5%")
    ax4.axvline(-0.05, color="red", linewidth=1, linestyle="--")
    ax4.set_title("15-min Return Distribution")
    ax4.set_xlabel("Return")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=7)

    # Panel 5: spread (high-low) over time
    ax5 = fig.add_subplot(gs[2, 1])
    sub["spread_pct"] = (sub["high"] - sub["low"]) / sub["close"] * 100
    ax5.plot(sub["ts"], sub["spread_pct"], linewidth=0.5, color="#e67e22", alpha=0.7)
    ax5.set_title("Bar Spread % (H-L / Close)")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Spread %")
    ax5.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))

    fig.savefig(OUT / f"{sym}_sanity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {sym}")

print(f"\nAll per-symbol charts saved to {OUT}/")
