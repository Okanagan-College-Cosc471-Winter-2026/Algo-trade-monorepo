"""
Fetch macro/market indicators (15-min bars + daily treasury rates) from FMP
for May 2026 and produce an EDA report alongside the top-29 stock data.

Indicators fetched
------------------
15-min bars  : GCUSD (Gold), SIUSD (Silver), CLUSD (Crude Oil),
               SPY (S&P 500), QQQ (Nasdaq), TLT (20yr bond),
               IEF (7-10yr bond), SHY (2yr bond)
Daily        : Treasury yield curve (1m → 30y) via /treasury-rates

Outputs  →  reports/macro_eda/
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from fmp_eda_common import (
    FMP_API_KEY,
    FMP_INTRADAY_BASE as FMP_BASE,
    fetch_15m_series,
    fetch_daily_close_series,
    load_or_fetch,
    make_session,
)

START_DATE  = date(2026, 5, 1)
END_DATE    = date(2026, 5, 31)

MACRO_SYMBOLS = {
    "GCUSD": "Gold ($/oz)",
    "SIUSD": "Silver ($/oz)",
    "CLUSD": "Crude Oil ($/bbl)",
    "SPY":   "S&P 500 ETF",
    "QQQ":   "Nasdaq-100 ETF",
    "TLT":   "20yr Treasury ETF",
    "IEF":   "7-10yr Treasury ETF",
    "SHY":   "2yr Treasury ETF",
}

OUT_DIR          = ROOT / "reports" / "macro_eda"
BARS_CSV         = OUT_DIR / "macro_15m_may2026.csv"
DAILY_CLOSE_CSV  = OUT_DIR / "macro_daily_close_may2026.csv"
TREASURY_CSV     = OUT_DIR / "treasury_rates_may2026.csv"
TOP29_CSV        = ROOT / "reports" / "top29_eda" / "top29_15m_may2026.csv"


# ---------------------------------------------------------------------------
# Fetch helpers (15-min bars + daily close come from fmp_eda_common; treasury
# rates are a macro-EDA-only endpoint so stay local)
# ---------------------------------------------------------------------------

def fetch_treasury_rates(session) -> pd.DataFrame:
    url = f"{FMP_BASE}/treasury-rates"
    params = {"from": START_DATE.isoformat(), "to": END_DATE.isoformat(),
              "apikey": FMP_API_KEY}
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        print(f"  WARN treasury: {exc}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("symbol")
    stats = pd.DataFrame({
        "bars":             g["close"].count(),
        "mean_close":       g["close"].mean().round(4),
        "std_close":        g["close"].std().round(4),
        "min_close":        g["close"].min().round(4),
        "max_close":        g["close"].max().round(4),
        "mean_volume":      g["volume"].mean().round(0),
    }).reset_index()
    first = g["close"].first()
    last  = g["close"].last()
    stats["monthly_return_pct"] = ((last / first - 1) * 100).round(2).values
    stats["label"] = stats["symbol"].map(MACRO_SYMBOLS).fillna(stats["symbol"])
    return stats.sort_values("monthly_return_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_price_grid(df: pd.DataFrame, out: Path) -> None:
    symbols = sorted(df["symbol"].unique())
    n    = len(symbols)
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = axes.flatten()
    for i, sym in enumerate(symbols):
        ax  = axes[i]
        sub = df[df["symbol"] == sym]
        ax.plot(sub["ts"], sub["close"], linewidth=0.7, color="#2980b9")
        ax.set_title(f"{sym}\n{MACRO_SYMBOLS.get(sym, '')}", fontsize=8, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.tick_params(labelsize=6)
        plt.setp(ax.get_xticklabels(), rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Macro Indicators — 15-min Close Price (May 2026)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_returns(stats: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = ["#e74c3c" if r < 0 else "#27ae60" for r in stats["monthly_return_pct"]]
    bars    = ax.barh(stats["label"], stats["monthly_return_pct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Monthly Return %")
    ax.set_title("Macro Indicators — May 2026 Returns")
    for bar, val in zip(bars, stats["monthly_return_pct"]):
        ax.text(val + (0.05 if val >= 0 else -0.05),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_treasury_curve(treasury: pd.DataFrame, out: Path) -> None:
    tenor_cols = ["month1", "month2", "month3", "month6",
                  "year1", "year2", "year3", "year5", "year7", "year10", "year20", "year30"]
    tenor_labels = ["1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Yield curve snapshots (first, mid, last trading day)
    rows_to_plot = [0, len(treasury) // 2, -1]
    colors_snap  = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels_snap  = ["May 1", "Mid-May", "May 29"]
    for idx, col, lbl in zip(rows_to_plot, colors_snap, labels_snap):
        row = treasury.iloc[idx]
        vals = [row[c] for c in tenor_cols if c in row]
        ax1.plot(tenor_labels[:len(vals)], vals, marker="o", color=col, label=lbl)
    ax1.set_title("Treasury Yield Curve — May 2026")
    ax1.set_xlabel("Tenor")
    ax1.set_ylabel("Yield (%)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 10Y and 2Y over the month
    ax2.plot(treasury["date"], treasury["year10"], label="10Y", color="#2980b9")
    ax2.plot(treasury["date"], treasury["year2"],  label="2Y",  color="#e74c3c")
    ax2.fill_between(treasury["date"],
                     treasury["year10"], treasury["year2"],
                     alpha=0.15, color="grey", label="Spread 10Y-2Y")
    ax2.set_title("10Y vs 2Y Treasury Yield (May 2026)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Yield (%)")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_normalised_overlay(df: pd.DataFrame, out: Path) -> None:
    # Normalise each symbol to 100 at first bar and plot together
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = plt.cm.tab10.colors
    for i, sym in enumerate(sorted(df["symbol"].unique())):
        sub   = df[df["symbol"] == sym].set_index("ts")["close"].resample("1h").last().dropna()
        normd = sub / sub.iloc[0] * 100
        ax.plot(normd.index, normd.values, label=f"{sym}", linewidth=1.2,
                color=palette[i % len(palette)])
    ax.axhline(100, color="black", linewidth=0.6, linestyle="--")
    ax.set_title("Macro Indicators — Normalised to 100 at May 1 Open (hourly resampled)")
    ax.set_ylabel("Normalised Price")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_macro_vs_stocks_corr(macro_df: pd.DataFrame, top29_df: pd.DataFrame, out: Path) -> None:
    # Hourly returns correlation: macro symbols vs top-29 stocks
    def hourly_returns(df: pd.DataFrame) -> pd.DataFrame:
        pivot = (df.set_index(["ts", "symbol"])["close"]
                   .unstack("symbol")
                   .resample("1h").last()
                   .dropna(how="all")
                   .pct_change()
                   .dropna(how="all"))
        return pivot

    macro_ret  = hourly_returns(macro_df)
    stock_ret  = hourly_returns(top29_df)
    combined   = pd.concat([macro_ret, stock_ret], axis=1).dropna()

    macro_syms = list(macro_df["symbol"].unique())
    stock_syms = list(top29_df["symbol"].unique())

    # Only keep columns present in combined
    macro_cols = [c for c in macro_syms if c in combined.columns]
    stock_cols = [c for c in stock_syms if c in combined.columns]

    cross = pd.DataFrame(index=stock_cols, columns=macro_cols, dtype=float)
    for m in macro_cols:
        for s in stock_cols:
            cross.loc[s, m] = combined[m].corr(combined[s])

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(cross.astype(float), ax=ax, cmap="coolwarm", center=0,
                vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.3)
    ax.set_title("Hourly Return Correlation: Top-29 Stocks vs Macro Indicators (May 2026)")
    ax.set_xlabel("Macro Indicator")
    ax.set_ylabel("Stock Symbol")
    plt.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_commodity_intraday(df: pd.DataFrame, out: Path) -> None:
    commodities = [s for s in ["GCUSD", "SIUSD", "CLUSD"] if s in df["symbol"].unique()]
    if not commodities:
        return
    fig, axes = plt.subplots(len(commodities), 1, figsize=(14, len(commodities) * 3), sharex=True)
    if len(commodities) == 1:
        axes = [axes]
    colors = {"GCUSD": "#f1c40f", "SIUSD": "#bdc3c7", "CLUSD": "#2c3e50"}
    for ax, sym in zip(axes, commodities):
        sub = df[df["symbol"] == sym]
        ax.plot(sub["ts"], sub["close"], linewidth=0.8, color=colors.get(sym, "#2980b9"))
        ax2 = ax.twinx()
        ax2.bar(sub["ts"], sub["volume"], alpha=0.2, color="grey", width=0.008)
        ax2.set_ylabel("Volume", fontsize=7)
        ax.set_ylabel(f"{MACRO_SYMBOLS.get(sym, sym)}", fontsize=8)
        ax.set_title(f"{sym} — Price + Volume (May 2026)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = make_session("algo-trade-macro-eda/1.0")
    symbols = list(MACRO_SYMBOLS)

    # ── 15-min bars (regular session; last bar/day starts 15:45, is NOT the 4pm close) ──
    df = load_or_fetch(
        BARS_CSV, symbols, key_col="symbol", start=START_DATE, end=END_DATE,
        fetch_fn=fetch_15m_series, session=session, label="15-min bars",
    )
    print(f"\nLoaded {len(df):,} bars for {df['symbol'].nunique()} macro symbols")
    print("Symbols:", sorted(df["symbol"].unique()))

    # ── Official daily close (true 4:00pm print for the equity ETFs; CME daily
    #    settlement close for the commodities) ─────────────────────────────────
    daily_close = load_or_fetch(
        DAILY_CLOSE_CSV, symbols, key_col="symbol", start=START_DATE, end=END_DATE,
        fetch_fn=fetch_daily_close_series, session=session, label="daily close", date_col="date",
    )
    print(f"Loaded {len(daily_close):,} daily closes for {daily_close['symbol'].nunique()} symbols")

    # ── Treasury rates ────────────────────────────────────────────────────────
    if TREASURY_CSV.exists():
        print(f"Loading cached treasury rates from {TREASURY_CSV.name} ...")
        treasury = pd.read_csv(TREASURY_CSV, parse_dates=["date"])
    else:
        print("Fetching treasury yield curve ...")
        treasury = fetch_treasury_rates(session)
        if treasury.empty:
            print("  WARNING: no treasury data")
        else:
            treasury.to_csv(TREASURY_CSV, index=False)
            print(f"Saved {len(treasury)} rows → {TREASURY_CSV}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = compute_stats(df)
    stats.to_csv(OUT_DIR / "macro_summary_stats.csv", index=False)
    print("\n=== Macro Summary Stats ===")
    print(stats[["symbol", "label", "bars", "mean_close", "std_close",
                 "min_close", "max_close", "monthly_return_pct"]].to_string(index=False))

    if not treasury.empty:
        tenor_cols = ["month1", "month3", "year1", "year2", "year5", "year10", "year30"]
        print("\n=== Treasury Yields (May 1 vs May 29) ===")
        first = treasury.iloc[0]
        last  = treasury.iloc[-1]
        for tc in tenor_cols:
            if tc in treasury.columns:
                chg = last[tc] - first[tc]
                print(f"  {tc:8s}: {first[tc]:.2f}% → {last[tc]:.2f}%  ({chg:+.2f}bp × 100)")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nGenerating charts ...")
    plot_price_grid(df, OUT_DIR / "01_macro_price_grid.png")
    plot_returns(stats, OUT_DIR / "02_macro_returns.png")
    plot_normalised_overlay(df, OUT_DIR / "03_macro_normalised.png")
    plot_commodity_intraday(df, OUT_DIR / "04_commodity_price_volume.png")

    if not treasury.empty:
        plot_treasury_curve(treasury, OUT_DIR / "05_treasury_curve.png")

    if TOP29_CSV.exists():
        print("Loading top-29 stock data for cross-correlation ...")
        top29 = pd.read_csv(TOP29_CSV, parse_dates=["ts"])
        plot_macro_vs_stocks_corr(df, top29, OUT_DIR / "06_macro_vs_stocks_corr.png")
    else:
        print(f"  SKIP correlation chart — {TOP29_CSV} not found")

    print(f"\nDone. All outputs in {OUT_DIR}/")
    print("\nFiles written:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
