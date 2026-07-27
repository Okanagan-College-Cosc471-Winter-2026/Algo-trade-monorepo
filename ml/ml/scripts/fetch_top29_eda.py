"""
Standalone FMP fetch + EDA for the top-29 predicted stocks (no DB required).

Fetches 15-min OHLCV bars from FMP for May 2026, saves to CSV, and
produces basic EDA charts in reports/top29_eda/.
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

sys.path.insert(0, str(Path(__file__).parent))
from fmp_eda_common import (
    fetch_15m_series,
    fetch_daily_close_series,
    load_or_fetch,
    make_session,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]

# Top-29 by predicted return from warm-refresh model
TOP29 = [
    "PSKY", "CF",   "MCD",  "KEYS", "UNH",  "DPZ",  "CI",
    "ELV",  "GE",   "MKC",  "CBOE", "OMC",  "APA",  "HUM",
    "CCI",  "DVN",  "CVNA", "PM",   "XOM",  "INTC", "HLT",
    "CNC",  "MO",   "ADI",  "MNST", "CL",   "EOG",  "IT",
    "GOOGL",
]

START_DATE = date(2026, 5, 1)
END_DATE   = date(2026, 5, 31)

OUT_DIR  = ROOT / "reports" / "top29_eda"
CSV_FILE = OUT_DIR / "top29_15m_may2026.csv"
DAILY_CLOSE_CSV = OUT_DIR / "top29_daily_close_may2026.csv"

# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("symbol")
    stats = pd.DataFrame({
        "bars":         g["close"].count(),
        "mean_close":   g["close"].mean().round(2),
        "std_close":    g["close"].std().round(4),
        "min_close":    g["close"].min().round(2),
        "max_close":    g["close"].max().round(2),
        "mean_volume":  g["volume"].mean().round(0),
        "total_volume": g["volume"].sum().round(0),
    }).reset_index()

    # monthly return: last close / first close - 1
    first = g["close"].first()
    last  = g["close"].last()
    stats["monthly_return_pct"] = ((last / first - 1) * 100).round(2).values
    return stats.sort_values("monthly_return_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def plot_close_grid(df: pd.DataFrame, out: Path) -> None:
    symbols = df["symbol"].unique()
    n = len(symbols)
    cols = 5
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5))
    axes = axes.flatten()
    for i, sym in enumerate(sorted(symbols)):
        ax = axes[i]
        sub = df[df["symbol"] == sym]
        ax.plot(sub["ts"], sub["close"], linewidth=0.7)
        ax.set_title(sym, fontsize=9, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.tick_params(labelsize=6)
        ax.set_ylabel("Close", fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Top-29 Stocks — 15-min Close Price (May 2026)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_monthly_returns(stats: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in stats["monthly_return_pct"]]
    bars = ax.barh(stats["symbol"], stats["monthly_return_pct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Monthly Return %")
    ax.set_title("Top-29 Actual Monthly Returns — May 2026")
    for bar, val in zip(bars, stats["monthly_return_pct"]):
        ax.text(
            val + (0.1 if val >= 0 else -0.1),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=7,
        )
    plt.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_volume_heatmap(df: pd.DataFrame, out: Path) -> None:
    df2 = df.copy()
    df2["hour"] = df2["ts"].dt.hour
    df2["date"] = df2["ts"].dt.date
    pivot = df2.groupby(["symbol", "hour"])["volume"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", fmt=".0f",
                linewidths=0.3, cbar_kws={"label": "Avg Volume"})
    ax.set_title("Average Volume by Symbol & Hour (May 2026)")
    ax.set_xlabel("Hour of Day (ET)")
    ax.set_ylabel("Symbol")
    plt.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_volatility(stats: pd.DataFrame, out: Path) -> None:
    s = stats.sort_values("std_close", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(s["symbol"], s["std_close"], color="#3498db")
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Std Dev of 15-min Close")
    ax.set_title("Price Volatility (Std Dev) — Top-29, May 2026")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_correlation(df: pd.DataFrame, out: Path) -> None:
    pivot = (
        df.set_index(["ts", "symbol"])["close"]
        .unstack("symbol")
        .resample("1h").last()
        .dropna(how="all")
    )
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = ~corr.notna()
    sns.heatmap(
        corr, ax=ax, cmap="coolwarm", center=0,
        vmin=-1, vmax=1, mask=mask,
        linewidths=0.2, annot=False,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Hourly Close Return Correlation — Top-29, May 2026")
    plt.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session("algo-trade-eda/1.0")

    # ---- 15-min bars (regular session; last bar per day starts 15:45, is NOT the 4pm close) ----
    df = load_or_fetch(
        CSV_FILE, TOP29, key_col="symbol", start=START_DATE, end=END_DATE,
        fetch_fn=fetch_15m_series, session=session, label="15-min bars",
    )
    print(f"\nLoaded {len(df):,} bars for {df['symbol'].nunique()} symbols")

    # ---- Official daily close (true 4:00pm print, joined on symbol+date) ----
    daily_close = load_or_fetch(
        DAILY_CLOSE_CSV, TOP29, key_col="symbol", start=START_DATE, end=END_DATE,
        fetch_fn=fetch_daily_close_series, session=session, label="daily close", date_col="date",
    )
    print(f"Loaded {len(daily_close):,} daily closes for {daily_close['symbol'].nunique()} symbols")

    # ---- Stats ----
    stats = compute_stats(df)
    stats_path = OUT_DIR / "summary_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"\n=== Summary Stats ===")
    print(stats.to_string(index=False))

    # ---- Charts ----
    print("\nGenerating charts ...")
    plot_close_grid(df, OUT_DIR / "01_close_grid.png")
    plot_monthly_returns(stats, OUT_DIR / "02_monthly_returns.png")
    plot_volume_heatmap(df, OUT_DIR / "03_volume_heatmap.png")
    plot_volatility(stats, OUT_DIR / "04_volatility.png")
    plot_correlation(df, OUT_DIR / "05_correlation.png")

    print(f"\nDone. All outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
