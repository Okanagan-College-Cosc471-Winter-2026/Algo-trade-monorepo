"""
Standalone FMP fetch + EDA for the top-29 predicted stocks (no DB required).

Fetches 15-min OHLCV bars from FMP for May 2026, saves to CSV, and
produces basic EDA charts in reports/top29_eda/.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import date

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "ml" / "ml" / ".env")

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"

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
CHUNK_DAYS = 5       # FMP free tier handles ~5-day chunks well
DELAY_SEC  = 0.4     # polite rate-limit

OUT_DIR  = ROOT / "reports" / "top29_eda"
CSV_FILE = OUT_DIR / "top29_15m_may2026.csv"

# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "algo-trade-eda/1.0"
    return s


def fetch_chunk(session: requests.Session, symbol: str, from_: date, to: date) -> list[dict]:
    url = f"{FMP_BASE}/historical-chart/15min"
    params = {
        "symbol": symbol,
        "from":   from_.isoformat(),
        "to":     to.isoformat(),
        "apikey": FMP_API_KEY,
    }
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
    except Exception as exc:
        print(f"  WARN {symbol} {from_}→{to}: {exc}")
    return []


def fetch_symbol(session: requests.Session, symbol: str) -> pd.DataFrame:
    from datetime import timedelta
    rows = []
    cur = START_DATE
    while cur <= END_DATE:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), END_DATE)
        bars = fetch_chunk(session, symbol, cur, chunk_end)
        rows.extend(bars)
        cur = chunk_end + timedelta(days=1)
        time.sleep(DELAY_SEC)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "ts"})
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["symbol", "ts", "open", "high", "low", "close", "volume"]]


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

    # ---- Fetch ----
    if CSV_FILE.exists():
        print(f"Loading cached data from {CSV_FILE.name} ...")
        df = pd.read_csv(CSV_FILE, parse_dates=["ts"])
        already = set(df["symbol"].unique())
        missing = [s for s in TOP29 if s not in already]
        if missing:
            print(f"Fetching {len(missing)} missing symbols: {missing}")
            session = _session()
            parts = [df]
            for i, sym in enumerate(missing, 1):
                print(f"  [{i}/{len(missing)}] {sym}")
                part = fetch_symbol(session, sym)
                if not part.empty:
                    parts.append(part)
            df = pd.concat(parts, ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
    else:
        print(f"Fetching {len(TOP29)} symbols from FMP ({START_DATE} → {END_DATE}) ...")
        session = _session()
        frames = []
        for i, sym in enumerate(TOP29, 1):
            print(f"  [{i}/{len(TOP29)}] {sym}")
            part = fetch_symbol(session, sym)
            if part.empty:
                print(f"    WARNING: no data for {sym}")
            else:
                print(f"    {len(part):,} bars")
                frames.append(part)
        if not frames:
            print("ERROR: No data fetched at all. Check FMP API key.")
            sys.exit(1)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        print(f"\nSaved {len(df):,} rows → {CSV_FILE}")

    print(f"\nLoaded {len(df):,} bars for {df['symbol'].nunique()} symbols")

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
