"""
fill_gap.py — Fetch missing 5-min OHLCV data from FMP and append to Postgres.

For each ticker, reads the current max date from market."{TICKER}" and
downloads only the missing range up to today, then inserts without touching
existing rows.

Usage:
    python ml/scripts/fill_gap.py
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(ROOT, ".env"))          # DB creds (app DB)
load_dotenv(os.path.join(ROOT, "ml", ".env"))    # FMP_API_KEY

FMP_API_KEY  = os.getenv("FMP_API_KEY")
DB_USER      = os.getenv("POSTGRES_USER", "postgres")
DB_PASS      = os.getenv("POSTGRES_PASSWORD", "changethis")
DB_NAME      = os.getenv("POSTGRES_DB", "app")
DB_HOST      = "localhost"
DB_PORT      = 5432

TICKERS = [
    "AAPL","AMD","AMZN","BA","BABA","BAC","C","CSCO","CVX",
    "DIS","F","GE","GOOGL","IBM","INTC","JNJ","JPM","KO",
    "MCD","META","MSFT","NFLX","NVDA","PFE","T","TSLA","VZ","WMT","XOM",
]

CHUNK_DAYS   = 25   # FMP max window per request
SLEEP_S      = 0.5  # rate-limit delay between chunks

CSV_DIR = os.path.join(ROOT, "ml", "data", "fmp_gap_fill")
os.makedirs(CSV_DIR, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_engine():
    return create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


def get_max_date(engine, ticker: str) -> datetime | None:
    """Return the latest timestamp in market."{ticker}", or None if empty."""
    with engine.connect() as conn:
        row = conn.execute(
            text(f'SELECT MAX(date) FROM market."{ticker}"')
        ).one()
    val = row[0]
    if val is None:
        return None
    if isinstance(val, str):
        return pd.to_datetime(val).to_pydatetime()
    return val  # already datetime


def fetch_fmp(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download one 5-min chunk from FMP. Returns DataFrame or None."""
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker}"
        f"?from={start}&to={end}&extended=true&apikey={FMP_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)[["date","open","high","low","close","volume"]]
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as exc:
        print(f"    [warn] {exc}")
        return None


def fill_ticker(engine, ticker: str, today: datetime) -> int:
    """Download and insert missing rows for one ticker. Returns rows inserted."""
    max_dt = get_max_date(engine, ticker)

    if max_dt is None:
        print(f"  {ticker}: table empty — skipping (run full import first)")
        return 0

    # Start the day after the last known row
    gap_start = (max_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    if gap_start.date() >= today.date():
        print(f"  {ticker}: already up to date ({max_dt.date()})")
        return 0

    print(f"  {ticker}: fetching {gap_start.date()} → {today.date()}")

    chunks = []
    cur = gap_start
    while cur.date() <= today.date():
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), today)
        df = fetch_fmp(ticker, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        if df is not None:
            chunks.append(df)
        time.sleep(SLEEP_S)
        if chunk_end.date() >= today.date():
            break
        cur = chunk_end + timedelta(days=1)

    if not chunks:
        print(f"    no new data returned")
        return 0

    new_df = (
        pd.concat(chunks, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date"])
    )

    # Safety: only keep rows strictly after existing max date
    new_df = new_df[new_df["date"] > max_dt]

    if new_df.empty:
        print(f"    all rows already present")
        return 0

    # Save to CSV (append if file already exists from a previous partial run)
    csv_path = os.path.join(CSV_DIR, f"{ticker}.csv")
    if os.path.exists(csv_path):
        existing_csv = pd.read_csv(csv_path, parse_dates=["date"])
        new_df = (
            pd.concat([existing_csv, new_df], ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset=["date"])
        )
        new_df.to_csv(csv_path, index=False)
        print(f"    updated CSV → {csv_path}")
    else:
        new_df.to_csv(csv_path, index=False)
        print(f"    saved CSV  → {csv_path}")

    new_df.to_sql(ticker, engine, schema="market", if_exists="append", index=False)
    print(f"    inserted {len(new_df):,} rows  (up to {new_df['date'].max()})")
    return len(new_df)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY not found in .env")

    engine = get_engine()
    today  = datetime.now()
    total  = 0

    print(f"Gap-fill run: target range ends {today.date()}\n")

    for ticker in TICKERS:
        total += fill_ticker(engine, ticker, today)

    print(f"\nDone — {total:,} rows inserted across {len(TICKERS)} tickers.")


if __name__ == "__main__":
    main()
