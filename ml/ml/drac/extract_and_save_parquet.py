#!/usr/bin/env python3
"""
extract_and_save_parquet.py
────────────────────────────
Pulls all 5-min OHLCV data from the VPS Postgres database and writes
a dated Parquet file to the specified output directory.

Usage (on DRAC, after loading arrow module and activating venv):
    python extract_and_save_parquet.py --out_dir /scratch/$USER/ml/dt=2026-03-03

Environment variables (can also be passed as args):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text


def get_engine(args):
    host = args.host or os.getenv("POSTGRES_HOST", "localhost")
    port = args.port or os.getenv("POSTGRES_PORT", "5432")
    db   = args.db   or os.getenv("POSTGRES_DB",   "app")
    user = args.user or os.getenv("POSTGRES_USER", "postgres")
    pw   = args.password or os.getenv("POSTGRES_PASSWORD", "changethis")
    url  = f"postgresql://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url)


def get_all_tickers(engine) -> list[str]:
    q = text("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'market'
        ORDER BY table_name
    """)
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()
    return [r[0] for r in rows if r[0] not in ("daily_prices", "stocks")]


def extract_ticker(engine, ticker: str, start_date=None, end_date=None) -> pd.DataFrame:
    q = f'SELECT * FROM market."{ticker}"'
    conditions = []
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    if conditions:
        q += " WHERE " + " AND ".join(conditions)
    q += " ORDER BY date ASC"
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)
    if not df.empty and "symbol" not in df.columns:
        df["symbol"] = ticker
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract Postgres → Parquet")
    parser.add_argument("--out_dir",   required=True, help="Output directory for Parquet file")
    parser.add_argument("--host",      default=None)
    parser.add_argument("--port",      default=None)
    parser.add_argument("--db",        default=None)
    parser.add_argument("--user",      default=None)
    parser.add_argument("--password",  default=None)
    parser.add_argument("--start_date", default=None, help="e.g. 2020-01-01")
    parser.add_argument("--end_date",   default=None, help="e.g. 2025-12-31")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Connecting to Postgres at {args.host or os.getenv('POSTGRES_HOST', 'localhost')} ...")
    engine = get_engine(args)

    tickers = get_all_tickers(engine)
    print(f"[extract] Found {len(tickers)} ticker tables: {tickers}")

    t0 = time.time()
    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = extract_ticker(engine, ticker, args.start_date, args.end_date)
        if not df.empty:
            frames.append(df)
        print(f"  [{i:2d}/{len(tickers)}] {ticker}: {len(df):,} rows", flush=True)

    if not frames:
        print("[extract] ERROR: No data extracted. Exiting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    out_file = out_dir / f"snapshot_{date.today().isoformat()}.parquet"
    table = pa.Table.from_pandas(combined)
    pq.write_table(table, out_file)

    elapsed = time.time() - t0
    print(f"\n[extract] Done — {len(combined):,} rows, {len(tickers)} tickers")
    print(f"[extract] Saved to: {out_file}")
    print(f"[extract] Elapsed : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
