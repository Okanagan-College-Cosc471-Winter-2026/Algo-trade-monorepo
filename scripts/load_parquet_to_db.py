#!/usr/bin/env python3
"""
Load datasets/snapshot_2026-04-18.parquet into:
  - dw.market_data_15m   (6.6M rows, bulk COPY)
  - market.stocks        (505 symbols, idempotent)
  - staging.ingestion_progress (505 symbols)
"""
import os, sys, io, time
import pandas as pd
import psycopg2
from psycopg2 import sql

DB = dict(
    host=os.getenv("POSTGRES_SERVER", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5433")),
    dbname=os.getenv("POSTGRES_DB", "algotrade"),
    user=os.getenv("POSTGRES_USER", "appuser"),
    password=os.getenv("POSTGRES_PASSWORD", "apppassword"),
)

PARQUET = os.getenv("PARQUET_PATH", "datasets/snapshot_2026-04-18.parquet")

# Columns to insert (exclude agg_id — it's GENERATED ALWAYS AS IDENTITY)
DW_COLS = [
    "symbol", "window_ts", "trade_date",
    "open", "high", "low", "close", "volume", "slot_count", "status", "created_at",
    "lag_close_1", "lag_close_5", "lag_close_10",
    "close_diff_1", "close_diff_5",
    "pct_change_1", "pct_change_5", "log_return_1",
    "sma_close_5", "sma_close_10", "sma_close_20", "sma_volume_5",
    "day_of_week", "hour_of_day", "month_of_year",
    "day_monday", "day_tuesday", "day_wednesday", "day_thursday", "day_friday",
    "quarter_1", "quarter_2", "quarter_3", "quarter_4",
    "hour_early_morning", "hour_mid_morning", "hour_afternoon", "hour_late_afternoon",
    "previous_close", "overnight_gap_pct", "overnight_log_return",
    "is_gap_up", "is_gap_down",
]

CHUNK = 500_000  # rows per COPY batch

def load_parquet(conn, df: pd.DataFrame):
    total = len(df)
    print(f"  Loading {total:,} rows in chunks of {CHUNK:,}…")
    cur = conn.cursor()
    loaded = 0
    t0 = time.time()
    for start in range(0, total, CHUNK):
        chunk = df.iloc[start:start + CHUNK][DW_COLS].copy()
        buf = io.StringIO()
        chunk.to_csv(buf, index=False, header=False, na_rep=r"\N")
        buf.seek(0)
        cur.copy_expert(
            "COPY dw.market_data_15m ("
            + ", ".join(DW_COLS)
            + ") FROM STDIN WITH (FORMAT csv, NULL '\\N')",
            buf,
        )
        loaded += len(chunk)
        elapsed = time.time() - t0
        print(f"  {loaded:,}/{total:,} rows  ({elapsed:.1f}s)", flush=True)
    conn.commit()
    cur.close()
    print(f"  dw.market_data_15m loaded in {time.time()-t0:.1f}s")


def seed_market_stocks(conn, symbols: list[str]):
    print(f"  Seeding market.stocks ({len(symbols)} symbols)…")
    cur = conn.cursor()
    for sym in symbols:
        cur.execute("""
            INSERT INTO market.stocks (symbol, name, sector, industry, exchange)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO NOTHING
        """, (sym, sym, "Unknown", "Unknown", "Unknown"))
    conn.commit()
    cur.close()
    print("  market.stocks done.")


def seed_ingestion_progress(conn, symbols: list[str]):
    print(f"  Seeding staging.ingestion_progress ({len(symbols)} symbols)…")
    cur = conn.cursor()
    for sym in symbols:
        cur.execute("""
            INSERT INTO staging.ingestion_progress (symbol, last_ts)
            VALUES (%s, NULL)
            ON CONFLICT (symbol) DO NOTHING
        """, (sym,))
    conn.commit()
    cur.close()
    print("  staging.ingestion_progress done.")


def main():
    print(f"Reading {PARQUET}…")
    df = pd.read_parquet(PARQUET)
    print(f"  Shape: {df.shape}")
    symbols = sorted(df["symbol"].unique().tolist())
    print(f"  Symbols: {len(symbols)}")

    print(f"\nConnecting to DB {DB['host']}:{DB['port']}/{DB['dbname']}…")
    conn = psycopg2.connect(**DB)

    print("\n[1/3] Loading dw.market_data_15m…")
    load_parquet(conn, df)

    print("\n[2/3] Seeding market.stocks…")
    seed_market_stocks(conn, symbols)

    print("\n[3/3] Seeding staging.ingestion_progress…")
    seed_ingestion_progress(conn, symbols)

    conn.close()

    print("\nDone! Verifying row counts…")
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM dw.market_data_15m")
    dw_rows = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM market.stocks")
    stock_rows = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM staging.ingestion_progress")
    prog_rows = cur.fetchone()[0]
    conn.close()
    print(f"  dw.market_data_15m:        {dw_rows:,}")
    print(f"  market.stocks:             {stock_rows:,}")
    print(f"  staging.ingestion_progress:{prog_rows:,}")


if __name__ == "__main__":
    main()
