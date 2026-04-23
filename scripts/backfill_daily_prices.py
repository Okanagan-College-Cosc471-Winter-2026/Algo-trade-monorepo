"""
Replace synthetic market.daily_prices with real FMP historical daily OHLC.

Steps:
  1. Truncate existing synthetic rows
  2. Fetch /api/v3/historical-price-full/{symbol} for every symbol in market.stocks
  3. Bulk-insert into market.daily_prices

Usage:
    python scripts/backfill_daily_prices.py
    python scripts/backfill_daily_prices.py --from-date 2024-01-01
"""

import argparse
import os
import time
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import execute_values
import requests

FMP_API_KEY = os.getenv("FMP_API_KEY", "7iSiCJecOuzJYx5xQr61Xd0f8NgNOnsU")
FMP_DELAY   = float(os.getenv("FMP_API_DELAY_SECONDS", "0.35"))
FMP_URL     = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5433")),
    dbname=os.getenv("DB_NAME", "algotrade"),
    user=os.getenv("DB_USER", "appuser"),
    password=os.getenv("DB_PASSWORD", "changeme"),
)

BATCH = 500  # rows per INSERT


def fetch_daily(symbol: str, from_date: str, to_date: str) -> list[dict]:
    try:
        r = requests.get(
            FMP_URL.format(symbol=symbol),
            params={"from": from_date, "to": to_date, "apikey": FMP_API_KEY},
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("historical", [])
    except Exception as e:
        print(f"  WARN [{symbol}] {e}")
        return []


def insert_batch(cur, rows: list[tuple]) -> int:
    execute_values(
        cur,
        """
        INSERT INTO market.daily_prices
            (symbol, date, open, high, low, close, volume, previous_close, change, change_pct)
        VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            open           = EXCLUDED.open,
            high           = EXCLUDED.high,
            low            = EXCLUDED.low,
            close          = EXCLUDED.close,
            volume         = EXCLUDED.volume,
            previous_close = EXCLUDED.previous_close,
            change         = EXCLUDED.change,
            change_pct     = EXCLUDED.change_pct
        """,
        rows,
    )
    return cur.rowcount


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default="2024-01-01",
                        help="Start date YYYY-MM-DD (default 2024-01-01)")
    args = parser.parse_args()

    to_date   = date.today().isoformat()
    from_date = args.from_date

    conn = psycopg2.connect(**DB)

    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM market.stocks ORDER BY symbol")
        symbols = [r[0] for r in cur.fetchall()]

    print(f"Symbols: {len(symbols)}")
    print(f"Date range: {from_date} → {to_date}")
    print(f"Estimated time: ~{len(symbols) * FMP_DELAY / 60:.1f} min\n")

    # Wipe synthetic rows
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE market.daily_prices")
    conn.commit()
    print("Truncated market.daily_prices\n")

    total_rows = 0
    skipped = []

    for i, symbol in enumerate(symbols, 1):
        rows_raw = fetch_daily(symbol, from_date, to_date)
        time.sleep(FMP_DELAY)

        if not rows_raw:
            skipped.append(symbol)
            print(f"[{i:3d}/{len(symbols)}] {symbol:8s}  NO DATA")
            continue

        rows = []
        for r in rows_raw:
            try:
                close  = float(r["close"])
                change = float(r.get("change", 0))
                rows.append((
                    symbol,
                    r["date"],
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    close,
                    int(r.get("volume") or 0),
                    round(close - change, 6),   # previous_close
                    round(change, 6),
                    round(float(r.get("changePercent", 0)), 4),
                ))
            except (KeyError, TypeError, ValueError):
                continue

        if not rows:
            skipped.append(symbol)
            print(f"[{i:3d}/{len(symbols)}] {symbol:8s}  PARSE ERROR")
            continue

        with conn.cursor() as cur:
            for start in range(0, len(rows), BATCH):
                insert_batch(cur, rows[start:start + BATCH])
        conn.commit()
        total_rows += len(rows)
        print(f"[{i:3d}/{len(symbols)}] {symbol:8s}  {len(rows)} days")

    conn.close()
    print(f"\nDone. {total_rows:,} rows inserted across {len(symbols) - len(skipped)} symbols.")
    if skipped:
        print("Skipped:", ", ".join(skipped))


if __name__ == "__main__":
    main()
