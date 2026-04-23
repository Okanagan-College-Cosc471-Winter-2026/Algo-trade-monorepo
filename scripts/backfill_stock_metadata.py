"""
Backfill market.stocks with real metadata from FMP /api/v3/profile/{symbol}.

Updates: name, sector, industry, exchange, currency
Skips symbols where FMP returns no data or an error.

Usage:
    python scripts/backfill_stock_metadata.py [--dry-run]
"""

import os
import sys
import time
import argparse
import requests
import psycopg2
from psycopg2.extras import execute_values

FMP_API_KEY = os.getenv("FMP_API_KEY", "7iSiCJecOuzJYx5xQr61Xd0f8NgNOnsU")
FMP_DELAY   = float(os.getenv("FMP_API_DELAY_SECONDS", "0.35"))
FMP_PROFILE = "https://financialmodelingprep.com/api/v3/profile/{symbol}"

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5433")),
    dbname=os.getenv("DB_NAME", "algotrade"),
    user=os.getenv("DB_USER", "appuser"),
    password=os.getenv("DB_PASSWORD", "changeme"),
)


def get_symbols(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM market.stocks ORDER BY symbol")
        return [r[0] for r in cur.fetchall()]


def fetch_profile(symbol: str) -> dict | None:
    url = FMP_PROFILE.format(symbol=symbol)
    try:
        r = requests.get(url, params={"apikey": FMP_API_KEY}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data and isinstance(data, list):
            return data[0]
    except Exception as e:
        print(f"  WARN [{symbol}] fetch error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Fetch but do not write to DB")
    parser.add_argument("--expand", metavar="SYMBOLS", default="",
                        help="Comma-separated symbols to INSERT into market.stocks before updating")
    args = parser.parse_args()

    conn = psycopg2.connect(**DB)

    if args.expand:
        new_syms = [s.strip().upper() for s in args.expand.split(",") if s.strip()]
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO market.stocks (symbol, name, is_active) VALUES %s "
                "ON CONFLICT (symbol) DO NOTHING",
                [(s, s, True) for s in new_syms],
            )
        conn.commit()
        print(f"Inserted {len(new_syms)} new symbols (skipped duplicates)")

    symbols = get_symbols(conn)
    print(f"Found {len(symbols)} symbols to update")
    print(f"Estimated time: ~{len(symbols) * FMP_DELAY / 60:.1f} min at {FMP_DELAY}s delay\n")

    rows = []
    skipped = []

    for i, symbol in enumerate(symbols, 1):
        profile = fetch_profile(symbol)
        time.sleep(FMP_DELAY)

        if not profile:
            skipped.append(symbol)
            print(f"[{i:3d}/{len(symbols)}] {symbol:8s}  NO DATA")
            continue

        name     = profile.get("companyName") or symbol
        sector   = profile.get("sector") or "Unknown"
        industry = profile.get("industry") or "Unknown"
        exchange = profile.get("exchangeShortName") or "Unknown"
        currency = profile.get("currency") or "USD"

        rows.append((name, sector, industry, exchange, currency, symbol))
        print(f"[{i:3d}/{len(symbols)}] {symbol:8s}  {sector} / {industry[:40]}")

    print(f"\nFetched {len(rows)} profiles, skipped {len(skipped)}")
    if skipped:
        print("Skipped:", ", ".join(skipped))

    if args.dry_run:
        print("\n--dry-run: no DB writes.")
        conn.close()
        return

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            UPDATE market.stocks AS s SET
                name     = v.name,
                sector   = v.sector,
                industry = v.industry,
                exchange = v.exchange,
                currency = v.currency
            FROM (VALUES %s) AS v(name, sector, industry, exchange, currency, symbol)
            WHERE s.symbol = v.symbol
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s)",
        )
        updated = cur.rowcount
    conn.commit()
    conn.close()

    print(f"Updated {updated} rows in market.stocks")


if __name__ == "__main__":
    main()
