#!/usr/bin/env python3
"""
Backfill missing symbol data for 2026-04-22 to 2026-04-29.

The intraday pipeline ran with only 2-21 symbols during this window instead
of the full ~505. This script:
  1. Fetches 5-min bars for all active symbols via gather_past_data.py
  2. Re-exports stg_raw -> core_dbms.market_data_5m
  3. Re-runs dw.process_15min_window() for every affected 15-min slot

Run from repo root:
    python scripts/backfill_gap_apr22_29.py
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_SRC = ROOT / "services" / "collector" / "src"
PIPELINE_PYTHON = Path(sys.executable)  # use the same interpreter running this script

load_dotenv(ROOT / ".env")

# When running outside Docker, the DB is exposed on localhost:5433 (not db:5432)
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = os.getenv("DB_NAME", os.getenv("POSTGRES_DB", ""))
DB_USER = os.getenv("DB_USER", os.getenv("POSTGRES_USER", ""))
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", ""))

ET = ZoneInfo("America/New_York")

BACKFILL_FROM = "2026-04-22"
BACKFILL_TO   = "2026-04-29"

# Weekdays in the backfill range that actually traded
TRADING_DATES = [
    dt.date(2026, 4, 22),
    dt.date(2026, 4, 23),
    dt.date(2026, 4, 24),
    dt.date(2026, 4, 27),
    dt.date(2026, 4, 28),
    dt.date(2026, 4, 29),
]

# Market hours in ET — 04:00 to 20:00 (pre + RTH + AH), 15-min slots
MARKET_OPEN_ET  = dt.time(4, 0)
MARKET_CLOSE_ET = dt.time(20, 0)


def get_engine():
    from sqlalchemy import create_engine
    url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


def load_symbols() -> str:
    """Return comma-separated list of all active symbols from market.stocks."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT symbol FROM market.stocks WHERE is_active = true ORDER BY symbol")
        ).fetchall()
    engine.dispose()
    symbols = [r[0] for r in rows]
    print(f"[symbols] Loaded {len(symbols)} active symbols from market.stocks")
    return ",".join(symbols)


def step1_fetch(symbols: str) -> None:
    """Fetch 5-min bars for all symbols for the backfill range."""
    print(f"\n{'='*60}")
    print(f"STEP 1 — Fetch raw bars: {BACKFILL_FROM} → {BACKFILL_TO}")
    print(f"{'='*60}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(COLLECTOR_SRC)
    env.setdefault("DB_HOST",     DB_HOST)
    env.setdefault("DB_PORT",     str(DB_PORT))
    env.setdefault("DB_NAME",     DB_NAME)
    env.setdefault("DB_USER",     DB_USER)
    env.setdefault("DB_PASSWORD", DB_PASSWORD)

    r = subprocess.run(
        [
            str(PIPELINE_PYTHON),
            str(COLLECTOR_SRC / "gather_past_data.py"),
            "--from-date", BACKFILL_FROM,
            "--to-date",   BACKFILL_TO,
            "--symbols",   symbols,
        ],
        cwd=str(COLLECTOR_SRC),
        env=env,
    )
    if r.returncode != 0:
        print("STEP 1 FAILED", file=sys.stderr)
        sys.exit(1)
    print("STEP 1 OK")


def step2_export() -> None:
    """Export validated rows from stg_raw -> core_dbms.market_data_5m."""
    print(f"\n{'='*60}")
    print("STEP 2 — Export stg_raw -> core_dbms.market_data_5m")
    print(f"{'='*60}")
    sys.path.insert(0, str(COLLECTOR_SRC))
    from model.orm_db import get_engine as _get_engine, get_session_factory
    from utils.scheduled_pipeline import export_staging_to_core

    engine = _get_engine(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        summary = export_staging_to_core(session)
        session.commit()
    engine.dispose()
    print(
        f"STEP 2 OK — processed={summary.processed_rows} exported={summary.exported_rows} "
        f"duplicates={summary.duplicate_rows} quality_errors={summary.quality_error_rows}"
    )


def step3_aggregate() -> None:
    """Re-run dw.process_15min_window for every 15-min slot on trading dates."""
    print(f"\n{'='*60}")
    print("STEP 3 — Re-aggregate dw.market_data_15m for all affected slots")
    print(f"{'='*60}")
    engine = get_engine()
    slots = []
    for d in TRADING_DATES:
        slot = dt.datetime.combine(d, MARKET_OPEN_ET, tzinfo=ET).astimezone(dt.timezone.utc)
        close_utc = dt.datetime.combine(d, MARKET_CLOSE_ET, tzinfo=ET).astimezone(dt.timezone.utc)
        while slot < close_utc:
            slots.append(slot)
            slot += dt.timedelta(minutes=15)

    print(f"  Calling process_15min_window for {len(slots)} slots across {len(TRADING_DATES)} days...")
    ok = 0
    with engine.begin() as conn:
        for ts in slots:
            try:
                conn.execute(text("CALL dw.process_15min_window(:ts)"), {"ts": ts})
                ok += 1
            except Exception as exc:
                print(f"  WARNING: slot {ts.isoformat()} failed: {exc}", file=sys.stderr)
    engine.dispose()
    print(f"STEP 3 OK — {ok}/{len(slots)} slots aggregated")


def main() -> None:
    symbols = load_symbols()
    step1_fetch(symbols)
    step2_export()
    step3_aggregate()
    print(f"\nBackfill complete. All {len(TRADING_DATES)} trading days should now have full symbol coverage.")


if __name__ == "__main__":
    main()
