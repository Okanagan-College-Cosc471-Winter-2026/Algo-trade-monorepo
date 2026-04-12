"""
TEST 01 — Stage 1: Raw Ingest → stg_raw.market_data
──────────────────────────────────────────────────────
Simulates what intraday_data_collection.py does: insert 5-min bars
directly into stg_raw.market_data.  No FMP API key needed.

What we're proving:
  - Rows can be inserted into stg_raw.market_data
  - The UNIQUE(symbol, ts) constraint rejects exact duplicates
  - Duplicate inserts (same symbol+ts) are handled without crashing
"""
import datetime as dt
from datetime import timezone

import pytest
from sqlalchemy import text

from conftest import TEST_SYMBOL, BARS_5MIN


@pytest.fixture(autouse=True)
def cleanup(engine):
    """Remove test rows before and after each test."""
    def _purge():
        with engine.begin() as conn:
            conn.execute(text(
                "DELETE FROM stg_raw.market_data WHERE symbol = :s AND source = 'test'"
            ), {"s": TEST_SYMBOL})
    _purge()
    yield
    _purge()


def insert_bar(conn, ts, open_, high, low, close, volume):
    conn.execute(text("""
        INSERT INTO stg_raw.market_data
            (symbol, ts, open, high, low, close, volume, asset_type, source)
        VALUES
            (:symbol, :ts, :open, :high, :low, :close, :volume, 'stock', 'test')
        ON CONFLICT (symbol, ts) DO NOTHING
    """), {
        "symbol": TEST_SYMBOL, "ts": ts,
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


class TestStgRawIngest:

    def test_insert_three_bars(self, engine, raw_conn):
        with engine.begin() as conn:
            for ts, o, h, l, c, v in BARS_5MIN:
                insert_bar(conn, ts, o, h, l, c, v)

        with raw_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM stg_raw.market_data "
                "WHERE symbol = %s AND source = 'test'", (TEST_SYMBOL,)
            )
            count = cur.fetchone()[0]
        assert count == 3, f"Expected 3 rows, got {count}"

    def test_duplicate_insert_is_ignored(self, engine, raw_conn):
        ts, o, h, l, c, v = BARS_5MIN[0]
        with engine.begin() as conn:
            insert_bar(conn, ts, o, h, l, c, v)
            insert_bar(conn, ts, o, h, l, c, v)  # exact duplicate

        with raw_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM stg_raw.market_data "
                "WHERE symbol = %s AND source = 'test'", (TEST_SYMBOL,)
            )
            count = cur.fetchone()[0]
        assert count == 1, f"Expected 1 row after duplicate insert, got {count}"

    def test_bar_values_stored_correctly(self, engine, raw_conn):
        ts, o, h, l, c, v = BARS_5MIN[0]
        with engine.begin() as conn:
            insert_bar(conn, ts, o, h, l, c, v)

        with raw_conn.cursor() as cur:
            cur.execute(
                "SELECT open, high, low, close, volume "
                "FROM stg_raw.market_data WHERE symbol = %s AND ts = %s",
                (TEST_SYMBOL, ts)
            )
            row = cur.fetchone()
        assert row is not None
        assert float(row[0]) == pytest.approx(o, rel=1e-4)
        assert float(row[1]) == pytest.approx(h, rel=1e-4)
        assert float(row[2]) == pytest.approx(l, rel=1e-4)
        assert float(row[3]) == pytest.approx(c, rel=1e-4)
        assert int(row[4]) == v
