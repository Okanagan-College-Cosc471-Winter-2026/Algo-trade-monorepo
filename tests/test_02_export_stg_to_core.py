"""
TEST 02 — Stage 2: Export stg_raw → core_dbms.market_data_5m
──────────────────────────────────────────────────────────────
Exercises export_staging_to_core() directly:
  - Seeds 3 bars into stg_raw.market_data
  - Runs the export function
  - Confirms rows appear in core_dbms.market_data_5m
  - Confirms re-running (idempotency) does not crash or duplicate

Requires: collector src on PYTHONPATH
  export PYTHONPATH=/data/projects/Algo-trade-monorepo/services/collector/src
  or run via: docker compose exec scheduler pytest tests/
"""
import datetime as dt
import sys
from pathlib import Path
from datetime import timezone

import pytest
from sqlalchemy import text

# Make sure collector src is importable when running tests on host
COLLECTOR_SRC = Path(__file__).parents[1] / "services/collector/src"
if str(COLLECTOR_SRC) not in sys.path:
    sys.path.insert(0, str(COLLECTOR_SRC))

from conftest import TEST_SYMBOL, BARS_5MIN


def seed_stg_raw(engine):
    with engine.begin() as conn:
        for ts, o, h, l, c, v in BARS_5MIN:
            conn.execute(text("""
                INSERT INTO stg_raw.market_data
                    (symbol, ts, open, high, low, close, volume, asset_type, source)
                VALUES
                    (:symbol, :ts, :open, :high, :low, :close, :volume, 'stock', 'test')
                ON CONFLICT (symbol, ts) DO NOTHING
            """), {
                "symbol": TEST_SYMBOL, "ts": ts,
                "open": o, "high": h, "low": l, "close": c, "volume": v,
            })


def purge_test_rows(engine):
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM stg_raw.market_data WHERE symbol = :s AND source = 'test'"
        ), {"s": TEST_SYMBOL})
        conn.execute(text(
            "DELETE FROM core_dbms.market_data_5m WHERE symbol = :s"
        ), {"s": TEST_SYMBOL})
        conn.execute(text(
            "DELETE FROM operation_logs.pipeline_watermarks WHERE pipeline_name = 'export_stg_to_core'"
        ))


@pytest.fixture(autouse=True)
def cleanup(engine):
    purge_test_rows(engine)
    yield
    purge_test_rows(engine)


class TestExportStagingToCore:

    def test_export_moves_rows_to_core(self, session_factory, raw_conn):
        from utils.scheduled_pipeline import export_staging_to_core

        # Seed raw data
        seed_stg_raw(session_factory.kw.get("bind") or session_factory.bind)  # fallback
        # Use the engine from the session_factory
        with session_factory() as session:
            # seed via the session's connection
            for ts, o, h, l, c, v in BARS_5MIN:
                session.execute(text("""
                    INSERT INTO stg_raw.market_data
                        (symbol, ts, open, high, low, close, volume, asset_type, source)
                    VALUES
                        (:symbol, :ts, :open, :high, :low, :close, :volume, 'stock', 'test')
                    ON CONFLICT (symbol, ts) DO NOTHING
                """), {"symbol": TEST_SYMBOL, "ts": ts,
                       "open": o, "high": h, "low": l, "close": c, "volume": v})

            summary = export_staging_to_core(session)
            session.commit()

        assert summary.exported_rows >= 3, (
            f"Expected >=3 exported rows, got {summary.exported_rows}"
        )
        assert summary.quality_error_rows == 0

        with raw_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM core_dbms.market_data_5m WHERE symbol = %s",
                (TEST_SYMBOL,)
            )
            count = cur.fetchone()[0]
        assert count >= 3, f"Expected >=3 rows in core_dbms.market_data_5m, got {count}"

    def test_export_is_idempotent(self, engine, session_factory, raw_conn):
        """Re-running export on already-exported rows should not add duplicates."""
        from utils.scheduled_pipeline import export_staging_to_core

        for _ in range(2):
            with session_factory() as session:
                for ts, o, h, l, c, v in BARS_5MIN:
                    session.execute(text("""
                        INSERT INTO stg_raw.market_data
                            (symbol, ts, open, high, low, close, volume, asset_type, source)
                        VALUES
                            (:symbol, :ts, :open, :high, :low, :close, :volume, 'stock', 'test')
                        ON CONFLICT (symbol, ts) DO NOTHING
                    """), {"symbol": TEST_SYMBOL, "ts": ts,
                           "open": o, "high": h, "low": l, "close": c, "volume": v})
                export_staging_to_core(session)
                session.commit()

        with raw_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM core_dbms.market_data_5m WHERE symbol = %s",
                (TEST_SYMBOL,)
            )
            count = cur.fetchone()[0]
        # Should still be exactly 3 — no duplication from second run
        assert count == 3, f"Expected exactly 3 rows after idempotent export, got {count}"

    def test_bad_row_goes_to_quality_errors(self, session_factory, raw_conn):
        """A bar with close=0 should be rejected to data_quality_errors, not core."""
        from utils.scheduled_pipeline import export_staging_to_core

        bad_ts = dt.datetime(2026, 4, 7, 15, 0, 0, tzinfo=timezone.utc)
        with session_factory() as session:
            session.execute(text("""
                INSERT INTO stg_raw.market_data
                    (symbol, ts, open, high, low, close, volume, asset_type, source)
                VALUES
                    (:symbol, :ts, 0, 0, 0, 0, 0, 'stock', 'test')
                ON CONFLICT (symbol, ts) DO NOTHING
            """), {"symbol": TEST_SYMBOL, "ts": bad_ts})
            summary = export_staging_to_core(session)
            session.commit()

        assert summary.quality_error_rows >= 1, (
            "Expected at least 1 quality error row for close=0 bar"
        )
