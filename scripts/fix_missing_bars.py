from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_SRC = ROOT / "services" / "collector" / "src"

# Environment
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "algotrade"
DB_USER = "appuser"
DB_PASSWORD = "changeme"

def get_engine():
    from sqlalchemy import create_engine
    url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

def step2_export() -> None:
    print("\nSTEP 2 — Export stg_raw -> core_dbms.market_data_5m")
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

def step3_aggregate(slots: list[dt.datetime]) -> None:
    print("\nSTEP 3 — Re-aggregate dw.market_data_15m")
    engine = get_engine()
    ok = 0
    with engine.begin() as conn:
        for ts in slots:
            try:
                conn.execute(text("CALL dw.process_15min_window(:ts)"), {"ts": ts})
                print(f"  Aggregated {ts.isoformat()}")
                ok += 1
            except Exception as exc:
                print(f"  WARNING: slot {ts.isoformat()} failed: {exc}")
    engine.dispose()
    print(f"STEP 3 OK — {ok}/{len(slots)} slots aggregated")

if __name__ == "__main__":
    # Missing slots in UTC
    MISSING_SLOTS = [
        dt.datetime(2026, 5, 1, 19, 45, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 5, 4, 15, 30, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 5, 4, 15, 45, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 5, 4, 19, 45, tzinfo=dt.timezone.utc),
    ]
    
    step2_export()
    step3_aggregate(MISSING_SLOTS)
