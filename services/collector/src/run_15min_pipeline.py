#!/usr/bin/env python3
"""
15-Minute Pipeline Runner

Chains the three stages that must complete within every 15-min window:
  1. Collect  — fetch 5-min OHLCV bars from FMP → stg_raw.market_data
  2. Export   — deduplicate + validate → core_dbms.market_data_5m
  3. Aggregate — call dw.process_15min_window(window_ts)
                 → dw.market_data_15m (all 176 engineered features)

Intended to run every 15 minutes via cron during market hours.
The script exits cleanly (code 0) when invoked outside market hours.

Usage:
    python src/run_15min_pipeline.py

Environment variables (same as intraday_data_collection.py):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    MARKET_TZ, MARKET_OPEN, MARKET_CLOSE
    FMP_API_KEY, SYMBOLS
    LOG_DIR
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import sys
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from model.orm_db import get_engine, get_session_factory
from model.models import PipelineLog
from utils.scheduled_pipeline import export_staging_to_core
from utils.time_utils import is_market_open, parse_hhmm

load_dotenv()

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline_15m.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline_15m")


def align_to_15_minute(ts: dt.datetime) -> dt.datetime:
    """Floor timestamp to the nearest 15-minute boundary."""
    ts = ts.replace(second=0, microsecond=0)
    return ts - dt.timedelta(minutes=ts.minute % 15)


def build_engine():
    return get_engine(
        os.getenv("DB_HOST", "localhost"),
        int(os.getenv("DB_PORT", "5432")),
        os.getenv("DB_NAME", ""),
        os.getenv("DB_USER", ""),
        os.getenv("DB_PASSWORD", ""),
    )


def log_pipeline(session_factory: sessionmaker, stage: str, status: str, message: str | None = None) -> None:
    try:
        with session_factory() as session:
            session.add(PipelineLog(
                pipeline_stage=stage,
                status=status,
                created_at=dt.datetime.now(timezone.utc),
                message=message,
            ))
            session.commit()
    except Exception as exc:
        logger.warning("Could not write pipeline log for %s: %s", stage, exc)


def run_collect() -> bool:
    """Run intraday data collection. Returns True on success, False on error."""
    try:
        # Import here so that PYTHONPATH is already resolved
        import intraday_data_collection as collector
        collector.main()
        return True
    except SystemExit as exc:
        if exc.code == 0:
            # main() exits 0 when window is outside market hours or no data — that is fine
            logger.info("[collect] Exited cleanly (no data or outside window).")
            return True
        logger.error("[collect] Exited with code %s", exc.code)
        return False
    except Exception as exc:
        logger.error("[collect] Unexpected error: %s", exc, exc_info=True)
        return False


def run_export(session_factory: sessionmaker) -> bool:
    """Export validated rows from stg_raw → core_dbms.market_data_5m."""
    try:
        with session_factory() as session:
            summary = export_staging_to_core(session)
            session.commit()
        logger.info(
            "[export] processed=%d exported=%d duplicates=%d quality_errors=%d",
            summary.processed_rows,
            summary.exported_rows,
            summary.duplicate_rows,
            summary.quality_error_rows,
        )
        return True
    except Exception as exc:
        logger.error("[export] Failed: %s", exc, exc_info=True)
        return False


def run_aggregate(engine, window_ts: dt.datetime) -> bool:
    """Call dw.process_15min_window for the closed 15-min bar."""
    ts_str = window_ts.strftime("%Y-%m-%d %H:%M:%S+00")
    try:
        with engine.begin() as conn:
            conn.execute(
                text("CALL dw.process_15min_window(:window_ts)"),
                {"window_ts": window_ts},
            )
        logger.info("[aggregate] process_15min_window(%s) OK", ts_str)
        return True
    except Exception as exc:
        logger.error("[aggregate] process_15min_window(%s) failed: %s", ts_str, exc, exc_info=True)
        return False


def main() -> int:
    market_tz_str = os.getenv("MARKET_TZ", "America/New_York")
    market_open_str = os.getenv("MARKET_OPEN", "04:00")
    market_close_str = os.getenv("MARKET_CLOSE", "21:00")

    tz = ZoneInfo(market_tz_str)
    now_local = dt.datetime.now(tz)
    now_utc = dt.datetime.now(timezone.utc)

    try:
        open_time = parse_hhmm(market_open_str)
        close_time = parse_hhmm(market_close_str)
    except ValueError as exc:
        logger.error("Invalid market hours: %s", exc)
        return 1

    if not is_market_open(now_local, open_time, close_time):
        logger.info(
            "Outside market hours (%s–%s %s). Nothing to do.",
            market_open_str, market_close_str, market_tz_str,
        )
        return 0

    # The window that just closed (floor to 15 min, then step back one window)
    current_window = align_to_15_minute(now_utc) - dt.timedelta(minutes=15)

    logger.info("=" * 60)
    logger.info("15-min pipeline starting | window_ts=%s UTC", current_window.strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    try:
        engine = build_engine()
        session_factory = get_session_factory(engine)
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)
        return 1

    # ── Stage 1: Collect ──────────────────────────────────────────
    collect_ok = run_collect()
    log_pipeline(
        session_factory,
        "collect_15m",
        "success" if collect_ok else "failed",
        f"window={current_window.isoformat()}",
    )
    if not collect_ok:
        logger.error("Collection failed — aborting pipeline for this window.")
        engine.dispose()
        return 1

    # ── Stage 2: Export staging → core ────────────────────────────
    export_ok = run_export(session_factory)
    log_pipeline(
        session_factory,
        "export_15m",
        "success" if export_ok else "failed",
        f"window={current_window.isoformat()}",
    )
    if not export_ok:
        logger.error("Export failed — skipping aggregation.")
        engine.dispose()
        return 1

    # ── Stage 3: Aggregate + feature engineering ──────────────────
    agg_ok = run_aggregate(engine, current_window)
    log_pipeline(
        session_factory,
        "aggregate_15m",
        "success" if agg_ok else "failed",
        f"window={current_window.isoformat()}",
    )

    engine.dispose()

    if agg_ok:
        logger.info("Pipeline complete. window_ts=%s ready for inference.", current_window.strftime("%Y-%m-%d %H:%M"))
        return 0
    else:
        logger.error("Aggregation failed for window %s.", current_window.strftime("%Y-%m-%d %H:%M"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
