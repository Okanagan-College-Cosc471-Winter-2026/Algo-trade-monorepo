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
    LOG_DIR, INTRADAY_FRESHNESS_FILE (optional; defaults under repo logs/)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from model.orm_db import get_engine, get_session_factory
from model.models import PipelineLog
from utils.scheduled_pipeline import export_staging_to_core
from utils.exchange_calendar import compute_last_closed_rth_window_start_utc

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[3]

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


@dataclass(frozen=True)
class LatestBarGate:
    ok: bool
    hard_fail: bool
    message: str


def _allow_symbol_subset() -> bool:
    return os.environ.get("ALLOW_SYMBOL_SUBSET", "").strip().lower() in {"1", "true", "yes", "on"}


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


def check_latest_5m_bar(engine, window_ts: dt.datetime) -> LatestBarGate:
    """
    Verify the latest 5-minute bar for a 15-minute window exists in core table.

    Example:
      window_ts=13:30 UTC checks for 13:40 UTC bar.
    """
    latest_5m_ts = window_ts + dt.timedelta(minutes=10)
    try:
        with engine.connect() as conn:
            symbol_count = conn.execute(
                text(
                    """
                    SELECT COUNT(DISTINCT symbol)
                    FROM core_dbms.market_data_5m
                    WHERE ts = :bar_ts
                      AND asset_type = 'stock'
                    """
                ),
                {"bar_ts": latest_5m_ts},
            ).scalar()
            active_symbol_count = conn.execute(
                text("SELECT COUNT(*) FROM market.stocks WHERE is_active = true")
            ).scalar()
        count = int(symbol_count or 0)
        active_symbols = int(active_symbol_count or 0)
        latest_bar_str = latest_5m_ts.strftime("%Y-%m-%d %H:%M:%S+00")
        if count <= 0:
            message = f"window={window_ts.isoformat()} gate_reason=missing_latest_bar latest_bar={latest_bar_str}"
            logger.warning("[aggregate] missing latest bar ts=%s; waiting for closed window", latest_bar_str)
            return LatestBarGate(ok=False, hard_fail=False, message=message)

        raw_ratio = os.environ.get("MIN_SYMBOL_COVERAGE_RATIO", "0.95")
        try:
            coverage_ratio = float(raw_ratio)
        except ValueError:
            coverage_ratio = 0.95
            logger.warning("[aggregate] invalid MIN_SYMBOL_COVERAGE_RATIO=%r; defaulting to 0.95", raw_ratio)

        if _allow_symbol_subset() or coverage_ratio <= 0 or active_symbols <= 0:
            logger.info(
                "[aggregate] latest bar check passed ts=%s symbols=%d active=%d subset_override=%s",
                latest_bar_str,
                count,
                active_symbols,
                _allow_symbol_subset(),
            )
            return LatestBarGate(
                ok=True,
                hard_fail=False,
                message=(
                    f"window={window_ts.isoformat()} latest_bar={latest_bar_str} "
                    f"symbols={count} active={active_symbols}"
                ),
            )

        required_symbols = math.ceil(active_symbols * coverage_ratio)
        if count < required_symbols:
            message = (
                f"window={window_ts.isoformat()} gate_reason=insufficient_symbol_coverage "
                f"latest_bar={latest_bar_str} symbols={count} active={active_symbols} required>={required_symbols}"
            )
            logger.error(
                "[aggregate] insufficient symbol coverage ts=%s symbols=%d active=%d required>=%d ratio=%.3f",
                latest_bar_str,
                count,
                active_symbols,
                required_symbols,
                coverage_ratio,
            )
            return LatestBarGate(ok=False, hard_fail=True, message=message)

        logger.info(
            "[aggregate] latest bar check passed ts=%s symbols=%d active=%d required>=%d",
            latest_bar_str,
            count,
            active_symbols,
            required_symbols,
        )
        return LatestBarGate(
            ok=True,
            hard_fail=False,
            message=(
                f"window={window_ts.isoformat()} latest_bar={latest_bar_str} "
                f"symbols={count} active={active_symbols} required>={required_symbols}"
            ),
        )
    except Exception as exc:
        logger.error("[aggregate] latest bar check failed: %s", exc, exc_info=True)
        return LatestBarGate(
            ok=False,
            hard_fail=True,
            message=f"window={window_ts.isoformat()} gate_reason=latest_bar_check_failed error={exc}",
        )


def write_freshness_marker(window_ts: dt.datetime) -> None:
    """
    Persist latest successful 15-minute data window for downstream DAG gating.
    """
    marker_path = Path(
        os.getenv(
            "INTRADAY_FRESHNESS_FILE",
            str(_REPO_ROOT / "logs" / "intraday_data_freshness.json"),
        )
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latest_window_ts_utc": window_ts.isoformat(),
        "updated_at_utc": dt.datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    market_tz_str = os.getenv("MARKET_TZ", "America/New_York")
    current_window, session_status = compute_last_closed_rth_window_start_utc(
        now=dt.datetime.now(timezone.utc),
        market_tz_name=market_tz_str,
    )
    if current_window is None:
        logger.info(
            "Skipping pipeline gate_reason=%s is_trading_day=%s is_half_day=%s now_et=%s",
            session_status.reason,
            session_status.is_trading_day,
            session_status.is_half_day,
            session_status.now_et.isoformat(),
        )
        return 0

    logger.info("=" * 60)
    logger.info(
        "15-min pipeline starting | window_ts=%s UTC session_open=%s session_close=%s",
        current_window.strftime("%Y-%m-%d %H:%M"),
        session_status.session_open_et.isoformat() if session_status.session_open_et else "n/a",
        session_status.session_close_et.isoformat() if session_status.session_close_et else "n/a",
    )
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
    latest_bar_gate = check_latest_5m_bar(engine, current_window)
    if not latest_bar_gate.ok:
        log_pipeline(
            session_factory,
            "aggregate_15m",
            "failed" if latest_bar_gate.hard_fail else "warning",
            latest_bar_gate.message,
        )
        engine.dispose()
        return 1 if latest_bar_gate.hard_fail else 0

    agg_ok = run_aggregate(engine, current_window)
    log_pipeline(
        session_factory,
        "aggregate_15m",
        "success" if agg_ok else "failed",
        f"window={current_window.isoformat()}",
    )

    engine.dispose()

    if agg_ok:
        try:
            write_freshness_marker(current_window)
        except Exception as exc:
            logger.warning("Could not write freshness marker: %s", exc)
        logger.info("Pipeline complete. window_ts=%s ready for inference.", current_window.strftime("%Y-%m-%d %H:%M"))
        return 0
    else:
        logger.error("Aggregation failed for window %s.", current_window.strftime("%Y-%m-%d %H:%M"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
