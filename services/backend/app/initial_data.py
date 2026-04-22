"""
Startup bootstrap for the backend service.

Called by prestart.sh immediately after Alembic migrations complete.

Steps:
  1. Verify bridge views exist  (ml.market_data_15m, historical.market_data_5m)
  2. Seed market.stocks + market.daily_prices  (idempotent)
  3. Seed staging.ingestion_progress for tracked symbols  (idempotent)
  4. Validate reference data — raise loudly if market.stocks is still empty
"""

import logging
import sys

from sqlalchemy import create_engine, text

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bridge views that must exist for the backend to serve data.
_REQUIRED_VIEWS = [
    ("ml", "market_data_15m"),
    ("historical", "market_data_5m"),
]


def _ensure_schemas_and_views(conn) -> None:
    """Warn if bridge views created by db/init/06_bridge_views.sql are absent."""
    for schema, view in _REQUIRED_VIEWS:
        exists = conn.execute(
            text(f"SELECT to_regclass('{schema}.{view}') IS NOT NULL")
        ).scalar()
        if not exists:
            logger.warning(
                "Bridge view %s.%s not found — check db/init/06_bridge_views.sql. "
                "Inference and OHLC endpoints will return empty results until this is fixed.",
                schema,
                view,
            )
        else:
            logger.info("View %s.%s OK", schema, view)


def _seed_market(engine) -> None:
    """
    Idempotent seed of market.stocks and market.daily_prices.

    Delegates to scripts/seed_market.py::seed() which uses ON CONFLICT DO NOTHING.
    """
    # scripts/ lives one directory above app/ inside the backend package root.
    # Ensure it is on sys.path so the import resolves regardless of working dir.
    import os
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    scripts_dir = os.path.abspath(scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from seed_market import seed  # type: ignore[import]
    seed(engine)


def _seed_ingestion_progress(conn) -> None:
    """
    Insert tracked symbols into staging.ingestion_progress if absent.

    dw.process_15min_window() only processes symbols that have a row here,
    so populating this table is a prerequisite for the 15-min pipeline to
    write into dw.market_data_15m.
    """
    import os
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    scripts_dir = os.path.abspath(scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from stock_config import STOCKS  # type: ignore[import]

    inserted = 0
    for stock in STOCKS:
        result = conn.execute(
            text("""
                INSERT INTO staging.ingestion_progress (symbol)
                VALUES (:symbol)
                ON CONFLICT (symbol) DO NOTHING
            """),
            {"symbol": stock["symbol"]},
        )
        inserted += result.rowcount

    logger.info(
        "staging.ingestion_progress: %d new rows inserted (%d symbols total tracked)",
        inserted,
        len(STOCKS),
    )


def _validate_reference_data(conn) -> None:
    """
    Assert market.stocks is non-empty.

    Raises RuntimeError on failure so prestart.sh exits non-zero and
    Docker healthcheck marks the container as unhealthy.
    """
    count = conn.execute(text("SELECT COUNT(*) FROM market.stocks")).scalar()
    if not count:
        raise RuntimeError(
            "market.stocks is empty after bootstrap — backend cannot serve stock data. "
            "Check that seed_market ran without errors and that Alembic migrations "
            "created the market schema tables."
        )
    logger.info("market.stocks validated: %d rows present", count)


def init() -> None:
    engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))
    try:
        with engine.begin() as conn:
            _ensure_schemas_and_views(conn)
            _seed_ingestion_progress(conn)

        # seed_market needs its own transaction scope (it may open multiple)
        _seed_market(engine)

        with engine.connect() as conn:
            _validate_reference_data(conn)
    finally:
        engine.dispose()


def main() -> None:
    logger.info("Starting bootstrap (initial_data)")
    init()
    logger.info("Bootstrap complete")


if __name__ == "__main__":
    main()
