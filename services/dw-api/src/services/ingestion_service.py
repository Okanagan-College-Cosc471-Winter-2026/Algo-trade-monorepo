from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
import logging

logger = logging.getLogger(__name__)

async def insert_market_data(db: AsyncSession, item: dict):
    query = text("""
        INSERT INTO staging.market_data_5m
            (symbol, ts, open, high, low, close, volume, asset_type, source)
        VALUES
            (:symbol, :ts, :open, :high, :low, :close, :volume, 'realtime', 'dw-api')
        ON CONFLICT (symbol, ts) DO NOTHING
    """)
    progress_query = text("""
        INSERT INTO staging.ingestion_progress (symbol, last_ingested_ts)
        VALUES (:symbol, :ts)
        ON CONFLICT (symbol) DO UPDATE SET
            last_ingested_ts = GREATEST(staging.ingestion_progress.last_ingested_ts, EXCLUDED.last_ingested_ts),
            updated_at = CURRENT_TIMESTAMP
    """)
    try:
        await db.execute(query, item)
        await db.execute(progress_query, {"symbol": item["symbol"], "ts": item["ts"]})
        await db.commit()
        logger.info("Inserted market data for symbol=%s ts=%s", item["symbol"], item.get("ts"))
    except IntegrityError as e:
        await db.rollback()
        logger.warning("Integrity error: %s", str(e))
        raise ValueError("Data validation failed")
    except Exception:
        await db.rollback()
        logger.exception("Unexpected error during insert")
        raise
