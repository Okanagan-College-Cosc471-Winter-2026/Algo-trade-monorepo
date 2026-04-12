from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
import logging

logger = logging.getLogger(__name__)

async def insert_market_data(db: AsyncSession, item: dict):
    query = text("""
        INSERT INTO staging.market_data (symbol, ts, open, high, low, close, volume, vwap)
        VALUES (:symbol, :ts, :open, :high, :low, :close, :volume, :vwap)
        ON CONFLICT (symbol, ts) DO NOTHING
    """)
    try:
        result = await db.execute(query, item)
        await db.commit()
        logger.info(
            "Market data processed for symbol=%s ts=%s (affected: %d)",
            item["symbol"], item.get("ts"), result.rowcount
        )
    except IntegrityError as e:
        await db.rollback()
        logger.warning("Integrity error on insert: %s", str(e))
        raise ValueError("Data validation failed – check required fields and formats")
    except Exception as e:
        await db.rollback()
        logger.exception("Unexpected error during insert")
        raise
