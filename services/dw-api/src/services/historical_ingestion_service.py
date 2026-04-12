from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def insert_historical_market_data(
    db: AsyncSession,
    symbol: str,
    ts: datetime,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    asset_type: str = "stock",
    source: str = "historical_dump"
):
    """
    Inserts a single historical market data record into staging.market_data.
    VWAP is omitted (set to NULL) since it is not present in historical data.
    """
    query = text("""
        INSERT INTO staging.market_data_5m (symbol, ts, open, high, low, close, volume, asset_type, source)
        VALUES (:symbol, :ts, :open, :high, :low, :close, :volume, :asset_type, :source)
        ON CONFLICT (symbol, ts) DO NOTHING
    """)
    params = {
        "symbol": symbol,
        "ts": ts,
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "asset_type": asset_type,
        "source": source
    }
    try:
        result = await db.execute(query, params)
        await db.commit()
        logger.info(
            "Historical market data inserted for symbol=%s ts=%s (affected: %d)",
            symbol, ts, result.rowcount
        )
    except IntegrityError as e:
        await db.rollback()
        logger.warning("Integrity error on historical insert: %s", str(e))
        raise ValueError("Data validation failed – check required fields and formats")
    except Exception as e:
        await db.rollback()
        logger.exception("Unexpected error during historical insert")
        raise
