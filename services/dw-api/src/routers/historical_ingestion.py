from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.schemas.historical import HistoricalMarketDataItem
from src.services.historical_ingestion_service import insert_historical_market_data
from src.database import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest/historical", tags=["historical-ingestion"])

@router.post("/single/")
async def ingest_historical_single(
    item: HistoricalMarketDataItem,
    db: AsyncSession = Depends(get_db)
):
    try:
        await insert_historical_market_data(
            db,
            item.symbol,
            item.ts,
            item.open,
            item.high,
            item.low,
            item.close,
            item.volume,
            item.asset_type,
            item.source
        )
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error ingesting historical market data")
        raise HTTPException(status_code=500, detail="Internal server error")
