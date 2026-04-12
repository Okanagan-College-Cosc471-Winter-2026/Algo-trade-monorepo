from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.schemas.market_data import MarketDataItem
from src.services.ingestion_service import insert_market_data
from src.database import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

@router.post("/market-data/")
async def ingest_single(item: MarketDataItem, db: AsyncSession = Depends(get_db)):
    try:
        await insert_market_data(db, item.model_dump())
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error ingesting market data")
        raise HTTPException(status_code=500, detail="Internal server error")
