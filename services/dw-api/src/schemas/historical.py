from pydantic import BaseModel
from datetime import datetime

class HistoricalMarketDataItem(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    asset_type: str = "stock"
    source: str = "historical_dump"
