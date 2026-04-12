from pydantic import BaseModel
from datetime import datetime

class MarketDataItem(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
