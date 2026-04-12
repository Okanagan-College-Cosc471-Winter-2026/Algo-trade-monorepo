"""
Market module API schemas.

Response shapes are designed to be consumed directly by
TradingView Lightweight Charts on the frontend.
"""

from datetime import date

from pydantic import BaseModel


class StockRead(BaseModel):
    """Stock metadata for the stock selector dropdown."""

    symbol: str
    name: str
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None


class CoverageRead(BaseModel):
    """Data availability summary for a single stock."""

    symbol: str
    data_from: date
    data_to: date
    rows: int
    gap_days: int


class OHLCRead(BaseModel):
    """
    Single daily OHLC bar for chart rendering.

    ``time`` is a Unix timestamp (seconds) matching the format
    that TradingView Lightweight Charts expects natively.
    """

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: int
