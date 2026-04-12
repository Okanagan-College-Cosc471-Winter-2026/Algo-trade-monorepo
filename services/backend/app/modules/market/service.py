"""
Market module business logic.

Converts DB rows into chart-friendly response shapes.
"""

import calendar
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.modules.market import crud
from app.modules.market.schemas import OHLCRead, StockRead


class MarketService:
    """Stateless service layer for market data."""

    @staticmethod
    def list_stocks(session: Session) -> list[StockRead]:
        """Return all active stocks as API response objects."""
        stocks = crud.get_active_stocks(session)
        return [
            StockRead(
                symbol=s.symbol,
                name=s.name,
                sector=s.sector,
                industry=s.industry,
                exchange=s.exchange,
            )
            for s in stocks
        ]

    @staticmethod
    def get_ohlc(
        session: Session,
        symbol: str,
        days: int = 365,
    ) -> list[OHLCRead]:
        """
        Return 15-min OHLC bars for the last *days* calendar days.

        Timestamps are Unix seconds (UTC) matching the window_ts values in
        ml.market_data_15m — consistent with what the simulation endpoints return.
        """
        rows = crud.get_ohlc(session, symbol, days)

        return [
            OHLCRead(
                time=int(r.date.timestamp()),
                open=float(r.open),
                high=float(r.high),
                low=float(r.low),
                close=float(r.close),
                volume=int(round(float(r.volume))),
            )
            for r in rows
        ]
