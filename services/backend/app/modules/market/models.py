"""
Market module ORM models.

Maps to the data warehouse star schema:
  - Stock    → dim_stock + dim_company (descriptive context)
  - DailyPrice → fact_market_metrics   (daily OHLC + volume)

Tables live in the ``market`` Postgres schema.
"""

from datetime import date as date_type

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Stock(Base):
    """
    Dimension table for tradable instruments.

    Mirrors dim_stock + dim_company from the data warehouse.
    Will eventually be populated by the DW pipeline; for now,
    seeded with sample data.
    """

    __tablename__ = "stocks"
    __table_args__ = {"schema": "market"}

    symbol: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    sector: Mapped[str | None] = mapped_column(nullable=True)
    industry: Mapped[str | None] = mapped_column(nullable=True)
    currency: Mapped[str] = mapped_column(default="USD")
    exchange: Mapped[str | None] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)


class DailyPrice(Base):
    """
    Fact table for daily market metrics.

    Mirrors the price columns of fact_market_metrics from the
    data warehouse. One row = one stock on one trading day.
    """

    __tablename__ = "daily_prices"
    __table_args__ = (
        UniqueConstraint("symbol", "date"),
        {"schema": "market"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("market.stocks.symbol"), index=True)
    date: Mapped[date_type] = mapped_column(index=True)
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    close: Mapped[float]
    volume: Mapped[int]
    previous_close: Mapped[float | None] = mapped_column(nullable=True)
    change: Mapped[float | None] = mapped_column(nullable=True)
    change_pct: Mapped[float | None] = mapped_column(nullable=True)
