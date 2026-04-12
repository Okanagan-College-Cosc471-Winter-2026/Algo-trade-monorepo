"""
Market API endpoints.

Provides stock metadata and daily OHLC data for chart rendering.
"""

from fastapi import APIRouter, HTTPException

from app.api.deps import SessionDep
from app.modules.market import crud
from app.modules.market.schemas import CoverageRead, OHLCRead, StockRead
from app.modules.market.service import MarketService

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/stocks", response_model=list[StockRead])
def list_stocks(session: SessionDep) -> list[StockRead]:
    """
    List all active stocks.

    Used by the frontend to populate the stock selector dropdown.
    Returns symbol, name, sector, industry, and exchange.
    """
    return MarketService.list_stocks(session)


@router.get("/stocks/{symbol}", response_model=StockRead)
def get_stock(symbol: str, session: SessionDep) -> StockRead:
    """
    Get metadata for a single stock.
    """
    stock = crud.get_stock(session, symbol)
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Stock not found: {symbol}")
    return StockRead(
        symbol=stock.symbol,
        name=stock.name,
        sector=stock.sector,
        industry=stock.industry,
        exchange=stock.exchange,
    )


@router.get("/coverage", response_model=list[CoverageRead])
def get_coverage(session: SessionDep) -> list[CoverageRead]:
    """
    Return data availability (first date, last date, row count, gap to today)
    for every active stock.
    """
    from datetime import date as today_date

    stocks = crud.get_active_stocks(session)
    today = today_date.today()
    result = []
    for stock in stocks:
        cov = crud.get_coverage(session, stock.symbol)
        result.append(
            CoverageRead(
                symbol=stock.symbol,
                data_from=cov["data_from"],
                data_to=cov["data_to"],
                rows=cov["rows"],
                gap_days=(today - cov["data_to"]).days if cov["data_to"] else 0,
            )
        )
    return result


@router.get("/stocks/{symbol}/ohlc", response_model=list[OHLCRead])
def get_ohlc(
    symbol: str,
    days: int = 365,
    session: SessionDep = None,  # type: ignore[assignment]
) -> list[OHLCRead]:
    """
    Get daily OHLC + volume data for a stock.

    Used by the frontend to render Lightweight Charts.
    ``time`` values are Unix timestamps (seconds, UTC midnight).

    Query params:
      - **days**: number of calendar days to look back (default 365)
    """
    stock = crud.get_stock(session, symbol)
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Stock not found: {symbol}")
    return MarketService.get_ohlc(session, symbol, days)
