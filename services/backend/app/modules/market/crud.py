"""Market module database queries.

Stock metadata is sourced from market.stocks (seeded via scripts/seed_market.py).
OHLC data is sourced from ml.market_data_15m (the live engineered feature table).
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class StockRecord:
    symbol: str
    name: str
    sector: str | None
    industry: str | None
    exchange: str | None


def get_active_stocks(session: Session) -> list[StockRecord]:
    """Return all active stocks from market.stocks, ordered by symbol."""
    rows = session.execute(
        text(
            """
            SELECT symbol, name, sector, industry, exchange
            FROM market.stocks
            WHERE is_active = TRUE
            ORDER BY symbol
            """
        )
    ).fetchall()
    return [
        StockRecord(
            symbol=row.symbol,
            name=row.name or row.symbol,
            sector=row.sector,
            industry=row.industry,
            exchange=row.exchange,
        )
        for row in rows
    ]


def get_stock(session: Session, symbol: str) -> StockRecord | None:
    """Return a single stock by symbol from market.stocks, or None."""
    row = session.execute(
        text(
            """
            SELECT symbol, name, sector, industry, exchange
            FROM market.stocks
            WHERE symbol = :symbol
            LIMIT 1
            """
        ),
        {"symbol": symbol.upper()},
    ).fetchone()
    if row is None:
        return None
    return StockRecord(
        symbol=row.symbol,
        name=row.name or row.symbol,
        sector=row.sector,
        industry=row.industry,
        exchange=row.exchange,
    )


def get_daily_prices(
    session: Session,
    symbol: str,
    start: date,
    end: date,
) -> list[Any]:
    """Return OHLC rows for a symbol within a date range."""
    return list(
        session.execute(
            text(
                """
                SELECT
                    d.datetime AS date,
                    f.open_price::numeric AS open,
                    f.high_price::numeric AS high,
                    f.low_price::numeric AS low,
                    f.close_price::numeric AS close,
                    f.volume::numeric AS volume
                FROM dw.fact_15min_stock_price f
                JOIN dw.dim_date d ON f.fk_date_id = d.sk_date_id
                JOIN dw.dim_instrument i ON f.fk_instrument_id = i.sk_instrument_id
                WHERE i.symbol = :symbol
                  AND d.date >= :start
                  AND d.date <= :end
                ORDER BY d.datetime ASC
                """
            ),
            {"symbol": symbol.upper(), "start": start, "end": end},
        ).fetchall()
    )


def get_coverage(session: Session, symbol: str) -> dict[str, Any]:
    """Return earliest date, latest date, and row count for a symbol."""
    row = session.execute(
        text(
            """
            SELECT
                MIN(trade_date)::date,
                MAX(trade_date)::date,
                COUNT(*)
            FROM ml.market_data_15m
            WHERE symbol = :symbol
            """
        ),
        {"symbol": symbol.upper()},
    ).one()
    return {"data_from": row[0], "data_to": row[1], "rows": row[2]}


def get_bars_for_features(session: Session, symbol: str, trading_days: int = 75) -> list[Any]:
    """
    Return the most recent `trading_days` worth of 15-min bars for feature engineering,
    including trade_count and vwap.  Bars are returned oldest-first.
    """
    results = session.execute(
        text(
            """
            SELECT date, open, high, low, close, volume, trade_count, vwap FROM (
                SELECT
                    d.datetime AS date,
                    f.open_price::float AS open,
                    f.high_price::float AS high,
                    f.low_price::float AS low,
                    f.close_price::float AS close,
                    f.volume::float AS volume,
                    COALESCE(f.trade_count, 0)::float AS trade_count,
                    COALESCE(f.vwap, f.close_price)::float AS vwap
                FROM dw.fact_15min_stock_price f
                JOIN dw.dim_date d ON f.fk_date_id = d.sk_date_id
                JOIN dw.dim_instrument i ON f.fk_instrument_id = i.sk_instrument_id
                WHERE i.symbol = :symbol
                ORDER BY d.datetime DESC
                LIMIT :limit
            ) sub
            ORDER BY date ASC
            """
        ),
        {"symbol": symbol.upper(), "limit": trading_days * 55},
    ).fetchall()
    return list(results)


def get_recent_inference_bars(
    session: Session,
    symbol: str,
    trading_days: int = 75,
) -> list[Any]:
    """
    Return recent engineered `ml.market_data_15m` rows for a symbol, oldest first.

    These rows contain the base engineered columns needed to reconstruct the full
    production inference feature vector expected by the active next-day model.
    """
    results = session.execute(
        text(
            """
            SELECT * FROM (
                SELECT
                    symbol,
                    trade_date,
                    window_ts,
                    open::float AS open,
                    high::float AS high,
                    low::float AS low,
                    close::float AS close,
                    volume::float AS volume,
                    slot_count::float AS slot_count,
                    lag_close_1::float AS lag_close_1,
                    lag_close_5::float AS lag_close_5,
                    lag_close_10::float AS lag_close_10,
                    close_diff_1::float AS close_diff_1,
                    close_diff_5::float AS close_diff_5,
                    pct_change_1::float AS pct_change_1,
                    pct_change_5::float AS pct_change_5,
                    log_return_1::float AS log_return_1,
                    sma_close_5::float AS sma_close_5,
                    sma_close_10::float AS sma_close_10,
                    sma_close_20::float AS sma_close_20,
                    sma_volume_5::float AS sma_volume_5,
                    day_of_week::float AS day_of_week,
                    hour_of_day::float AS hour_of_day,
                    day_monday::float AS day_monday,
                    day_tuesday::float AS day_tuesday,
                    day_wednesday::float AS day_wednesday,
                    day_thursday::float AS day_thursday,
                    day_friday::float AS day_friday,
                    quarter_1::float AS quarter_1,
                    quarter_2::float AS quarter_2,
                    quarter_3::float AS quarter_3,
                    quarter_4::float AS quarter_4,
                    hour_early_morning::float AS hour_early_morning,
                    hour_mid_morning::float AS hour_mid_morning,
                    hour_afternoon::float AS hour_afternoon,
                    hour_late_afternoon::float AS hour_late_afternoon,
                    previous_close::float AS previous_close,
                    overnight_gap_pct::float AS overnight_gap_pct,
                    overnight_log_return::float AS overnight_log_return,
                    is_gap_up::float AS is_gap_up,
                    is_gap_down::float AS is_gap_down,
                    month_of_year::float AS month_of_year
                FROM ml.market_data_15m
                WHERE symbol = :symbol
                ORDER BY window_ts DESC
                LIMIT :limit
            ) sub
            ORDER BY window_ts ASC
            """
        ),
        {"symbol": symbol.upper(), "limit": trading_days * 26},
    ).mappings().fetchall()
    return list(results)


def get_ohlc(session: Session, symbol: str, days: int = 365) -> list[Any]:
    """
    Return recent 15-min OHLC data for a symbol from ml.market_data_15m, oldest first.

    Fetches the latest `days * 26` bars regardless of wall-clock date so stale
    datasets still return data.
    """
    results = session.execute(
        text(
            """
            SELECT window_ts AS date,
                   open::numeric, high::numeric, low::numeric,
                   close::numeric, volume
            FROM (
                SELECT window_ts, open, high, low, close, volume
                FROM ml.market_data_15m
                WHERE symbol = :symbol
                ORDER BY window_ts DESC
                LIMIT :limit
            ) sub
            ORDER BY date ASC
            """
        ),
        {"symbol": symbol.upper(), "limit": days * 26},
    ).fetchall()
    return list(results)
