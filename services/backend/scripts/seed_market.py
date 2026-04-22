"""
Seed the market schema with sample data.

Creates:
  - market.stocks     → 5 sample stocks (mirrors dim_stock + dim_company)
  - market.daily_prices → ~2 years of daily OHLC per stock (mirrors fact_market_metrics)

Skips weekends and US market holidays for realistic trading calendars.

Public API:
  seed(engine=None)  — idempotent; safe to call on every startup.
                       Inserts stocks that are missing; only generates daily
                       prices when the table is empty (to avoid duplicate work).

Run standalone:  POSTGRES_SERVER=localhost python -m scripts.seed_market
"""

import sys
from pathlib import Path

# Ensure the backend directory is on the Python path so ``app`` is importable
# regardless of how the script is invoked (python scripts/seed_market.py or
# python -m scripts.seed_market).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from datetime import date, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from scripts.stock_config import STOCKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DAYS = 730  # ~2 years

# Fixed US holidays (month, day)
FIXED_HOLIDAYS = {(1, 1), (7, 4), (12, 25)}


# ---------------------------------------------------------------------------
# Holiday helpers
# ---------------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of weekday (0=Mon) in month/year."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of weekday (0=Mon) in month/year."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _build_holidays(start_year: int, end_year: int) -> set[date]:
    """Build a set of US market holidays for quick lookup."""
    holidays: set[date] = set()
    for y in range(start_year, end_year + 1):
        for m, d in FIXED_HOLIDAYS:
            holidays.add(date(y, m, d))
        holidays.add(_nth_weekday(y, 1, 0, 3))  # MLK Day
        holidays.add(_nth_weekday(y, 2, 0, 3))  # Presidents' Day
        holidays.add(_last_weekday(y, 5, 0))  # Memorial Day
        holidays.add(_nth_weekday(y, 9, 0, 1))  # Labor Day
        holidays.add(_nth_weekday(y, 11, 3, 4))  # Thanksgiving
        holidays.add(_easter(y) - timedelta(days=2))  # Good Friday
    return holidays


def _is_trading_day(d: date, holidays: set[date]) -> bool:
    """Return True if the date is a valid NYSE trading day."""
    return d.weekday() < 5 and d not in holidays


# ---------------------------------------------------------------------------
# OHLC generator
# ---------------------------------------------------------------------------


def _gen_daily_prices(
    symbol: str,
    start: date,
    end: date,
    start_price: float,
    holidays: set[date],
) -> list[dict]:
    """
    Generate synthetic daily OHLC data using a random walk.

    Each day:
      - close = previous close * (1 + drift + shock)
      - open  ≈ previous close ± small gap
      - high  = max(open, close) + random wick
      - low   = min(open, close) - random wick
      - change / change_pct derived from previous_close
    """
    rows: list[dict] = []
    current = start
    prev_close = start_price

    while current <= end:
        if not _is_trading_day(current, holidays):
            current += timedelta(days=1)
            continue

        # Random walk for close price
        shock = random.gauss(0, 0.015)  # ~1.5% daily std
        drift = 0.0003  # small upward drift
        close = max(1.0, prev_close * (1.0 + drift + shock))

        # Open = previous close ± overnight gap (0-0.5%)
        gap = prev_close * random.uniform(-0.005, 0.005)
        open_price = max(1.0, prev_close + gap)

        # Wicks extend beyond open/close
        body_high = max(open_price, close)
        body_low = min(open_price, close)
        wick = body_high * random.uniform(0.001, 0.008)
        high = body_high + wick
        low = max(0.01, body_low - body_high * random.uniform(0.001, 0.008))

        # Volume: baseline ± noise, higher on volatile days
        volatility_factor = abs(shock) / 0.015
        base_volume = 50_000_000
        volume = int(
            max(
                0,
                random.gauss(
                    base_volume * (1 + volatility_factor * 0.5),
                    base_volume * 0.25,
                ),
            )
        )

        change = round(close - prev_close, 6)
        change_pct = round((change / prev_close) * 100, 4) if prev_close else 0.0

        rows.append(
            {
                "symbol": symbol,
                "date": current,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
                "previous_close": round(prev_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
            }
        )

        prev_close = close
        current += timedelta(days=1)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def seed(engine: Engine | None = None) -> None:
    """
    Idempotent seed of market.stocks and market.daily_prices.

    - Inserts missing stocks (ON CONFLICT DO NOTHING on symbol PK).
    - Generates synthetic daily prices only when market.daily_prices is empty,
      so repeated calls on a live DB with real data are safe.

    Args:
        engine: SQLAlchemy Engine connected to the target database.
                When None, falls back to the app's default engine from
                app.core.config (legacy standalone usage).
    """
    if engine is None:
        from app.core.config import settings
        engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))

    end = date.today()
    start = end - timedelta(days=DAYS)
    holidays = _build_holidays(start.year, end.year)

    with engine.begin() as conn:
        # --- Seed market.stocks (idempotent) ---
        inserted = 0
        for stock in STOCKS:
            result = conn.execute(
                text("""
                    INSERT INTO market.stocks
                        (symbol, name, sector, industry, currency, exchange, is_active)
                    VALUES
                        (:symbol, :name, :sector, :industry, :currency, :exchange, :is_active)
                    ON CONFLICT (symbol) DO NOTHING
                """),
                {k: v for k, v in stock.items() if k != "start_price"},
            )
            inserted += result.rowcount
        print(f"market.stocks: {inserted} new rows inserted ({len(STOCKS) - inserted} already present)")

        # --- Seed market.daily_prices only when table is empty ---
        price_count = conn.execute(text("SELECT COUNT(*) FROM market.daily_prices")).scalar()
        if price_count and price_count > 0:
            print(f"market.daily_prices: {price_count} rows already present — skipping synthetic generation")
            return

        for stock in STOCKS:
            symbol = stock["symbol"]
            price = stock["start_price"]
            print(f"  {symbol} (${price:.2f}) ...", end=" ")
            rows = _gen_daily_prices(symbol, start, end, price, holidays)
            for row in rows:
                conn.execute(
                    text("""
                        INSERT INTO market.daily_prices
                            (symbol, date, open, high, low, close, volume,
                             previous_close, change, change_pct)
                        VALUES
                            (:symbol, :date, :open, :high, :low, :close, :volume,
                             :previous_close, :change, :change_pct)
                        ON CONFLICT (symbol, date) DO NOTHING
                    """),
                    row,
                )
            print(f"{len(rows)} trading days")

    print("seed_market: done.")


def main() -> None:
    """Standalone entry-point: truncates and re-seeds (dev/reset use only)."""
    from app.core.config import settings
    engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))

    end = date.today()
    start = end - timedelta(days=DAYS)
    holidays = _build_holidays(start.year, end.year)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE market.daily_prices"))
        conn.execute(text("DELETE FROM market.stocks"))

        for stock in STOCKS:
            conn.execute(
                text("""
                    INSERT INTO market.stocks
                        (symbol, name, sector, industry, currency, exchange, is_active)
                    VALUES
                        (:symbol, :name, :sector, :industry, :currency, :exchange, :is_active)
                """),
                {k: v for k, v in stock.items() if k != "start_price"},
            )
        print(f"Seeded {len(STOCKS)} stocks")

        for stock in STOCKS:
            symbol = stock["symbol"]
            price = stock["start_price"]
            print(f"  {symbol} (${price:.2f}) ...", end=" ")
            rows = _gen_daily_prices(symbol, start, end, price, holidays)
            for row in rows:
                conn.execute(
                    text("""
                        INSERT INTO market.daily_prices
                            (symbol, date, open, high, low, close, volume,
                             previous_close, change, change_pct)
                        VALUES
                            (:symbol, :date, :open, :high, :low, :close, :volume,
                             :previous_close, :change, :change_pct)
                    """),
                    row,
                )
            print(f"{len(rows)} trading days")

    print("\nDone.")


if __name__ == "__main__":
    main()
