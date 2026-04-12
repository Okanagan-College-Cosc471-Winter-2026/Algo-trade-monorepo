"""
Simulation API endpoints.

Provides a replay/demo mode for 2026-04-07 using pre-computed model artifacts.

Endpoints:
  GET /simulation/session
      → session metadata (step count, timestamps, tree counts)

  GET /simulation/base/{symbol}
      → full 26-bar prediction from base model (trained through 2026-04-06)
         implemented as step_00 predictions (1,157 trees, no warm-refresh)

  GET /simulation/step/{symbol}/{step}
      → prediction from the warm-refreshed model at one 15-min bar (step 0–25)
         each step adds 30 trees trained on real intraday data up to that bar
"""

from fastapi import APIRouter, HTTPException

from app.api.deps import SessionDep
from app.modules.simulation.schemas import (
    SimBaseResponse,
    SimSessionInfo,
    SimStepResponse,
)
from app.modules.simulation.service import SimulationService

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.get("/symbols", response_model=list[str])
def get_simulation_symbols() -> list[str]:
    """
    Return the list of symbols available in the simulation (sourced from the loaded prediction CSVs).
    Used by the frontend to populate the simulation asset selector.
    """
    from app.modules.simulation.loader import simulation_loader
    return simulation_loader.available_symbols()


@router.get("/session", response_model=SimSessionInfo)
def get_session_info() -> SimSessionInfo:
    """
    Return replay session metadata.
    Use this to populate the frontend's step slider / button labels.
    """
    return SimulationService.session_info()


@router.get("/base/{symbol}", response_model=SimBaseResponse)
def get_base_prediction(symbol: str) -> SimBaseResponse:
    """
    Full-day 26-bar prediction for a symbol using the base model (trained through 2026-04-06).

    This uses step_00 of the warm-refresh replay (base 1,157 trees only — no warm refresh
    applied yet), which is equivalent to running the base model cold on 2026-04-07 open data.

    Args:
        symbol: Stock symbol (e.g. 'AAPL', 'MSFT', 'NVDA')
    """
    try:
        return SimulationService.base_predictions(symbol.upper())
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")


@router.get("/step/{symbol}/{step}", response_model=SimStepResponse)
def get_step_prediction(symbol: str, step: int) -> SimStepResponse:
    """
    Prediction from the warm-refreshed model at one specific 15-min bar (step 0–25).

    Each step appends 30 trees trained on intraday data observed up to that bar:
      step  0 → 09:30  (1,157 + 30 =  1,187 trees)
      step  1 → 09:45  (1,157 + 60 =  1,217 trees)
      …
      step 25 → 15:45  (1,157 + 780 = 1,937 trees)

    Args:
        symbol: Stock symbol (e.g. 'AAPL', 'MSFT', 'NVDA')
        step:   Bar index 0–25 (0 = 09:30, 25 = 15:45)
    """
    try:
        return SimulationService.step_prediction(symbol.upper(), step)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")


@router.get("/history/{symbol}", response_model=list[dict])
def get_simulation_history(symbol: str, session: SessionDep) -> list[dict]:
    """
    Return full 15-min OHLC bars for the 5 trading days leading up to and including
    the simulation date (2026-03-31 → 2026-04-07), regular session only.
    Used to render the week-in-context candlestick chart.
    """
    from sqlalchemy import text

    rows = session.execute(
        text(
            """
            SELECT
                window_ts,
                trade_date::text AS trade_date,
                open::float,
                high::float,
                low::float,
                close::float,
                volume::bigint
            FROM ml.market_data_15m
            WHERE symbol = :symbol
              AND trade_date BETWEEN '2026-03-31' AND '2026-04-07'
              AND window_ts AT TIME ZONE 'America/New_York' >= trade_date + TIME '09:30'
              AND window_ts AT TIME ZONE 'America/New_York' <= trade_date + TIME '15:45'
            ORDER BY window_ts ASC
            """
        ),
        {"symbol": symbol.upper()},
    ).fetchall()

    return [
        {
            "time": int(r.window_ts.timestamp()),
            "trade_date": r.trade_date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ]


@router.get("/ohlc/{symbol}", response_model=list[dict])
def get_simulation_ohlc(
    symbol: str, session: SessionDep
) -> list[dict]:
    """
    Fetch real 15-min OHLC bars from ml.market_data_15m for the simulation date (2026-04-07),
    regular session only (09:30–15:45 ET = 13:30–19:45 UTC).
    """
    from sqlalchemy import text

    rows = session.execute(
        text(
            """
            SELECT
                window_ts,
                open::float,
                high::float,
                low::float,
                close::float,
                volume::bigint
            FROM ml.market_data_15m
            WHERE symbol = :symbol
              AND trade_date = '2026-04-07'
              AND window_ts AT TIME ZONE 'America/New_York' >= '2026-04-07 09:30:00'
              AND window_ts AT TIME ZONE 'America/New_York' <= '2026-04-07 15:45:00'
            ORDER BY window_ts ASC
            """
        ),
        {"symbol": symbol.upper()},
    ).fetchall()

    return [
        {
            "time": int(r.window_ts.timestamp()),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": int(r.volume),
        }
        for r in rows
    ]
