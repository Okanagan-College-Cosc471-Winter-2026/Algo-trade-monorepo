"""
Inference API endpoints.

Provides stock price prediction using trained XGBoost model.
The response shape depends on ACTIVE_MODEL:
  - stock_prediction_xgb_global  → PredictionResponse
  - nextday_15m_path_final       → NextDayPredictionResponse
"""

from fastapi import APIRouter, HTTPException

from app.api.deps import SessionDep
from app.modules.inference.schemas import NextDayPredictionResponse
from app.modules.inference.service import InferenceService

router = APIRouter(prefix="/inference", tags=["inference"])


@router.get("/predict/{symbol}", response_model=NextDayPredictionResponse)
def predict_stock_price(symbol: str, session: SessionDep) -> NextDayPredictionResponse:
    """
    Get next-day 26-bar 15-min path prediction for a stock.

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')

    Raises:
        404: Stock not found
        400: Insufficient data or missing features
        500: Model error
    """
    try:
        return InferenceService.predict_stock_price(session, symbol.upper())
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
