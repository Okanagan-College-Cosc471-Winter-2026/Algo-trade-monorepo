"""
Inference API endpoints.

Provides stock price prediction using trained XGBoost model.
The response shape depends on ACTIVE_MODEL:
  - stock_prediction_xgb_global  → PredictionResponse
  - nextday_15m_path_final       → NextDayPredictionResponse
"""

from fastapi import APIRouter, HTTPException

from app.api.deps import SessionDep
from app.modules.inference.model_loader import get_base_eod_bundle, get_model_bundle
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


@router.get("/predict-base/{symbol}", response_model=NextDayPredictionResponse)
def predict_base_stock_price(symbol: str, session: SessionDep) -> NextDayPredictionResponse:
    """Prediction from the EOD base model (trained at yesterday's market close, static all day)."""
    try:
        return InferenceService.predict_stock_price(session, symbol.upper(), bundle=get_base_eod_bundle())
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/admin/reload-base-model", tags=["admin"])
def reload_base_model() -> dict:
    """Clear the EOD base model cache and reload from disk."""
    try:
        get_base_eod_bundle.cache_clear()
        bundle = get_base_eod_bundle()
        model_version = bundle.metadata.get("model_id", "unknown") if hasattr(bundle, "metadata") else "unknown"
        return {"status": "reloaded", "model_version": model_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@router.post("/admin/reload-model", tags=["admin"])
def reload_model() -> dict:
    """
    Clear the in-process model cache and reload the bundle from disk.

    Call this after promoting a new warm-refresh bundle via promote_model.sh.
    The next request to /predict will load the updated model.
    """
    try:
        get_model_bundle.cache_clear()
        bundle = get_model_bundle()
        model_version = bundle.metadata.get("model_id", "unknown") if hasattr(bundle, "metadata") else "unknown"
        return {"status": "reloaded", "model_version": model_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")
