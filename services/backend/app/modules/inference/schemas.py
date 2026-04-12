"""
Pydantic schemas for inference API.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    """Response model for stock price prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "current_price": 135.22,
                "predicted_price": 136.45,
                "predicted_return": 0.91,
                "prediction_date": "2026-02-12T00:00:00Z",
                "confidence": 0.65,
                "model_version": "xgboost-v1-20260211",
            }
        }
    )

    symbol: str
    current_price: float
    predicted_price: float
    predicted_return: float  # percentage change
    prediction_date: datetime
    confidence: float | None = None
    model_version: str


class NextDayBarPrediction(BaseModel):
    """Single 15-min bar in a next-day path prediction."""

    bar_idx: int    # 0–25
    bar_time: str   # "09:30", "09:45", …
    pred_close: float


class NextDayPredictionResponse(BaseModel):
    """Response model for next-day 15-min path prediction."""

    symbol: str
    current_price: float
    prediction_date: datetime           # next trading session date
    predicted_full_day_return: float    # percentage, bar 25 vs current_price
    predicted_direction: str            # "up" | "down"
    path: list[NextDayBarPrediction]    # 26 bars
    model_version: str
