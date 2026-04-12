"""
Inference service for stock price predictions.
"""

import math
from datetime import timedelta

import pandas as pd

from sqlalchemy.orm import Session

from app.modules.inference.features import prepare_production_features
from app.modules.inference.model_loader import get_model_bundle
from app.modules.inference.schemas import (
    NextDayBarPrediction,
    NextDayPredictionResponse,
    # PredictionResponse,  # legacy single-horizon — commented out
)
from app.modules.market import crud

# 15-min bar start times for a standard 09:30–16:00 session (26 bars)
_BAR_TIMES = [
    f"{9 + (30 + i * 15) // 60:02d}:{(30 + i * 15) % 60:02d}" for i in range(26)
]


class InferenceService:
    """Service for making stock price predictions."""

    @staticmethod
    def predict_stock_price(session: Session, symbol: str) -> NextDayPredictionResponse:
        return InferenceService._predict_next_day_path(session, symbol)

    # ------------------------------------------------------------------
    # Legacy single-horizon predictor — commented out, bars-only now
    # ------------------------------------------------------------------

    # @staticmethod
    # def _predict_single(session: Session, symbol: str) -> PredictionResponse:
    #     stock = crud.get_stock(session, symbol)
    #     if stock is None:
    #         raise ValueError(f"Stock not found: {symbol}")
    #     ohlc_data = crud.get_ohlc(session, symbol, days=10)
    #     if len(ohlc_data) < 150:
    #         raise ValueError(
    #             f"Insufficient data for {symbol}. Need at least 150 bars, got {len(ohlc_data)}"
    #         )
    #     df = pd.DataFrame(
    #         [{"date": row.date, "open": row.open, "high": row.high,
    #           "low": row.low, "close": row.close, "volume": row.volume}
    #          for row in ohlc_data]
    #     )
    #     try:
    #         ticker_encoder = model_bundle.ticker_encoder
    #         if ticker_encoder is None:
    #             raise ValueError("Ticker encoder not loaded")
    #         features = prepare_features_for_prediction(
    #             df, symbol, ticker_encoder, model_bundle.feature_names
    #         )
    #     except Exception as e:
    #         raise ValueError(f"Error calculating features: {str(e)}")
    #     predicted_return = model_bundle.predict(features)
    #     current_price = float(df["close"].iloc[-1])
    #     predicted_price = current_price * (1 + predicted_return)
    #     last_date = df["date"].iloc[-1]
    #     if isinstance(last_date, str):
    #         last_date = pd.to_datetime(last_date)
    #     horizon = model_bundle.metadata.get("horizon", 78)
    #     prediction_date = last_date + timedelta(minutes=5 * horizon)
    #     split_date = model_bundle.metadata.get("split_date", "unknown")
    #     model_version = f"xgboost-v1-{split_date}"
    #     return PredictionResponse(
    #         symbol=symbol, current_price=current_price,
    #         predicted_price=predicted_price,
    #         predicted_return=predicted_return * 100,
    #         prediction_date=prediction_date, confidence=None,
    #         model_version=model_version,
    #     )

    # ------------------------------------------------------------------
    # Next-day 26-bar path predictor
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_next_day_path(session: Session, symbol: str) -> NextDayPredictionResponse:
        """
        Predict the full next-day 15-min price path using NextDayPathBundle.
        """
        # 1. Verify stock exists
        stock = crud.get_stock(session, symbol)
        if stock is None:
            raise ValueError(f"Stock not found: {symbol}")

        # 2. Load recent warehouse bars and reconstruct the production feature row.
        recent_bars = crud.get_recent_inference_bars(session, symbol)
        if not recent_bars:
            raise ValueError(f"No engineered inference features available for {symbol}")

        model_bundle = get_model_bundle()
        bars_df = pd.DataFrame(recent_bars)
        features = prepare_production_features(bars_df, model_bundle.feature_names)
        latest_bar = bars_df.sort_values("window_ts").iloc[-1]
        current_price = float(latest_bar["close"])

        # 3. Predict 26 log-returns
        log_returns: list[float] = model_bundle.predict(features)

        # 4. Convert log-returns to absolute prices
        path: list[NextDayBarPrediction] = [
            NextDayBarPrediction(
                bar_idx=i,
                bar_time=_BAR_TIMES[i],
                pred_close=current_price * math.exp(lr),
            )
            for i, lr in enumerate(log_returns)
        ]

        # 5. Full-day return from last bar vs current price
        final_price = path[-1].pred_close
        predicted_full_day_return = (final_price / current_price - 1) * 100

        # 6. Prediction date: next calendar day from the latest engineered bar.
        prediction_date = pd.to_datetime(latest_bar["window_ts"]) + timedelta(days=1)

        model_version = model_bundle.metadata.get("model_version", "nextday_15m_path_final")

        return NextDayPredictionResponse(
            symbol=symbol,
            current_price=current_price,
            prediction_date=prediction_date,
            predicted_full_day_return=predicted_full_day_return,
            predicted_direction="up" if predicted_full_day_return >= 0 else "down",
            path=path,
            model_version=model_version,
        )
