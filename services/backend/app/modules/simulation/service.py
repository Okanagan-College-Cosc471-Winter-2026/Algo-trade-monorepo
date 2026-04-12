"""
Simulation service — converts raw CSV rows into API response schemas.
"""

import math

from app.modules.simulation.loader import simulation_loader
from app.modules.simulation.schemas import (
    SimBar,
    SimBaseResponse,
    SimSessionInfo,
    SimStepResponse,
)

# 15-min bar start times: 09:30 … 15:45  (26 bars)
_BAR_TIMES = [
    f"{9 + (30 + i * 15) // 60:02d}:{(30 + i * 15) % 60:02d}" for i in range(26)
]

_LOG_RETURN_COLS = [f"pred_log_return_h{i:02d}" for i in range(26)]


def _build_bars(row, anchor_close: float | None = None) -> list[SimBar]:
    """Convert a prediction row's 26 log-return columns into SimBar objects."""
    bars: list[SimBar] = []
    for i, col in enumerate(_LOG_RETURN_COLS):
        log_ret = float(row[col])
        pred_close = (anchor_close * math.exp(log_ret)) if anchor_close is not None else None
        bars.append(
            SimBar(
                bar_idx=i,
                bar_time=_BAR_TIMES[i],
                pred_log_return=log_ret,
                pred_close=round(pred_close, 4) if pred_close is not None else None,
            )
        )
    return bars


class SimulationService:

    @staticmethod
    def base_predictions(symbol: str) -> SimBaseResponse:
        """
        Return the base model view for a symbol on the simulation replay date.
        Uses step_00 (1,157 base trees, no warm refresh) as the base view.
        """
        row = simulation_loader.get_base_row(symbol)
        if row is None:
            raise ValueError(f"Symbol not found in simulation data: {symbol}")

        bars = _build_bars(row)
        full_day_return = float(row["predicted_full_day_return"])

        return SimBaseResponse(
            symbol=symbol.upper(),
            date=simulation_loader.replay_date,
            model_id=simulation_loader.model_id,
            effective_as_of_date=simulation_loader.effective_as_of_date,
            bars=bars,
            predicted_full_day_return=round(full_day_return * 100, 4),  # convert to %
            predicted_direction=str(row["predicted_direction"]),
        )

    @staticmethod
    def step_prediction(symbol: str, step: int) -> SimStepResponse:
        """
        Return the warm-refresh model's prediction for a symbol at one 15-min step.
        """
        if step < 0 or step >= simulation_loader.step_count:
            raise ValueError(
                f"Step {step} out of range. Valid range: 0–{simulation_loader.step_count - 1}"
            )

        row, step_info = simulation_loader.get_step_row(symbol, step)
        if row is None:
            raise ValueError(f"Symbol not found in simulation data: {symbol}")

        bars = _build_bars(row)
        full_day_return = float(row["predicted_full_day_return"])
        base_trees: int = int(step_info["base_trees"])
        # Cumulative warm trees = (step+1) × warm_trees_per_step
        warm_added: int = (step + 1) * simulation_loader.warm_trees_per_step

        return SimStepResponse(
            symbol=symbol.upper(),
            step=step,
            as_of_ts=str(step_info["as_of_ts"]),
            slot_label=str(step_info["slot_label"]),
            base_trees=base_trees,
            warm_trees_added=warm_added,
            total_trees=int(step_info["total_trees_estimate"]),
            bars=bars,
            predicted_full_day_return=round(full_day_return * 100, 4),  # convert to %
            predicted_direction=str(row["predicted_direction"]),
        )

    @staticmethod
    def session_info() -> SimSessionInfo:
        """Return session-level metadata for the frontend to build its step UI."""
        return SimSessionInfo(
            replay_date=simulation_loader.replay_date,
            effective_as_of_date=simulation_loader.effective_as_of_date,
            steps_completed=simulation_loader.step_count,
            step_labels=simulation_loader.step_labels(),
            warm_trees_per_step=simulation_loader.warm_trees_per_step,
            base_trees=1157,
        )

    @staticmethod
    def step_rankings(step: int) -> list[dict]:
        """Return all symbols ranked by predicted_full_day_return for a given step."""
        if step < 0 or step >= simulation_loader.step_count:
            raise ValueError(f"Step {step} out of range. Valid range: 0–{simulation_loader.step_count - 1}")
        df = simulation_loader.get_step_rankings(step)
        step_info = simulation_loader._step_info.get(step, {})
        slot_label = step_info.get("slot_label", "")
        return [
            {
                "rank": i + 1,
                "symbol": str(row["symbol"]),
                "slot_label": slot_label,
                "predicted_full_day_return": round(float(row["predicted_full_day_return"]) * 100, 4),
                "predicted_direction": str(row["predicted_direction"]),
            }
            for i, row in df.iterrows()
        ]
