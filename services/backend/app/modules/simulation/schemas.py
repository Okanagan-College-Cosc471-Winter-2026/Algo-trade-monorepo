"""
Pydantic schemas for the simulation/demo endpoints.
"""

from pydantic import BaseModel


class SimBar(BaseModel):
    """One 15-min bar's prediction from either the base or a warm-refresh step model."""

    bar_idx: int          # 0–25
    bar_time: str         # "09:30", "09:45", … "15:45"
    pred_log_return: float
    pred_close: float | None = None   # anchor_close * exp(log_return); None if no anchor


class SimBaseResponse(BaseModel):
    """
    Full-day prediction for a symbol using the base model (trained through effective_as_of_date).

    Uses step_00 predictions (1,157 base trees, no warm refresh) as the base view because
    the base bundle's predictions/ directory is empty.
    """

    symbol: str
    date: str                        # replay date, e.g. "2026-04-07"
    model_id: str                    # from base bundle metadata.json
    effective_as_of_date: str        # last date in base model training, e.g. "2026-04-06"
    bars: list[SimBar]               # 26 bars, h00–h25
    predicted_full_day_return: float  # % return at h25 vs anchor
    predicted_direction: str         # "up" | "down"


class SimStepResponse(BaseModel):
    """
    Prediction from the warm-refresh model at one 15-min bar (step 0–25).

    Each step appends 30 trees to the prior model state, trained on intraday data
    up to that bar.
    """

    symbol: str
    step: int                        # 0–25
    as_of_ts: str                    # e.g. "2026-04-07 09:30:00"
    slot_label: str                  # "09:30"
    base_trees: int                  # always 1157
    warm_trees_added: int            # cumulative warm trees added so far
    total_trees: int                 # base + warm
    bars: list[SimBar]               # 26 bars, h00–h25
    predicted_full_day_return: float
    predicted_direction: str


class SimSessionInfo(BaseModel):
    """
    Summary of the replay session — for the frontend to know step count/timestamps.
    Used by /simulation/session endpoint.
    """

    replay_date: str
    effective_as_of_date: str   # last date included in base model training (day before replay)
    steps_completed: int
    step_labels: list[str]   # ["09:30", "09:45", … "15:45"]
    warm_trees_per_step: int
    base_trees: int
