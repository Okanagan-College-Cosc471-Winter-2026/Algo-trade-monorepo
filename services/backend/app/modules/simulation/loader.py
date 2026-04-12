"""
Simulation loader — reads and caches pre-computed prediction CSVs and replay metadata.

Paths (relative to repo root):
  Base:       model_artifacts/current_base        → symlink → base_2026-04-07/
  Simulation: model_artifacts/current_simulation  → symlink → simulation_2026-04-07/

To switch to a new simulation date, update the symlinks:
  ln -sfn base_YYYY-MM-DD       model_artifacts/current_base
  ln -sfn simulation_YYYY-MM-DD model_artifacts/current_simulation

NOTE: The "base" view (no warm refresh) uses step_00 of the simulation
(1,157 base trees, 0 warm trees added yet).
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Repo root = 5 levels up from this file
# backend/app/modules/simulation/loader.py  → 4 parents → backend/  → repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]

_BASE_BUNDLE_PATH = _REPO_ROOT / "model_artifacts" / "current_base"
_REPLAY_PATH = _REPO_ROOT / "model_artifacts" / "current_simulation"
_STEPS = 26


class SimulationLoader:
    """
    Singleton loader for simulation artifacts.

    Loaded once at module import time; DataFrames are held in memory.
    The total memory footprint is modest — each predictions.csv is ~300 KB
    and there are 26 step files (~7.5 MB uncompressed in total).
    """

    def __init__(self) -> None:
        self.base_metadata: dict = {}
        self.sim_summary: dict = {"replay_date": "N/A", "warm_trees_per_step": 0, "steps_completed": 0, "steps": []}
        self._step_info: dict[int, dict] = {}
        self._base_df: pd.DataFrame = pd.DataFrame(columns=["symbol"])
        self._step_dfs: dict[int, pd.DataFrame] = {}

        try:
            # --- Base metadata ---
            with open(_BASE_BUNDLE_PATH / "metadata.json") as f:
                self.base_metadata = json.load(f)

            # --- Replay simulation summary (step timestamps, tree counts) ---
            with open(_REPLAY_PATH / "simulation_summary.json") as f:
                self.sim_summary = json.load(f)

            # Build a fast lookup: step_index → step info dict
            self._step_info = {
                s["step"]: s for s in self.sim_summary["steps"]
            }

            # --- Load step_00 as the "base" view ---
            self._base_df = self._load_step_csv(0)

            # --- Load all 26 step CSVs (lazy-ish: load at startup, cache in dict) ---
            for i in range(_STEPS):
                self._step_dfs[i] = self._load_step_csv(i)

            logger.info(
                "SimulationLoader ready | replay_date=%s | steps=%d | symbols=%d",
                self.sim_summary["replay_date"],
                self.sim_summary["steps_completed"],
                len(self._base_df["symbol"].unique()),
            )
        except Exception as e:
            logger.warning("Simulation artifacts not found, skipping load. Error: %s", e)


    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.base_metadata.get("model_id", "base_20260320")

    @property
    def effective_as_of_date(self) -> str:
        return self.base_metadata.get("effective_as_of_date", "2026-03-20")

    @property
    def replay_date(self) -> str:
        return self.sim_summary["replay_date"]

    @property
    def warm_trees_per_step(self) -> int:
        return self.sim_summary["warm_trees_per_step"]

    @property
    def step_count(self) -> int:
        return self.sim_summary["steps_completed"]

    def step_labels(self) -> list[str]:
        return [self._step_info.get(i, {}).get("slot_label", f"Step {i}") for i in range(self.step_count)]

    def get_base_row(self, symbol: str) -> pd.Series | None:
        """Return the step_00 row for a symbol (used as base-model view)."""
        df = self._base_df
        rows = df[df["symbol"] == symbol.upper()]
        if rows.empty:
            return None
        return rows.iloc[0]

    def get_step_row(self, symbol: str, step: int) -> tuple[pd.Series | None, dict]:
        """Return (predictions row, step_info dict) for a symbol at a given step."""
        df = self._step_dfs.get(step, pd.DataFrame(columns=["symbol"]))
        rows = df[df["symbol"] == symbol.upper()]
        row = rows.iloc[0] if not rows.empty else None
        return row, self._step_info.get(step, {})

    def available_symbols(self) -> list[str]:
        return sorted(self._base_df["symbol"].unique().tolist())

    def get_step_rankings(self, step: int) -> pd.DataFrame:
        """Return all symbols for a step, sorted by predicted_full_day_return descending."""
        df = self._step_dfs.get(step, pd.DataFrame())
        if df.empty:
            return df
        return df.sort_values("predicted_full_day_return", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_step_csv(step: int) -> pd.DataFrame:
        csv_path = _REPLAY_PATH / f"step_{step:02d}" / "predictions" / "predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
        return pd.read_csv(csv_path)


# Module-level singleton
simulation_loader = SimulationLoader()
