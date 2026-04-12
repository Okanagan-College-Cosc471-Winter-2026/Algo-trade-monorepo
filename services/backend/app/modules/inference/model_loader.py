"""
Model loader for XGBoost stock prediction models.

Supports two bundle types:
  - LegacyXGBBundle (stock_prediction_xgb_global): single XGBRegressor → scalar return
  - NextDayPathBundle (nextday_15m_path_final): 26 native boosters → list of log-returns

Switch bundles by setting ACTIVE_MODEL in .env.
"""

from functools import lru_cache
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import pandas as pd
import xgboost

from app.core.config import settings

logger = logging.getLogger(__name__)


class BaseModelBundle(ABC):
    """Abstract base for all model bundles."""

    model_type: str
    feature_names: list[str]
    metadata: dict[str, Any]
    model_path: Path

    @abstractmethod
    def predict(self, feature_row: pd.DataFrame) -> Any:
        """Run inference on a single prepared feature row."""
        ...


class LegacyXGBBundle(BaseModelBundle):
    """Bundle for stock_prediction_xgb_global (single XGBRegressor, scalar output)."""

    model_type = "legacy_xgb"

    def __init__(self, path: Path) -> None:
        self.model_path = path

        metadata_file = path / "meta.json"
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata.get("feature_cols", [])

        model_file = path / "xgb_global.pkl"
        logger.info("Loading legacy XGBRegressor from %s …", model_file)
        self._model: xgboost.XGBRegressor = joblib.load(model_file)

        encoder_file = path / "encoder.pkl"
        self.ticker_encoder = joblib.load(encoder_file) if encoder_file.exists() else None

        logger.info(
            "LegacyXGBBundle loaded | horizon=%s | split_date=%s | features=%d | tickers=%s",
            self.metadata.get("horizon"),
            self.metadata.get("split_date"),
            len(self.feature_names),
            len(self.ticker_encoder.classes_) if self.ticker_encoder is not None else "n/a",
        )

    def predict(self, feature_row: pd.DataFrame) -> float:
        return float(self._model.predict(feature_row)[0])


class NextDayPathBundle(BaseModelBundle):
    """Bundle for nextday_15m_path_final (26 native boosters, list[float] output)."""

    model_type = "next_day_path"

    def __init__(self, path: Path) -> None:
        self.model_path = path

        feature_names_file = path / "feature_names.json"
        with open(feature_names_file) as f:
            self.feature_names = json.load(f)

        metadata_file = path / "metadata.json"
        with open(metadata_file) as f:
            self.metadata = json.load(f)

        manifest_file = path / "models" / "model_manifest.json"
        with open(manifest_file) as f:
            manifest = json.load(f)

        # The manifest stores absolute training-environment paths; use local filenames only.
        models_dir = path / "models"
        ordered_keys = sorted(manifest["models"].keys())  # target_h00 … target_h25
        self.boosters: list[xgboost.Booster] = []
        for key in ordered_keys:
            filename = Path(manifest["models"][key]).name  # e.g. "horizon_00.json"
            booster = xgboost.Booster()
            booster.load_model(str(models_dir / filename))
            self.boosters.append(booster)

        logger.info(
            "NextDayPathBundle loaded | horizons=%d | features=%d | path=%s",
            len(self.boosters),
            len(self.feature_names),
            path,
        )

    def predict(self, feature_row: pd.DataFrame) -> list[float]:
        """Return 26 log-return predictions, one per 15-min bar."""
        dmatrix = xgboost.DMatrix(feature_row)
        return [float(booster.predict(dmatrix)[0]) for booster in self.boosters]


def create_model_bundle(path: Path) -> BaseModelBundle:
    """Auto-detect and instantiate the correct bundle type."""
    if (path / "models" / "model_manifest.json").exists():
        return NextDayPathBundle(path)
    elif (path / "xgb_global.pkl").exists():
        return LegacyXGBBundle(path)
    else:
        raise ValueError(f"Unknown model bundle structure at {path}")


@lru_cache(maxsize=1)
def get_model_bundle() -> BaseModelBundle:
    """
    Lazily load the active model bundle.

    This keeps application imports safe in environments like CI where model
    artifacts may be absent, while preserving a singleton-style cache once the
    bundle is first requested by an inference path.
    """
    bundle_path = Path(__file__).resolve().parents[4] / "model_artifacts" / settings.ACTIVE_MODEL
    logger.info("Loading active model bundle from %s", bundle_path)
    return create_model_bundle(bundle_path)
