from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost.core import XGBoostError


sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 300)

LOGGER = logging.getLogger("XG_boost_3")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DEFAULT_DB_URL = "postgresql+psycopg2://harshsaw:@localhost:15432/penguin_db"
DEFAULT_SOURCE_TABLE = "ml.market_data_15m"
BEST_FIXED_VARIANT = {
    "name": "best_fixed",
    "learning_rate": 0.015924834009065022,
    "max_depth": 4,
    "subsample": 0.849153687610374,
    "colsample_bytree": 0.7446024807258872,
    "min_child_weight": 7.800340477474306,
    "n_estimators": 1157,
    "reg_alpha": 0.9916893895440203,
    "reg_lambda": 3.983536629736281,
}
FAST_REFRESH_VARIANT = {
    "name": "fast_refresh_fixed",
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 6.0,
    "n_estimators": 400,
    "reg_alpha": 0.5,
    "reg_lambda": 2.5,
}

DEFAULT_PARAM_VARIANTS = [BEST_FIXED_VARIANT]

REGULAR_OPEN_MINUTE = 9 * 60 + 30
REGULAR_CLOSE_MINUTE = 16 * 60
EXPECTED_REGULAR_BARS = 26
TARGET_COLUMNS = [f"target_h{idx:02d}" for idx in range(EXPECTED_REGULAR_BARS)]
MAX_ABS_TARGET = 1.5
MAX_ANOMALY_DROP_FRACTION = 0.10
PRODUCTION_TARGET_DEFINITION = "log(next_regular_session_close_h / current_bar_close)"
REPLAY_TARGET_DEFINITION = "log(next_regular_session_close_h / current_bar_close)"

SUMMARY_IDENTIFIER_COLS = [
    "agg_id",
    "symbol",
    "trade_date",
    "bar_ts",
    "window_ts",
    "bar_time",
    "status",
    "next_trade_date",
]

NON_FEATURE_COLS = set(
    SUMMARY_IDENTIFIER_COLS
    + TARGET_COLUMNS
    + [f"next_close_h{idx:02d}" for idx in range(EXPECTED_REGULAR_BARS)]
    + ["cutoff_time"]
)


def default_output_dir() -> str:
    return str((PROJECT_ROOT / "artifacts" / "nextday_15m_path_v3").resolve())


def default_run_root() -> str:
    return str((PROJECT_ROOT / "artifacts" / "production_nextday").resolve())


def default_replay_root() -> str:
    return str((PROJECT_ROOT / "artifacts" / "replay_nextday").resolve())


def resolve_runtime_path(path_value: str | None, fallback: Path | None = None) -> Path:
    if path_value:
        path = Path(path_value)
        if path.is_absolute():
            return path.resolve()
        return (PROJECT_ROOT / path).resolve()
    if fallback is not None:
        return fallback.resolve()
    raise ValueError("A path value or fallback must be provided.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-oriented XGBoost next-day 15-minute path pipeline with fixed best settings."
    )
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "refresh", "predict", "cycle", "replay_intraday", "warm_refresh", "simulate_warm_refresh", "train", "infer", "full_run"],
        default="bootstrap",
    )
    parser.add_argument("--warm-trees", type=int, default=30, help="New trees to add per warm-start cycle.")
    parser.add_argument("--warm-refresh-days", type=int, default=1, help="Trading days of data for warm-start refresh.")
    parser.add_argument("--simulation-out", default=None, help="Output dir for simulate_warm_refresh (default: <run-root>/simulations/<replay-date>).")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL)
    parser.add_argument("--db-user", default=os.environ.get("PGUSER", "harshsaw"))
    parser.add_argument("--db-password", default=os.environ.get("PGPASSWORD", ""))
    parser.add_argument("--db-host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("PGPORT", "15432")))
    parser.add_argument("--db-name", default=os.environ.get("PGDATABASE", "penguin_db"))
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--source-parquet", default=None, help="Temporary local parquet source for ml.market_data_15m-compatible data.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--as-of-date", default=None, help="Origin date for inference. Defaults to training end date.")
    parser.add_argument("--as-of-ts", default=None, help="Exact cutoff timestamp for simulation, e.g. 2026-03-23T10:15:00.")
    parser.add_argument("--replay-date", default=None, help="Replay date for intraday simulation, YYYY-MM-DD.")
    parser.add_argument("--truth-date", default=None, help="Truth date for replay evaluation, YYYY-MM-DD.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--base-window-months", type=int, default=18)
    parser.add_argument("--refresh-days", "--fast-refresh-days", dest="fast_refresh_days", type=int, default=60)
    parser.add_argument("--daily-windows", type=int, nargs="+", default=[5, 10, 20, 60])
    parser.add_argument("--slot-windows", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--parallel-horizons", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--winsor-pct", type=float, default=0.005)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--test-block-days", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--report-days-back", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-profile", choices=["base", "fast_refresh", "both"], default="both")
    parser.add_argument("--train-policy", choices=["simple", "validated"], default="simple")
    parser.add_argument("--run-root", default=default_run_root())
    parser.add_argument("--refresh-budget-sec", type=int, default=780)
    parser.add_argument("--write-reports", action="store_true")
    parser.add_argument("--visual-symbol-count", type=int, default=10)
    parser.add_argument("--output-dir", default=None, help="Legacy compatibility path. Prefer --run-root.")
    parser.add_argument("--run-dir", default=None, help="Legacy compatibility path. Prefer --run-root.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def configure_logging(output_dir: Path, level_name: str, log_filename: str = "run.log") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(output_dir / log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def resolve_db_url(args: argparse.Namespace) -> str:
    if args.db_url:
        return args.db_url
    return (
        f"postgresql+psycopg2://{args.db_user}:{args.db_password}"
        f"@{args.db_host}:{args.db_port}/{args.db_name}"
    )


def resolve_source_parquet_path(args: argparse.Namespace) -> Path | None:
    if not getattr(args, "source_parquet", None):
        return None
    return resolve_runtime_path(args.source_parquet)


def describe_source(args: argparse.Namespace) -> str:
    parquet_path = resolve_source_parquet_path(args)
    if parquet_path is not None:
        return f"parquet:{parquet_path}"
    return resolve_db_url(args)


def resolve_xgb_params(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    requested = args.device
    auto_has_gpu = bool(os.environ.get("CUDA_VISIBLE_DEVICES")) and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
    if requested == "auto":
        device = "cuda" if auto_has_gpu else "cpu"
    else:
        device = requested
    if requested == "cuda" and not auto_has_gpu and not os.environ.get("NVIDIA_VISIBLE_DEVICES"):
        raise RuntimeError("CUDA was requested but no visible GPU was detected in the environment.")
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "learning_rate": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "n_estimators": 600,
        "random_state": args.random_state,
        "n_jobs": -1,
    }
    return params, device


def variant_params(base_params: dict[str, Any], variant_name: str) -> dict[str, Any]:
    params = base_params.copy()
    for variant in DEFAULT_PARAM_VARIANTS:
        if variant["name"] == variant_name:
            params.update({k: v for k, v in variant.items() if k != "name"})
            break
    return params


def fit_with_optional_fallback(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: dict[str, Any],
    allow_gpu_fallback: bool,
    sample_weight: np.ndarray | None = None,
    init_model_path: str | None = None,
) -> tuple[np.ndarray, XGBRegressor, dict[str, Any]]:
    active_params = params.copy()
    try:
        model = XGBRegressor(**active_params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False,
                  xgb_model=init_model_path)
        if active_params.get("device") == "cuda":
            preds = model.get_booster().predict(xgb.DMatrix(X_test))
        else:
            preds = model.predict(X_test)
        return preds, model, active_params
    except XGBoostError as exc:
        if active_params.get("device") != "cuda" or not allow_gpu_fallback:
            raise
        fallback_params = active_params.copy()
        fallback_params["device"] = "cpu"
        LOGGER.warning("GPU training failed, retrying on CPU: %s", exc)
        model = XGBRegressor(**fallback_params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False,
                  xgb_model=init_model_path)
        return model.predict(X_test), model, fallback_params


SOURCE_BASE_COLUMNS = [
    "agg_id",
    "symbol",
    "window_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "slot_count",
    "status",
    "lag_close_1",
    "lag_close_5",
    "lag_close_10",
    "close_diff_1",
    "close_diff_5",
    "pct_change_1",
    "pct_change_5",
    "log_return_1",
    "sma_close_5",
    "sma_close_10",
    "sma_close_20",
    "sma_volume_5",
    "day_of_week",
    "hour_of_day",
    "day_monday",
    "day_tuesday",
    "day_wednesday",
    "day_thursday",
    "day_friday",
    "quarter_1",
    "quarter_2",
    "quarter_3",
    "quarter_4",
    "hour_early_morning",
    "hour_mid_morning",
    "hour_afternoon",
    "hour_late_afternoon",
    "previous_close",
    "overnight_gap_pct",
    "overnight_log_return",
    "is_gap_up",
    "is_gap_down",
    "month_of_year",
]
REQUIRED_SOURCE_COLUMNS = [
    "symbol",
    "window_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "slot_count",
    "status",
]


def normalize_loaded_bars(bars: pd.DataFrame, source_name: str) -> pd.DataFrame:
    missing = [col for col in REQUIRED_SOURCE_COLUMNS if col not in bars.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Source {source_name} is missing required columns: {missing_str}")

    out = bars.copy()
    if "agg_id" not in out.columns:
        out["agg_id"] = np.arange(1, len(out) + 1, dtype=np.int64)
    out["window_ts"] = pd.to_datetime(out["window_ts"], errors="coerce")
    if out["window_ts"].isna().any():
        raise RuntimeError(f"Source {source_name} contains invalid window_ts values.")
    if "bar_ts" not in out.columns:
        out["bar_ts"] = out["window_ts"]
    else:
        out["bar_ts"] = pd.to_datetime(out["bar_ts"], errors="coerce")
        out["bar_ts"] = out["bar_ts"].fillna(out["window_ts"])
    if "trade_date" not in out.columns:
        out["trade_date"] = out["window_ts"]
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    if out["trade_date"].isna().any():
        out["trade_date"] = out["window_ts"].dt.normalize()
    ordered_cols = ["agg_id", "symbol", "window_ts", "bar_ts", "trade_date"] + [col for col in SOURCE_BASE_COLUMNS if col not in {"agg_id", "symbol", "window_ts"}]
    available_cols = [col for col in ordered_cols if col in out.columns]
    return out[available_cols].sort_values(["window_ts", "symbol"]).reset_index(drop=True)


def load_parquet_source(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise RuntimeError(f"Parquet source does not exist: {parquet_path}")
    bars = pd.read_parquet(parquet_path)
    return normalize_loaded_bars(bars, source_name=str(parquet_path))


def open_source(args: argparse.Namespace):
    parquet_path = resolve_source_parquet_path(args)
    if parquet_path is not None:
        LOGGER.info("Using parquet source %s", parquet_path)
        return load_parquet_source(parquet_path)

    db_url = resolve_db_url(args)
    LOGGER.info("Connecting to %s:%s/%s", args.db_host, args.db_port, args.db_name)
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def end_of_day(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def resolve_requested_as_of(engine, args: argparse.Namespace) -> pd.Timestamp:
    if args.as_of_ts:
        return pd.Timestamp(args.as_of_ts)
    if args.as_of_date:
        return end_of_day(pd.Timestamp(args.as_of_date))
    if args.end_date:
        return end_of_day(pd.Timestamp(args.end_date))
    minmax = load_minmax(engine, source_table=args.source_table)
    return end_of_day(pd.Timestamp(minmax.max_date.iloc[0]))


def clip_bars_to_requested_as_of(bars: pd.DataFrame, requested_as_of: pd.Timestamp) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    ts = requested_as_of if requested_as_of.tzinfo is not None else requested_as_of.tz_localize("UTC")
    clipped = bars[pd.to_datetime(bars["window_ts"], errors="coerce") <= ts].copy()
    return clipped.sort_values(["window_ts", "symbol"]).reset_index(drop=True)


def load_minmax(engine, source_table: str = DEFAULT_SOURCE_TABLE) -> pd.DataFrame:
    if isinstance(engine, pd.DataFrame):
        frame = engine.copy()
        if "status" in frame.columns:
            frame = frame[frame["status"] != "provisional"].copy()
        if frame.empty:
            raise RuntimeError("No non-provisional rows are available in the parquet source.")
        dates = pd.to_datetime(frame["window_ts"], errors="coerce").dt.normalize()
        return pd.DataFrame({"min_date": [dates.min()], "max_date": [dates.max()]})
    return pd.read_sql(
        text(
            f"""
            SELECT
                MIN(window_ts::date) AS min_date,
                MAX(window_ts::date) AS max_date
            FROM {source_table}
            WHERE status <> 'provisional'
            """
        ),
        engine,
    )


BASE_FACT_SQL = """
SELECT
    ROW_NUMBER() OVER (ORDER BY window_ts, symbol) AS agg_id,
    symbol,
    window_ts,
    window_ts AS bar_ts,
    window_ts::date AS trade_date,
    open,
    high,
    low,
    close,
    volume,
    slot_count,
    status,
    lag_close_1,
    lag_close_5,
    lag_close_10,
    close_diff_1,
    close_diff_5,
    pct_change_1,
    pct_change_5,
    log_return_1,
    sma_close_5,
    sma_close_10,
    sma_close_20,
    sma_volume_5,
    day_of_week,
    hour_of_day,
    day_monday,
    day_tuesday,
    day_wednesday,
    day_thursday,
    day_friday,
    quarter_1,
    quarter_2,
    quarter_3,
    quarter_4,
    hour_early_morning,
    hour_mid_morning,
    hour_afternoon,
    hour_late_afternoon,
    previous_close,
    overnight_gap_pct,
    overnight_log_return,
    is_gap_up,
    is_gap_down,
    month_of_year
FROM {source_table}
WHERE window_ts::date BETWEEN :start_date AND :end_date
  AND (CAST(:symbols AS text) IS NULL OR symbol = ANY(string_to_array(CAST(:symbols AS text), ',')))
"""


def load_bars(
    engine,
    start_date: str,
    end_date: str,
    symbols: list[str] | None = None,
    source_table: str = DEFAULT_SOURCE_TABLE,
    include_provisional: bool = False,
) -> pd.DataFrame:
    if isinstance(engine, pd.DataFrame):
        frame = engine.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        mask = frame["trade_date"].between(start_ts, end_ts)
        if symbols:
            mask &= frame["symbol"].isin(symbols)
        if not include_provisional and "status" in frame.columns:
            mask &= frame["status"] != "provisional"
        return frame.loc[mask].sort_values(["window_ts", "symbol"]).reset_index(drop=True)
    sql = BASE_FACT_SQL.format(source_table=source_table)
    if not include_provisional:
        sql += " AND status <> 'provisional'"
    sql += " ORDER BY window_ts, symbol"
    params: dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "symbols": ",".join(symbols) if symbols else None,
    }
    return pd.read_sql(text(sql), engine, params=params, parse_dates=["window_ts", "bar_ts", "trade_date"])


def minute_of_day(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def prepare_session_bars(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = bars.sort_values(["symbol", "trade_date", "bar_ts"]).copy()
    bars["minute_of_day"] = minute_of_day(bars["bar_ts"])
    bars["is_regular"] = (bars["minute_of_day"] >= REGULAR_OPEN_MINUTE) & (bars["minute_of_day"] < REGULAR_CLOSE_MINUTE)
    bars["is_premarket"] = bars["minute_of_day"] < REGULAR_OPEN_MINUTE

    regular = bars[bars["is_regular"]].copy()
    regular["slot_idx"] = regular.groupby(["symbol", "trade_date"]).cumcount()
    valid_sessions = (
        regular.groupby(["symbol", "trade_date"], observed=True)
        .size()
        .rename("regular_bar_count")
        .reset_index()
    )
    valid_sessions = valid_sessions[valid_sessions["regular_bar_count"] == EXPECTED_REGULAR_BARS][["symbol", "trade_date"]]
    if valid_sessions.empty:
        raise RuntimeError("No complete regular-hours sessions were found in the requested date range.")

    regular = regular.merge(valid_sessions, on=["symbol", "trade_date"], how="inner")
    regular["slot_idx"] = regular.groupby(["symbol", "trade_date"]).cumcount()
    premarket = bars[bars["is_premarket"]].merge(valid_sessions, on=["symbol", "trade_date"], how="inner")
    return regular.reset_index(drop=True), premarket.reset_index(drop=True)


def weighted_vwap(group: pd.DataFrame) -> float:
    total_volume = float(group["volume"].sum())
    if total_volume <= 0:
        return float(group["vwap"].mean())
    return float((group["vwap"] * group["volume"]).sum() / total_volume)


def pivot_slot_values(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    index_cols = ["symbol", "trade_date"]
    for col in value_cols:
        pivot = df.pivot(index=index_cols, columns="slot_idx", values=col)
        pivot.columns = [f"slot_{int(slot):02d}_{col}" for slot in pivot.columns]
        parts.append(pivot)
    if not parts:
        return pd.DataFrame(columns=index_cols)
    return pd.concat(parts, axis=1).reset_index()


def add_group_rolling_features(df: pd.DataFrame, group_col: str, value_cols: list[str], windows: list[int], suffix_template: str) -> pd.DataFrame:
    df = df.sort_values([group_col, "trade_date"]).copy()
    grouped = df.groupby(group_col, observed=True)
    feature_map: dict[str, pd.Series] = {}
    for window in windows:
        for col in value_cols:
            feature_map[suffix_template.format(col=col, window=window, stat="mean")] = grouped[col].transform(
                lambda s: s.rolling(window, min_periods=2).mean()
            )
            feature_map[suffix_template.format(col=col, window=window, stat="std")] = grouped[col].transform(
                lambda s: s.rolling(window, min_periods=2).std(ddof=0)
            )
    if not feature_map:
        return df
    return pd.concat([df, pd.DataFrame(feature_map, index=df.index)], axis=1).copy()


def add_group_rolling_features_by_keys(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    windows: list[int],
    suffix_template: str,
) -> pd.DataFrame:
    df = df.sort_values(group_cols + ["trade_date"]).copy()
    grouped = df.groupby(group_cols, observed=True)
    feature_map: dict[str, pd.Series] = {}
    for window in windows:
        for col in value_cols:
            feature_map[suffix_template.format(col=col, window=window, stat="mean")] = grouped[col].transform(
                lambda s: s.rolling(window, min_periods=2).mean()
            )
            feature_map[suffix_template.format(col=col, window=window, stat="std")] = grouped[col].transform(
                lambda s: s.rolling(window, min_periods=2).std(ddof=0)
            )
    if not feature_map:
        return df
    return pd.concat([df, pd.DataFrame(feature_map, index=df.index)], axis=1).copy()


def winsorize_columns(df: pd.DataFrame, cols: list[str], pct: float) -> pd.DataFrame:
    if pct <= 0:
        return df
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        series = out[col]
        if not np.issubdtype(series.dtype, np.number):
            continue
        lo, hi = series.quantile([pct, 1 - pct])
        out[col] = series.clip(lo, hi)
    return out


def safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = pd.to_numeric(numerator, errors="coerce").astype(float) / pd.to_numeric(denominator, errors="coerce").astype(float)
    ratio = ratio.where(ratio > 0)
    return np.log(ratio).replace([np.inf, -np.inf], np.nan)


def derive_market_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.sort_values(["symbol", "bar_ts"]).copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out["slot_idx"] = out.groupby(["symbol", "trade_date"], observed=True).cumcount()
    out["bar_time"] = out["bar_ts"].dt.strftime("%H:%M")

    day_group = out.groupby(["symbol", "trade_date"], observed=True)
    day_open = day_group["open"].transform("first")

    out["bars_seen"] = out["slot_idx"].astype(int) + 1
    out["bars_remaining"] = EXPECTED_REGULAR_BARS - out["bars_seen"]
    out["minutes_seen"] = out["bars_seen"].astype(float) * 15.0
    out["intraday_progress"] = out["bars_seen"].astype(float) / float(EXPECTED_REGULAR_BARS)

    out["cutoff_close"] = pd.to_numeric(out["close"], errors="coerce").astype(float)
    out["cutoff_high_seen"] = day_group["high"].cummax().astype(float)
    out["cutoff_low_seen"] = day_group["low"].cummin().astype(float)
    out["cutoff_volume_seen"] = day_group["volume"].cumsum().astype(float)
    out["cutoff_slot_idx"] = out["slot_idx"].astype(int)
    out["cutoff_time"] = out["bar_time"]

    out["close_open_log_return"] = safe_log_ratio(out["close"], out["open"])
    out["high_low_log_range"] = safe_log_ratio(out["high"], out["low"])
    out["close_vs_lag_close_1"] = safe_log_ratio(out["close"], out["lag_close_1"])
    out["close_vs_lag_close_5"] = safe_log_ratio(out["close"], out["lag_close_5"])
    out["close_vs_lag_close_10"] = safe_log_ratio(out["close"], out["lag_close_10"])
    out["close_vs_sma_close_5"] = safe_log_ratio(out["close"], out["sma_close_5"])
    out["close_vs_sma_close_10"] = safe_log_ratio(out["close"], out["sma_close_10"])
    out["close_vs_sma_close_20"] = safe_log_ratio(out["close"], out["sma_close_20"])
    out["cutoff_return_from_open"] = safe_log_ratio(out["close"], day_open)
    out["cutoff_range_seen"] = safe_log_ratio(out["cutoff_high_seen"], out["cutoff_low_seen"])
    out["cutoff_price_vs_prev_close"] = safe_log_ratio(out["close"], out["previous_close"])
    out["log_volume"] = np.log1p(pd.to_numeric(out["volume"], errors="coerce").clip(lower=0).fillna(0.0))
    out["log_slot_count"] = np.log1p(pd.to_numeric(out["slot_count"], errors="coerce").clip(lower=0).fillna(0.0))
    out["log_cutoff_volume_seen"] = np.log1p(out["cutoff_volume_seen"].clip(lower=0).fillna(0.0))
    out["volume_vs_sma_volume_5"] = (
        np.log1p(pd.to_numeric(out["volume"], errors="coerce").clip(lower=0).fillna(0.0))
        - np.log1p(pd.to_numeric(out["sma_volume_5"], errors="coerce").clip(lower=0).fillna(0.0))
    )
    return out


def build_next_session_targets(bars: pd.DataFrame) -> pd.DataFrame:
    session_meta = (
        bars[["symbol", "trade_date"]]
        .drop_duplicates()
        .sort_values(["symbol", "trade_date"])
        .reset_index(drop=True)
    )
    session_meta["next_trade_date"] = session_meta.groupby("symbol", observed=True)["trade_date"].shift(-1)

    session_counts = (
        bars.groupby(["symbol", "trade_date"], observed=True)
        .size()
        .rename("bar_count")
        .reset_index()
    )
    complete_sessions = session_counts[session_counts["bar_count"] == EXPECTED_REGULAR_BARS][["symbol", "trade_date"]]
    if complete_sessions.empty:
        targets = session_meta.copy()
        for slot in range(EXPECTED_REGULAR_BARS):
            targets[f"next_close_h{slot:02d}"] = np.nan
        return targets

    complete_bars = bars.merge(complete_sessions, on=["symbol", "trade_date"], how="inner")
    next_close_wide = complete_bars.pivot(index=["symbol", "trade_date"], columns="slot_idx", values="close")
    next_close_wide = next_close_wide.reindex(columns=list(range(EXPECTED_REGULAR_BARS)))
    next_close_wide.columns = [f"next_close_h{int(slot):02d}" for slot in next_close_wide.columns]
    next_close_wide = next_close_wide.reset_index().rename(columns={"trade_date": "next_trade_date"})
    return session_meta.merge(next_close_wide, on=["symbol", "next_trade_date"], how="left")


def build_symbol_day_dataset(
    bars: pd.DataFrame,
    daily_windows: list[int],
    slot_windows: list[int],
    winsor_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regular = derive_market_features(bars)
    target_frame = build_next_session_targets(regular)
    dataset = regular.merge(target_frame, on=["symbol", "trade_date"], how="left")

    roll_cols = [
        "close_open_log_return",
        "high_low_log_range",
        "pct_change_1",
        "pct_change_5",
        "log_return_1",
        "cutoff_return_from_open",
        "cutoff_range_seen",
        "cutoff_price_vs_prev_close",
        "cutoff_volume_seen",
        "overnight_gap_pct",
        "overnight_log_return",
    ]
    dataset = add_group_rolling_features_by_keys(
        dataset,
        group_cols=["symbol", "slot_idx"],
        value_cols=[col for col in roll_cols if col in dataset.columns],
        windows=daily_windows,
        suffix_template="{col}_{stat}_{window}d",
    )

    short_roll_cols = [
        "cutoff_return_from_open",
        "cutoff_range_seen",
        "cutoff_volume_seen",
        "pct_change_1",
    ]
    dataset = add_group_rolling_features_by_keys(
        dataset,
        group_cols=["symbol", "slot_idx"],
        value_cols=[col for col in short_roll_cols if col in dataset.columns],
        windows=slot_windows,
        suffix_template="{col}_{stat}_{window}cut",
    )

    for slot in range(EXPECTED_REGULAR_BARS):
        next_col = f"next_close_h{slot:02d}"
        dataset[f"target_h{slot:02d}"] = safe_log_ratio(dataset[next_col], dataset["close"])

    winsor_cols = [
        col
        for col in dataset.columns
        if col not in TARGET_COLUMNS
        and not col.startswith("next_close_h")
        and any(
            token in col
            for token in [
                "return",
                "range",
                "gap",
                "vol",
                "volume",
                "log_",
                "pct_change",
            ]
        )
    ]
    dataset = winsorize_columns(dataset, winsor_cols, winsor_pct)
    return dataset.reset_index(drop=True), regular.reset_index(drop=True)


def build_cutoff_replay_dataset(
    bars: pd.DataFrame,
    daily_windows: list[int],
    slot_windows: list[int],
    winsor_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_symbol_day_dataset(
        bars=bars,
        daily_windows=daily_windows,
        slot_windows=slot_windows,
        winsor_pct=winsor_pct,
    )


def make_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_COLUMNS].copy()
    X = df.drop(columns=[col for col in NON_FEATURE_COLS if col in df.columns], errors="ignore").copy()
    object_cols = [col for col in X.columns if X[col].dtype == "object"]
    if object_cols:
        X = pd.get_dummies(X, columns=object_cols, dummy_na=True)
    return X, y


def align_features_for_inference(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X, _ = make_feature_matrix(df.assign(**{col: 0.0 for col in TARGET_COLUMNS}))
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    return X


def validate_feature_strategies(strategies: list[str]) -> list[str]:
    valid: list[str] = []
    for strategy in strategies:
        if strategy == "full":
            valid.append(strategy)
            continue
        if strategy.startswith("top_corr_"):
            limit = strategy.removeprefix("top_corr_")
            if limit.isdigit() and int(limit) > 0:
                valid.append(strategy)
                continue
        raise ValueError(f"Unsupported feature strategy: {strategy}")
    return valid


def validate_weight_strategies(strategies: list[str]) -> list[str]:
    valid: list[str] = []
    for strategy in strategies:
        if strategy == "uniform":
            valid.append(strategy)
            continue
        if strategy.startswith("exp_decay_"):
            half_life = strategy.removeprefix("exp_decay_")
            if half_life.isdigit() and int(half_life) > 0:
                valid.append(strategy)
                continue
        raise ValueError(f"Unsupported weight strategy: {strategy}")
    return valid


def select_feature_columns(X_train: pd.DataFrame, y_train: pd.Series, strategy: str) -> list[str]:
    if strategy == "full":
        return list(X_train.columns)
    limit = int(strategy.removeprefix("top_corr_"))
    scores = X_train.corrwith(y_train, method="pearson").abs().fillna(0.0)
    if scores.empty:
        return list(X_train.columns)
    selected = scores.sort_values(ascending=False).head(min(limit, len(scores))).index.tolist()
    return selected or list(X_train.columns)


def restrict_mask_to_recent_months(df: pd.DataFrame, mask: pd.Series, months: int) -> pd.Series:
    if months <= 0:
        return mask
    masked_dates = df.loc[mask, "trade_date"]
    if masked_dates.empty:
        return mask
    cutoff = masked_dates.max() - pd.DateOffset(months=months)
    return mask & (df["trade_date"] >= cutoff)


def restrict_df_to_recent_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    if months <= 0 or df.empty:
        return df
    cutoff = df["trade_date"].max() - pd.DateOffset(months=months)
    return df[df["trade_date"] >= cutoff].copy()


def restrict_df_to_recent_trading_days(df: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    if trading_days <= 0 or df.empty:
        return df
    unique_dates = sorted(pd.Series(df["trade_date"].unique()).tolist())
    if len(unique_dates) <= trading_days:
        return df.copy()
    keep_dates = set(unique_dates[-trading_days:])
    return df[df["trade_date"].isin(keep_dates)].copy()


def compute_sample_weight(trade_dates: pd.Series, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.ones(len(trade_dates), dtype=float)
    half_life_days = int(strategy.removeprefix("exp_decay_"))
    latest_date = pd.Timestamp(trade_dates.max()).normalize()
    age_days = (latest_date - pd.to_datetime(trade_dates).dt.normalize()).dt.days.to_numpy(dtype=float)
    weights = np.power(0.5, age_days / float(half_life_days))
    return np.clip(weights, 1e-3, None)


def walkforward_folds(df: pd.DataFrame, n_folds: int = 5, min_train_frac: float = 0.5, test_block_days: int = 5):
    dates = sorted(df["trade_date"].unique())
    n_dates = len(dates)
    if n_dates < n_folds + test_block_days + 2:
        return []
    start_idx = max(int(n_dates * min_train_frac), 20)
    stop_idx = n_dates - test_block_days
    if stop_idx <= start_idx:
        return []
    test_starts = np.linspace(start_idx, stop_idx, n_folds, dtype=int)
    folds = []
    for start in test_starts:
        train_dates = set(dates[:start])
        test_dates = set(dates[start : start + test_block_days])
        if not test_dates:
            continue
        train_mask = df["trade_date"].isin(train_dates)
        test_mask = df["trade_date"].isin(test_dates)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        folds.append((train_mask, test_mask))
    return folds


def rmse_no_kw(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def summarize_direction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def experiment_key(
    window_months: int,
    variant_name: str,
    fold_strategy: str,
    feature_strategy: str,
    weight_strategy: str,
) -> tuple[int, str, str, str, str]:
    return int(window_months), str(variant_name), str(fold_strategy), str(feature_strategy), str(weight_strategy)


def persist_results(output_dir: Path, results: list[dict[str, Any]], fold_frames: list[pd.DataFrame]) -> None:
    results_path = output_dir / "cv_results.csv"
    folds_path = output_dir / "cv_fold_details.csv"
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(
            ["mean_rmse", "final_horizon_rmse", "rmse_std", "direction_sort"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
        res_df.to_csv(results_path, index=False)
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(folds_path, index=False)


def tune_params_for_final_horizon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight_train: np.ndarray,
    base_params: dict[str, Any],
    n_trials: int,
    timeout_min: int,
    allow_gpu_fallback: bool,
) -> dict[str, Any]:
    import optuna

    params = base_params.copy()

    def objective(trial):
        tuned = params.copy()
        tuned.update(
            {
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 4.0),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1400),
            }
        )
        pred, _, used = fit_with_optional_fallback(
            X_train,
            y_train,
            X_val,
            tuned,
            allow_gpu_fallback=allow_gpu_fallback,
            sample_weight=sample_weight_train,
        )
        if used["device"] != tuned["device"]:
            params["device"] = used["device"]
        return rmse_no_kw(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_min * 60)
    params.update(study.best_params)
    return params


def evaluate_experiment(
    df: pd.DataFrame,
    base_params: dict[str, Any],
    variant: dict[str, Any] | None,
    fold_strategy: str,
    feature_strategy: str,
    weight_strategy: str,
    args: argparse.Namespace,
    allow_gpu_fallback: bool,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame] | None:
    params = base_params.copy()
    variant_name = "base"
    if variant is not None:
        variant_name = str(variant["name"])
        params.update({k: v for k, v in variant.items() if k != "name"})

    X, y = make_feature_matrix(df)
    folds = walkforward_folds(df, n_folds=args.n_folds, test_block_days=args.test_block_days)
    if not folds:
        return None

    if getattr(args, "use_optuna", False):
        first_train_mask, first_test_mask = folds[0]
        if fold_strategy == "rolling":
            first_train_mask = restrict_mask_to_recent_months(df, first_train_mask, getattr(args, "rolling_train_months", 18))
        X_train_tune = X[first_train_mask]
        y_train_tune = y.loc[first_train_mask, TARGET_COLUMNS[-1]]
        if len(X_train_tune) < args.min_train_rows:
            return None
        selected_cols = select_feature_columns(X_train_tune, y_train_tune, feature_strategy)
        tune_weights = compute_sample_weight(df.loc[first_train_mask, "trade_date"], weight_strategy)
        params = tune_params_for_final_horizon(
            X_train=X_train_tune[selected_cols],
            y_train=y_train_tune,
            X_val=X.loc[first_test_mask, selected_cols],
            y_val=y.loc[first_test_mask, TARGET_COLUMNS[-1]],
            sample_weight_train=tune_weights,
            base_params=params,
            n_trials=args.optuna_trials,
            timeout_min=args.optuna_timeout_min,
            allow_gpu_fallback=allow_gpu_fallback,
        )

    t0 = time.time()
    active_device = params.get("device", "cpu")
    fold_rows: list[dict[str, Any]] = []
    fold_direction_rows: list[dict[str, Any]] = []
    feature_counts: list[int] = []

    for fold_idx, (train_mask, test_mask) in enumerate(folds):
        if fold_strategy == "rolling":
            train_mask = restrict_mask_to_recent_months(df, train_mask, getattr(args, "rolling_train_months", 18))
        if int(train_mask.sum()) < args.min_train_rows:
            return None

        X_train_full, X_test_full = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        selected_cols = select_feature_columns(X_train_full, y_train[TARGET_COLUMNS[-1]], feature_strategy)
        sample_weight = compute_sample_weight(df.loc[train_mask, "trade_date"], weight_strategy)
        feature_counts.append(len(selected_cols))
        X_train = X_train_full[selected_cols]
        X_test = X_test_full[selected_cols]
        final_fold_pred: np.ndarray | None = None
        final_fold_true: np.ndarray | None = None

        for horizon_idx, target_col in enumerate(TARGET_COLUMNS):
            pred, _, used_params = fit_with_optional_fallback(
                X_train,
                y_train[target_col],
                X_test,
                params,
                allow_gpu_fallback=allow_gpu_fallback,
                sample_weight=sample_weight,
            )
            active_device = used_params.get("device", active_device)
            params["device"] = active_device
            y_true = y_test[target_col].to_numpy()
            baseline_col = f"slot_{horizon_idx:02d}_slot_path_to_close"
            if baseline_col in df.columns:
                base_pred = df.loc[test_mask, baseline_col].to_numpy()
            else:
                base_pred = np.repeat(float(y_train[target_col].mean()), test_mask.sum())
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "horizon_idx": horizon_idx,
                    "rmse": rmse_no_kw(y_true, pred),
                    "mae": float(mean_absolute_error(y_true, pred)),
                    "base_rmse": rmse_no_kw(y_true, base_pred),
                    "rmse_uplift": rmse_no_kw(y_true, base_pred) - rmse_no_kw(y_true, pred),
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    "device": active_device,
                    "n_features": len(selected_cols),
                }
            )
            if horizon_idx == EXPECTED_REGULAR_BARS - 1:
                final_fold_pred = pred
                final_fold_true = y_true

        if final_fold_pred is not None and final_fold_true is not None:
            fold_direction_rows.append(
                {
                    "fold": fold_idx,
                    "direction_accuracy": summarize_direction(final_fold_true, final_fold_pred),
                }
            )

    elapsed_sec = float(time.time() - t0)
    fold_df = pd.DataFrame(fold_rows)
    direction_df = pd.DataFrame(fold_direction_rows)
    final_horizon_df = fold_df[fold_df["horizon_idx"] == EXPECTED_REGULAR_BARS - 1]

    summary = {
        "variant": variant_name,
        "fold_strategy": fold_strategy,
        "feature_strategy": feature_strategy,
        "weight_strategy": weight_strategy,
        "rolling_train_months": int(getattr(args, "rolling_train_months", 18)) if fold_strategy == "rolling" else 0,
        "mean_rmse": float(fold_df["rmse"].mean()),
        "mean_mae": float(fold_df["mae"].mean()),
        "final_horizon_rmse": float(final_horizon_df["rmse"].mean()),
        "direction_accuracy": float(direction_df["direction_accuracy"].mean()) if not direction_df.empty else float("nan"),
        "base_mean_rmse": float(fold_df["base_rmse"].mean()),
        "mean_rmse_uplift": float(fold_df["rmse_uplift"].mean()),
        "rmse_std": float(fold_df["rmse"].std(ddof=0)),
        "fit_sec": elapsed_sec,
        "train_rows_mean": float(fold_df["train_rows"].mean()),
        "test_rows_mean": float(fold_df["test_rows"].mean()),
        "n_features": int(round(float(np.mean(feature_counts)))) if feature_counts else int(X.shape[1]),
        "n_rows": int(len(df)),
        "device": active_device,
        "direction_sort": -float(direction_df["direction_accuracy"].mean()) if not direction_df.empty else 0.0,
    }
    return summary, params, fold_df


# ---------------------------------------------------------------------------
# Module-level worker state for ProcessPoolExecutor (avoids pickling X/y
# once per task — instead X is sent once per worker via initializer).
# ---------------------------------------------------------------------------
_GPU_WORKER_STATE: dict = {}


def _gpu_worker_init(
    X_data: "pd.DataFrame",
    y_data: "pd.DataFrame",
    params: dict,
    model_dir_str: str,
    allow_fallback: bool,
    sample_weight: "np.ndarray | None",
    num_gpus: int,
    init_model_paths: "list[str | None]",
) -> None:
    _GPU_WORKER_STATE["X"] = X_data
    _GPU_WORKER_STATE["y"] = y_data
    _GPU_WORKER_STATE["params"] = params
    _GPU_WORKER_STATE["model_dir"] = Path(model_dir_str)
    _GPU_WORKER_STATE["allow_fallback"] = allow_fallback
    _GPU_WORKER_STATE["sample_weight"] = sample_weight
    _GPU_WORKER_STATE["num_gpus"] = num_gpus
    _GPU_WORKER_STATE["init_model_paths"] = init_model_paths


def _gpu_worker_train(horizon_idx: int) -> tuple[int, str, list, dict, dict]:
    gpu_id = horizon_idx % _GPU_WORKER_STATE["num_gpus"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    X = _GPU_WORKER_STATE["X"]
    y = _GPU_WORKER_STATE["y"]
    params = _GPU_WORKER_STATE["params"].copy()
    model_dir = _GPU_WORKER_STATE["model_dir"]
    allow_fallback = _GPU_WORKER_STATE["allow_fallback"]
    sample_weight = _GPU_WORKER_STATE["sample_weight"]
    init_model_paths = _GPU_WORKER_STATE.get("init_model_paths", [])
    init_model_path = init_model_paths[horizon_idx] if init_model_paths else None
    target_col = TARGET_COLUMNS[horizon_idx]
    _, model, used_params = fit_with_optional_fallback(
        X, y[target_col], X, params, allow_fallback, sample_weight=sample_weight,
        init_model_path=init_model_path,
    )
    model_path = model_dir / f"horizon_{horizon_idx:02d}.json"
    model.get_booster().save_model(str(model_path))
    feature_types_local = list(model.get_booster().feature_types or [])
    score = {
        feature: float(gain)
        for feature, gain in model.get_booster().get_score(importance_type="gain").items()
    }
    return horizon_idx, str(model_path), feature_types_local, score, used_params


def train_full_model_set(
    df: pd.DataFrame,
    params: dict[str, Any],
    allow_gpu_fallback: bool,
    model_dir: Path,
    selected_feature_names: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
    parallel_horizons: int = 1,
    target_definition: str = PRODUCTION_TARGET_DEFINITION,
    init_model_paths: list[str | None] | None = None,
) -> tuple[list[str], list[str], dict[str, Any], pd.DataFrame]:
    model_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_feature_matrix(df)
    if selected_feature_names is not None:
        X = X[selected_feature_names].copy()
    feature_names = list(X.columns)
    feature_types: list[str] = []
    saved_models: dict[str, str] = {}
    gain_store: defaultdict[str, list[float]] = defaultdict(list)
    active_params = params.copy()
    _init_paths: list[str | None] = init_model_paths or [None] * EXPECTED_REGULAR_BARS

    def train_one(horizon_idx: int) -> tuple[int, str, list[str], dict[str, float], dict[str, Any]]:
        target_col = TARGET_COLUMNS[horizon_idx]
        worker_params = active_params.copy()
        _, model, used_params = fit_with_optional_fallback(
            X,
            y[target_col],
            X,
            worker_params,
            allow_gpu_fallback=allow_gpu_fallback,
            sample_weight=sample_weight,
            init_model_path=_init_paths[horizon_idx],
        )
        model_path = model_dir / f"horizon_{horizon_idx:02d}.json"
        model.get_booster().save_model(str(model_path))
        feature_types_local = list(model.get_booster().feature_types or [])
        score = {feature: float(gain) for feature, gain in model.get_booster().get_score(importance_type="gain").items()}
        return horizon_idx, str(model_path), feature_types_local, score, used_params

    num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    can_parallelize_cpu = active_params.get("device") == "cpu" and parallel_horizons > 1
    can_parallelize_gpu = active_params.get("device") == "cuda" and num_gpus > 1

    if can_parallelize_gpu:
        # Run workers_per_gpu concurrent jobs per GPU (H100 has 80 GB VRAM;
        # XGBoost hist typically uses 1-4 GB per model, so 4 per GPU is safe).
        workers_per_gpu = int(os.environ.get("XGB_WORKERS_PER_GPU", "4"))
        max_workers = num_gpus * workers_per_gpu
        LOGGER.info(
            "Multi-GPU horizon training: %d GPUs × %d workers = %d concurrent, distributing %d horizons.",
            num_gpus, workers_per_gpu, max_workers, EXPECTED_REGULAR_BARS,
        )
        active_params["n_jobs"] = 1
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_gpu_worker_init,
            initargs=(X, y, active_params, str(model_dir), allow_gpu_fallback, sample_weight, num_gpus, _init_paths),
        ) as executor:
            results = list(executor.map(_gpu_worker_train, range(EXPECTED_REGULAR_BARS)))
    elif can_parallelize_cpu:
        worker_count = max(1, min(parallel_horizons, EXPECTED_REGULAR_BARS))
        active_params["n_jobs"] = 1
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(train_one, range(EXPECTED_REGULAR_BARS)))
    else:
        if active_params.get("device") != "cpu" and parallel_horizons > 1:
            LOGGER.info("Disabling parallel horizon training on %s to avoid device contention.", active_params.get("device"))
        results = [train_one(horizon_idx) for horizon_idx in range(EXPECTED_REGULAR_BARS)]

    for horizon_idx, model_path_str, feature_types_local, score, used_params in sorted(results, key=lambda item: item[0]):
        target_col = TARGET_COLUMNS[horizon_idx]
        if horizon_idx == 0:
            feature_types = feature_types_local
        saved_models[target_col] = Path(model_path_str).name
        active_params["device"] = used_params.get("device", active_params.get("device"))
        for feature, gain in score.items():
            gain_store[feature].append(gain)

    feature_importance = (
        pd.DataFrame(
            {
                "feature": list(gain_store.keys()),
                "mean_gain": [float(np.mean(vals)) for vals in gain_store.values()],
                "n_models": [len(vals) for vals in gain_store.values()],
            }
        )
        .sort_values(["mean_gain", "n_models"], ascending=[False, False])
        .head(25)
        .reset_index(drop=True)
    )
    manifest = {
        "horizon_count": EXPECTED_REGULAR_BARS,
        "target_definition": target_definition,
        "models": saved_models,
    }
    with open(model_dir / "model_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return feature_names, feature_types, active_params, feature_importance


def save_feature_importance_plot(feature_importance: pd.DataFrame, output_path: Path) -> None:
    if feature_importance.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_importance, y="feature", x="mean_gain")
    plt.title("Top 25 features by mean gain across horizons")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_feature_names(output_dir: Path, feature_names: list[str]) -> None:
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)


def save_metadata(
    output_dir: Path,
    best: pd.Series,
    final_params: dict[str, Any],
    feature_names: list[str],
    feature_types: list[str],
    args: argparse.Namespace,
    training_date: str,
    artifacts: dict[str, str],
    start_date: str,
    end_date: str,
) -> None:
    metadata = {
        "model_type": "XGBoost_multi_horizon",
        "model_version": output_dir.name,
        "training_date": training_date,
        "prediction_horizon": "next_regular_session_15m_close_path",
        "target_definition": PRODUCTION_TARGET_DEFINITION,
        "horizon_count": EXPECTED_REGULAR_BARS,
        "n_features": int(best.n_features),
        "n_rows": int(best.n_rows),
        "training_window_months": int(best.window_months),
        "variant": str(best.variant),
        "fold_strategy": str(best.fold_strategy),
        "feature_strategy": str(best.feature_strategy),
        "weight_strategy": str(best.weight_strategy),
        "rolling_train_months": int(best.rolling_train_months),
        "daily_windows": args.daily_windows,
        "slot_windows": args.slot_windows,
        "winsor_pct": args.winsor_pct,
        "random_state": args.random_state,
        "device": final_params.get("device"),
        "date_range": {"start_date": start_date, "end_date": end_date},
        "metrics": {
            "mean_rmse": float(best.mean_rmse),
            "mean_mae": float(best.mean_mae),
            "final_horizon_rmse": float(best.final_horizon_rmse),
            "direction_accuracy": float(best.direction_accuracy),
            "base_mean_rmse": float(best.base_mean_rmse),
            "mean_rmse_uplift": float(best.mean_rmse_uplift),
            "rmse_std": float(best.rmse_std),
        },
        "hyperparameters": final_params,
        "feature_count_check": len(feature_names),
        "feature_types": feature_types,
        "artifacts": artifacts,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)


def determine_dates(engine, args: argparse.Namespace) -> tuple[str, str]:
    if args.start_date and args.end_date:
        return args.start_date, args.end_date
    minmax = load_minmax(engine, source_table=args.source_table)
    start_date = args.start_date or minmax.min_date.iloc[0].strftime("%Y-%m-%d")
    end_date = args.end_date or minmax.max_date.iloc[0].strftime("%Y-%m-%d")
    return start_date, end_date


def evaluate_last_block_holdout(
    df: pd.DataFrame,
    params: dict[str, Any],
    args: argparse.Namespace,
    allow_gpu_fallback: bool,
    variant_name: str,
    feature_strategy: str = "full",
    weight_strategy: str = "uniform",
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame] | None:
    dates = sorted(df["trade_date"].unique())
    if len(dates) <= args.test_block_days:
        return None

    test_dates = set(dates[-args.test_block_days :])
    train_dates = set(dates[: -args.test_block_days])
    train_mask = df["trade_date"].isin(train_dates)
    test_mask = df["trade_date"].isin(test_dates)
    if int(train_mask.sum()) < args.min_train_rows or int(test_mask.sum()) == 0:
        return None

    X, y = make_feature_matrix(df)
    X_train_full = X[train_mask]
    X_test_full = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    selected_cols = select_feature_columns(X_train_full, y_train[TARGET_COLUMNS[-1]], feature_strategy)
    X_train = X_train_full[selected_cols]
    X_test = X_test_full[selected_cols]
    sample_weight = compute_sample_weight(df.loc[train_mask, "trade_date"], weight_strategy)

    active_params = params.copy()
    active_device = active_params.get("device", "cpu")
    fold_rows: list[dict[str, Any]] = []
    final_fold_pred: np.ndarray | None = None
    final_fold_true: np.ndarray | None = None

    for horizon_idx, target_col in enumerate(TARGET_COLUMNS):
        pred, _, used_params = fit_with_optional_fallback(
            X_train,
            y_train[target_col],
            X_test,
            active_params,
            allow_gpu_fallback=allow_gpu_fallback,
            sample_weight=sample_weight,
        )
        active_device = used_params.get("device", active_device)
        active_params["device"] = active_device
        y_true = y_test[target_col].to_numpy()
        baseline_col = f"slot_{horizon_idx:02d}_slot_path_to_close"
        if baseline_col in df.columns:
            base_pred = df.loc[test_mask, baseline_col].to_numpy()
        else:
            base_pred = np.repeat(float(y_train[target_col].mean()), test_mask.sum())
        fold_rows.append(
            {
                "fold": 0,
                "horizon_idx": horizon_idx,
                "rmse": rmse_no_kw(y_true, pred),
                "mae": float(mean_absolute_error(y_true, pred)),
                "base_rmse": rmse_no_kw(y_true, base_pred),
                "rmse_uplift": rmse_no_kw(y_true, base_pred) - rmse_no_kw(y_true, pred),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "device": active_device,
                "n_features": len(selected_cols),
            }
        )
        if horizon_idx == EXPECTED_REGULAR_BARS - 1:
            final_fold_pred = pred
            final_fold_true = y_true

    fold_df = pd.DataFrame(fold_rows)
    final_horizon_df = fold_df[fold_df["horizon_idx"] == EXPECTED_REGULAR_BARS - 1]
    direction_accuracy = (
        summarize_direction(final_fold_true, final_fold_pred)
        if final_fold_pred is not None and final_fold_true is not None
        else float("nan")
    )
    summary = {
        "variant": variant_name,
        "fold_strategy": "last_block_holdout",
        "feature_strategy": feature_strategy,
        "weight_strategy": weight_strategy,
        "rolling_train_months": 0,
        "mean_rmse": float(fold_df["rmse"].mean()),
        "mean_mae": float(fold_df["mae"].mean()),
        "final_horizon_rmse": float(final_horizon_df["rmse"].mean()),
        "direction_accuracy": float(direction_accuracy),
        "base_mean_rmse": float(fold_df["base_rmse"].mean()),
        "mean_rmse_uplift": float(fold_df["rmse_uplift"].mean()),
        "rmse_std": float(fold_df["rmse"].std(ddof=0)),
        "fit_sec": float("nan"),
        "train_rows_mean": float(fold_df["train_rows"].mean()),
        "test_rows_mean": float(fold_df["test_rows"].mean()),
        "n_features": int(len(selected_cols)),
        "n_rows": int(len(df)),
        "device": active_device,
        "direction_sort": -float(direction_accuracy) if not np.isnan(direction_accuracy) else 0.0,
    }
    return summary, active_params, fold_df


def direct_train_summary(
    train_df: pd.DataFrame,
    params: dict[str, Any],
    profile_name: str,
    variant_name: str,
    window_months: int,
    training_window_days: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    summary = {
        "variant": variant_name,
        "fold_strategy": "direct_train",
        "feature_strategy": "full",
        "weight_strategy": "uniform",
        "rolling_train_months": 0,
        "mean_rmse": float("nan"),
        "mean_mae": float("nan"),
        "final_horizon_rmse": float("nan"),
        "direction_accuracy": float("nan"),
        "base_mean_rmse": float("nan"),
        "mean_rmse_uplift": float("nan"),
        "rmse_std": float("nan"),
        "fit_sec": float("nan"),
        "train_rows_mean": float(len(train_df)),
        "test_rows_mean": 0.0,
        "n_features": 0,
        "n_rows": int(len(train_df)),
        "device": params.get("device", "cpu"),
        "direction_sort": 0.0,
        "window_months": int(window_months),
        "train_profile": profile_name,
    }
    if training_window_days is not None:
        summary["training_window_days"] = int(training_window_days)
    return summary, params.copy(), pd.DataFrame()


def train_profile_run(
    output_dir: Path,
    train_df: pd.DataFrame,
    regular: pd.DataFrame,
    base_params: dict[str, Any],
    requested_device: str,
    args: argparse.Namespace,
    allow_gpu_fallback: bool,
    start_date: str,
    end_date: str,
    db_url: str,
    profile_name: str,
    variant: dict[str, Any],
    validation_mode: str,
    window_months: int,
    training_window_days: int | None = None,
    init_model_paths: list[str | None] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir, args.log_level)

    LOGGER.info(
        "Training profile=%s validation=%s rows=%s symbols=%s dates=%s",
        profile_name,
        validation_mode,
        len(train_df),
        train_df.symbol.nunique(),
        train_df.trade_date.nunique(),
    )

    if args.train_policy == "simple":
        final_train_params = base_params.copy()
        final_train_params.update({k: v for k, v in variant.items() if k != "name"})
        summary_out = direct_train_summary(
            train_df=train_df,
            params=final_train_params,
            profile_name=profile_name,
            variant_name=str(variant["name"]),
            window_months=window_months,
            training_window_days=training_window_days,
        )
    elif validation_mode == "expanding_cv":
        summary_out = evaluate_experiment(
            df=train_df,
            base_params=base_params,
            variant=variant,
            fold_strategy="expanding",
            feature_strategy="full",
            weight_strategy="uniform",
            args=args,
            allow_gpu_fallback=allow_gpu_fallback,
        )
    elif validation_mode == "last_block_holdout":
        fast_params = base_params.copy()
        fast_params.update({k: v for k, v in variant.items() if k != "name"})
        summary_out = evaluate_last_block_holdout(
            df=train_df,
            params=fast_params,
            args=args,
            allow_gpu_fallback=allow_gpu_fallback,
            variant_name=str(variant["name"]),
        )
    else:
        raise ValueError(f"Unsupported validation mode: {validation_mode}")

    if summary_out is None:
        raise RuntimeError(f"Validation failed for profile {profile_name}. Check row counts and date coverage.")

    summary, validated_params, fold_df = summary_out
    summary["parallel_horizons"] = int(args.parallel_horizons)
    summary["train_profile"] = profile_name
    if training_window_days is not None:
        summary["training_window_days"] = int(training_window_days)
    results = [{"window_months": window_months, **summary}]
    fold_frames: list[pd.DataFrame] = []
    if not fold_df.empty:
        fold_df = fold_df.copy()
        fold_df["window_months"] = window_months
        fold_df["variant"] = summary["variant"]
        fold_df["fold_strategy"] = summary["fold_strategy"]
        fold_df["feature_strategy"] = summary["feature_strategy"]
        fold_df["weight_strategy"] = summary["weight_strategy"]
        fold_df["train_profile"] = profile_name
        if training_window_days is not None:
            fold_df["training_window_days"] = int(training_window_days)
        fold_frames.append(fold_df)
    persist_results(output_dir, results, fold_frames)

    res_df = pd.DataFrame(results)
    best = res_df.iloc[0]
    LOGGER.info(
        "Profile=%s result window=%s variant=%s mean_rmse=%.6f final_horizon_rmse=%.6f direction=%.4f device=%s",
        profile_name,
        f"{window_months}m" if training_window_days is None else f"{training_window_days}d",
        best.variant,
        best.mean_rmse,
        best.final_horizon_rmse,
        best.direction_accuracy,
        best.device,
    )

    feature_names, feature_types, final_params, feature_importance = train_full_model_set(
        df=train_df,
        params=validated_params,
        allow_gpu_fallback=allow_gpu_fallback,
        model_dir=models_dir,
        parallel_horizons=args.parallel_horizons,
        init_model_paths=init_model_paths,
    )
    summary["n_features"] = len(feature_names)
    summary["device"] = final_params.get("device", summary["device"])
    res_df = pd.DataFrame([{"window_months": window_months, **summary}])
    best = res_df.iloc[0]
    persist_results(output_dir, res_df.to_dict(orient="records"), fold_frames)
    save_feature_names(output_dir, feature_names)
    feature_importance.to_csv(output_dir / "feature_importance_top25.csv", index=False)
    save_feature_importance_plot(feature_importance, output_dir / "feature_importance_top25.png")

    artifacts = {
        "cv_results": str(output_dir / "cv_results.csv"),
        "cv_fold_details": str(output_dir / "cv_fold_details.csv"),
        "log_file": str(output_dir / "run.log"),
        "models_dir": str(models_dir),
        "model_manifest": str(models_dir / "model_manifest.json"),
        "metadata": str(output_dir / "metadata.json"),
        "feature_names": str(output_dir / "feature_names.json"),
        "feature_importance_csv": str(output_dir / "feature_importance_top25.csv"),
        "feature_importance_png": str(output_dir / "feature_importance_top25.png"),
        "predictions_dir": str(predictions_dir),
        "reports_dir": str(reports_dir),
    }

    run_summary = {
        "db_url_redacted": db_url.rsplit("@", 1)[-1],
        "mode": "train",
        "train_profile": profile_name,
        "start_date": start_date,
        "end_date": end_date,
        "symbols": args.symbols,
        "windows_months": args.windows_months,
        "fast_refresh_days": int(args.fast_refresh_days),
        "daily_windows": args.daily_windows,
        "slot_windows": args.slot_windows,
        "winsor_pct": args.winsor_pct,
        "random_state": args.random_state,
        "requested_device": requested_device,
        "target_mode": "next_regular_session_close_path",
        "best_result": best.to_dict(),
        "top_results": res_df.head(10).to_dict(orient="records"),
        "final_model_device": final_params.get("device"),
        "final_model_params": final_params,
        "artifacts": artifacts,
    }
    if training_window_days is not None:
        run_summary["training_window_days"] = int(training_window_days)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2, default=str)

    save_metadata(
        output_dir=output_dir,
        best=best,
        final_params=final_params,
        feature_names=feature_names,
        feature_types=feature_types,
        args=args,
        training_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        artifacts=artifacts,
        start_date=start_date,
        end_date=end_date,
    )
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["train_profile"] = profile_name
    if training_window_days is not None:
        metadata["training_window_days"] = int(training_window_days)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    LOGGER.info("Saved %s profile artifacts to %s", profile_name, output_dir)
    return run_summary


def train_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    root_output_dir = resolve_runtime_path(args.output_dir, fallback=PROJECT_ROOT / "artifacts" / "nextday_15m_path_v3")
    root_output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(root_output_dir, args.log_level)

    source_label = describe_source(args)
    base_params, requested_device = resolve_xgb_params(args)
    allow_gpu_fallback = args.device == "auto"

    LOGGER.info("Using source: %s", source_label)
    LOGGER.info("Requested XGBoost device: %s", requested_device)

    engine = open_source(args)

    start_date, end_date = determine_dates(engine, args)
    LOGGER.info("Using range %s to %s", start_date, end_date)
    requested_as_of = resolve_requested_as_of(engine, args)

    bars = load_bars(
        engine,
        start_date,
        end_date,
        args.symbols,
        source_table=args.source_table,
        include_provisional=False,
    )
    bars = clip_bars_to_requested_as_of(bars, requested_as_of)
    LOGGER.info(
        "Loaded bars: rows=%s symbols=%s dates=%s",
        len(bars),
        bars.symbol.nunique(),
        bars.trade_date.nunique(),
    )

    dataset, regular = build_symbol_day_dataset(
        bars=bars,
        daily_windows=args.daily_windows,
        slot_windows=args.slot_windows,
        winsor_pct=args.winsor_pct,
    )
    train_ready = dataset.dropna(subset=TARGET_COLUMNS).copy()
    if train_ready.empty:
        raise RuntimeError("No rows remain after building next-session targets. Check session completeness and date range.")

    LOGGER.info(
        "Built symbol-day dataset: rows=%s train_ready_rows=%s symbols=%s dates=%s",
        len(dataset),
        len(train_ready),
        train_ready.symbol.nunique(),
        train_ready.trade_date.nunique(),
    )

    profiles = ["base", "fast_refresh"] if args.train_profile == "both" else [args.train_profile]
    profile_summaries: dict[str, dict[str, Any]] = {}
    window_months = int(args.windows_months[0])

    for profile_name in profiles:
        if profile_name == "base":
            train_df = restrict_df_to_recent_months(train_ready, window_months)
            if len(train_df) < args.min_train_rows:
                raise RuntimeError(
                    f"Training rows in fixed {window_months}-month window are {len(train_df)} < {args.min_train_rows}."
                )
            profile_output_dir = root_output_dir / "base" if args.train_profile == "both" else root_output_dir
            run_summary = train_profile_run(
                output_dir=profile_output_dir,
                train_df=train_df,
                regular=regular,
                base_params=base_params,
                requested_device=requested_device,
                args=args,
                allow_gpu_fallback=allow_gpu_fallback,
                start_date=start_date,
                end_date=end_date,
                db_url=source_label,
                profile_name="base",
                variant=BEST_FIXED_VARIANT,
                validation_mode="expanding_cv",
                window_months=window_months,
            )
        elif profile_name == "fast_refresh":
            train_df = restrict_df_to_recent_trading_days(train_ready, int(args.fast_refresh_days))
            if len(train_df) < args.min_train_rows:
                raise RuntimeError(
                    f"Training rows in fixed {args.fast_refresh_days}-day fast-refresh window are {len(train_df)} < {args.min_train_rows}."
                )
            profile_output_dir = root_output_dir / "fast_refresh" if args.train_profile == "both" else root_output_dir
            approx_window_months = max(1, int(np.ceil(float(args.fast_refresh_days) / 21.0)))
            run_summary = train_profile_run(
                output_dir=profile_output_dir,
                train_df=train_df,
                regular=regular,
                base_params=base_params,
                requested_device=requested_device,
                args=args,
                allow_gpu_fallback=allow_gpu_fallback,
                start_date=start_date,
                end_date=end_date,
                db_url=source_label,
                profile_name="fast_refresh",
                variant=FAST_REFRESH_VARIANT,
                validation_mode="last_block_holdout",
                window_months=approx_window_months,
                training_window_days=int(args.fast_refresh_days),
            )
        else:
            raise ValueError(f"Unsupported train profile: {profile_name}")
        profile_summaries[profile_name] = run_summary

    if args.train_profile == "both":
        root_summary = {
            "mode": "train",
            "train_profile": "both",
            "start_date": start_date,
            "end_date": end_date,
            "target_mode": "next_regular_session_close_path",
            "model_profiles": {
                name: {
                    "run_dir": str((root_output_dir / name).resolve()),
                    "best_result": summary["best_result"],
                    "final_model_device": summary["final_model_device"],
                }
                for name, summary in profile_summaries.items()
            },
        }
        with open(root_output_dir / "run_summary.json", "w", encoding="utf-8") as fh:
            json.dump(root_summary, fh, indent=2, default=str)
        LOGGER.info("Saved combined profile summary to %s", root_output_dir)
        return root_summary

    return next(iter(profile_summaries.values()))


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))


def resolve_inference_run_dir(args: argparse.Namespace) -> Path:
    requested_dir = resolve_runtime_path(
        args.run_dir or args.output_dir,
        fallback=PROJECT_ROOT / "artifacts" / "nextday_15m_path_v3",
    )
    summary_path = requested_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("train_profile") == "both":
            selected_profile = args.train_profile if args.train_profile in {"base", "fast_refresh"} else "base"
            profile_dir = requested_dir / selected_profile
            if (profile_dir / "run_summary.json").exists():
                return profile_dir
        return requested_dir

    for candidate in ["base", "fast_refresh"]:
        profile_dir = requested_dir / candidate
        if (profile_dir / "run_summary.json").exists():
            if args.train_profile in {"base", "fast_refresh"}:
                explicit_dir = requested_dir / args.train_profile
                if (explicit_dir / "run_summary.json").exists():
                    return explicit_dir
            return profile_dir
    raise RuntimeError(f"Could not locate a runnable model directory under {requested_dir}.")


def regular_slot_labels() -> list[str]:
    labels = []
    minute = REGULAR_OPEN_MINUTE
    for _ in range(EXPECTED_REGULAR_BARS):
        hour = minute // 60
        minute_part = minute % 60
        labels.append(f"{hour:02d}:{minute_part:02d}")
        minute += 15
    return labels


def load_model_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "models" / "model_manifest.json").read_text(encoding="utf-8"))


def predict_path_matrix(run_dir: Path, X: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    manifest = load_model_manifest(run_dir)
    feature_names = json.loads((run_dir / "feature_names.json").read_text(encoding="utf-8"))
    if list(X.columns) != feature_names:
        raise RuntimeError("Inference feature order does not match saved feature_names.json.")

    dmat = xgb.DMatrix(X, feature_names=feature_names)
    preds = np.zeros((len(X), manifest["horizon_count"]), dtype=float)
    for horizon_idx in range(manifest["horizon_count"]):
        booster = xgb.Booster()
        model_ref = Path(manifest["models"][f"target_h{horizon_idx:02d}"])
        model_path = model_ref if model_ref.is_absolute() else run_dir / "models" / model_ref
        booster.load_model(str(model_path))
        preds[:, horizon_idx] = booster.predict(dmat)
    return preds


def predicted_direction_label(values: np.ndarray) -> list[str]:
    labels = []
    for value in values:
        if value > 0:
            labels.append("up")
        elif value < 0:
            labels.append("down")
        else:
            labels.append("flat")
    return labels


def plot_symbol_forecast(
    symbol: str,
    regular_history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    predicted_close: np.ndarray,
    out_path: Path,
    days_back: int,
) -> None:
    symbol_history = regular_history[regular_history["symbol"] == symbol].copy()
    history_dates = sorted([d for d in symbol_history["trade_date"].unique() if d <= as_of_date])[-days_back:]
    if not history_dates:
        return

    plt.figure(figsize=(12, 5))
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    x_cursor = 0
    palette = sns.color_palette("Blues", n_colors=max(len(history_dates), 2))

    for idx, trade_date in enumerate(history_dates):
        day = symbol_history[symbol_history["trade_date"] == trade_date].sort_values("slot_idx")
        xs = np.arange(x_cursor, x_cursor + len(day))
        plt.plot(
            xs,
            day["close"].to_numpy(dtype=float),
            color=palette[idx],
            linewidth=1.6,
            label=str(pd.Timestamp(trade_date).date()),
        )
        tick_positions.append(float(xs.mean()))
        tick_labels.append(pd.Timestamp(trade_date).strftime("%m-%d"))
        x_cursor += EXPECTED_REGULAR_BARS
        plt.axvline(x_cursor - 0.5, color="#cccccc", linewidth=0.6)

    pred_xs = np.arange(x_cursor, x_cursor + len(predicted_close))
    plt.plot(pred_xs, predicted_close, color="#F58518", linewidth=2.4, label="Predicted next session")
    tick_positions.append(float(pred_xs.mean()))
    tick_labels.append("Pred")
    plt.axvline(x_cursor - 0.5, color="#999999", linewidth=1.0, linestyle="--")
    plt.xticks(tick_positions, tick_labels)
    plt.ylabel("Close price")
    plt.title(f"{symbol}: recent regular bars + predicted next session path")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def update_run_summary_with_inference(run_dir: Path, extra_artifacts: dict[str, str]) -> None:
    summary_path = run_dir / "run_summary.json"
    summary = load_run_summary(run_dir)
    summary.setdefault("artifacts", {}).update(extra_artifacts)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)


def infer_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = resolve_inference_run_dir(args)
    run_summary = load_run_summary(run_dir)
    configure_logging(run_dir, args.log_level)

    LOGGER.info("Using source: %s", describe_source(args))
    engine = open_source(args)

    requested_as_of = pd.Timestamp(args.as_of_ts) if args.as_of_ts else end_of_day(pd.Timestamp(args.as_of_date or run_summary["end_date"]))
    load_start = (requested_as_of - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    load_end = requested_as_of.strftime("%Y-%m-%d")
    LOGGER.info("Loading inference bars from %s to %s", load_start, load_end)

    bars = load_bars(
        engine,
        load_start,
        load_end,
        run_summary.get("symbols"),
        source_table=args.source_table,
        include_provisional=True,
    )
    bars = clip_bars_to_requested_as_of(bars, requested_as_of)
    dataset, regular = build_symbol_day_dataset(
        bars=bars,
        daily_windows=list(run_summary["daily_windows"]),
        slot_windows=list(run_summary["slot_windows"]),
        winsor_pct=float(run_summary["winsor_pct"]),
    )

    origin_bar_ts = latest_available_origin_bar_ts(dataset, requested_as_of)
    as_of_rows = dataset[dataset["bar_ts"] == origin_bar_ts].copy()
    if as_of_rows.empty:
        raise RuntimeError(f"No origin rows were found for inference on or before {requested_as_of}.")

    feature_names = json.loads((run_dir / "feature_names.json").read_text(encoding="utf-8"))
    X_infer = align_features_for_inference(as_of_rows, feature_names)
    preds = predict_path_matrix(run_dir, X_infer)

    base_close = as_of_rows["close"].to_numpy(dtype=float)
    predicted_close = np.exp(preds) * base_close[:, None]
    predicted_full_day_return = preds[:, -1]

    predictions_dir = run_dir / "predictions"
    reports_dir = run_dir / "reports"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    slot_labels = regular_slot_labels()

    for row_idx, (_, row) in enumerate(as_of_rows.iterrows()):
        symbol = str(row["symbol"])
        entry = {
            "symbol": symbol,
            "forecast_origin_date": str(pd.Timestamp(row["trade_date"]).date()),
            "forecast_origin_ts": str(pd.Timestamp(row["bar_ts"])),
            "predicted_session": "next_regular_session",
            "origin_close": float(row["close"]),
            "predicted_full_day_return": float(predicted_full_day_return[row_idx]),
            "predicted_direction": predicted_direction_label(np.array([predicted_full_day_return[row_idx]]))[0],
        }
        for horizon_idx in range(EXPECTED_REGULAR_BARS):
            entry[f"pred_log_return_h{horizon_idx:02d}"] = float(preds[row_idx, horizon_idx])
            entry[f"pred_close_h{horizon_idx:02d}"] = float(predicted_close[row_idx, horizon_idx])
            path_rows.append(
                {
                    "symbol": symbol,
                    "forecast_origin_date": str(pd.Timestamp(row["trade_date"]).date()),
                    "forecast_origin_ts": str(pd.Timestamp(row["bar_ts"])),
                    "predicted_session": "next_regular_session",
                    "bar_idx": horizon_idx,
                    "bar_time": slot_labels[horizon_idx],
                    "pred_log_return": float(preds[row_idx, horizon_idx]),
                    "pred_close": float(predicted_close[row_idx, horizon_idx]),
                }
            )
        prediction_rows.append(entry)
        plot_symbol_forecast(
            symbol=symbol,
            regular_history=regular,
            as_of_date=pd.Timestamp(origin_bar_ts).normalize(),
            predicted_close=predicted_close[row_idx],
            out_path=reports_dir / f"{symbol}_nextday_path.png",
            days_back=args.report_days_back,
        )

    predictions_df = pd.DataFrame(prediction_rows).sort_values("predicted_full_day_return", ascending=False).reset_index(drop=True)
    path_df = pd.DataFrame(path_rows)
    predictions_path = predictions_dir / "predictions.csv"
    path_output_path = predictions_dir / "predicted_path.csv"
    predictions_df.to_csv(predictions_path, index=False)
    path_df.to_csv(path_output_path, index=False)

    report_summary_path = reports_dir / "report_summary.md"
    with open(report_summary_path, "w", encoding="utf-8") as fh:
        fh.write("# Next-Session 15-Minute Path Forecast\n\n")
        fh.write(f"- Forecast origin date: {pd.Timestamp(origin_bar_ts).date()}\n")
        fh.write(f"- Forecast origin bar: {origin_bar_ts}\n")
        fh.write(f"- Symbols scored: {len(predictions_df)}\n")
        fh.write(f"- Predictions file: `{predictions_path}`\n")
        fh.write(f"- Path file: `{path_output_path}`\n\n")
        fh.write("## Top predicted next-session returns\n\n")
        fh.write(predictions_df[["symbol", "predicted_full_day_return", "predicted_direction"]].head(20).to_markdown(index=False))
        fh.write("\n")

    update_run_summary_with_inference(
        run_dir,
        {
            "predictions_csv": str(predictions_path),
            "predicted_path_csv": str(path_output_path),
            "reports_dir": str(reports_dir),
            "report_summary": str(report_summary_path),
        },
    )

    LOGGER.info("Saved inference predictions to %s", predictions_path)
    LOGGER.info("Saved report artifacts to %s", reports_dir)
    return {
        "predictions_csv": str(predictions_path),
        "predicted_path_csv": str(path_output_path),
        "reports_dir": str(reports_dir),
        "report_summary": str(report_summary_path),
    }


def canonical_mode(mode: str) -> str:
    return {
        "train": "bootstrap",
        "infer": "predict",
        "full_run": "cycle",
    }.get(mode, mode)


def resolve_registry_root(args: argparse.Namespace) -> Path:
    preferred = args.run_root or args.output_dir or args.run_dir
    return resolve_runtime_path(preferred, fallback=PROJECT_ROOT / "artifacts" / "production_nextday")


def registry_paths(run_root: Path) -> dict[str, Path]:
    return {
        "root": run_root,
        "base": run_root / "base",
        "refresh": run_root / "refresh",
        "current": run_root / "current",
        "staging": run_root / "staging",
        "failed": run_root / "failed",
    }


def ensure_registry_layout(run_root: Path) -> dict[str, Path]:
    paths = registry_paths(run_root)
    for key, path in paths.items():
        if key == "root":
            path.mkdir(parents=True, exist_ok=True)
        elif key in {"staging", "failed"}:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def utc_now_str() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_compact_id() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")


def remove_tree_with_retries(path: Path, retries: int = 5, sleep_sec: float = 0.5) -> None:
    last_error: OSError | None = None
    for attempt in range(retries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(sleep_sec * (attempt + 1))
    if path.exists() and last_error is not None:
        raise last_error


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cycle_status(path: Path, payload: dict[str, Any]) -> None:
    write_json(path / "cycle_status.json", payload)


def rewrite_bundle_artifacts(bundle_dir: Path, extra_updates: dict[str, Any] | None = None) -> None:
    artifact_paths = {
        "cv_results": str(bundle_dir / "cv_results.csv"),
        "cv_fold_details": str(bundle_dir / "cv_fold_details.csv"),
        "log_file": str(bundle_dir / "run.log"),
        "models_dir": str(bundle_dir / "models"),
        "model_manifest": str(bundle_dir / "models" / "model_manifest.json"),
        "metadata": str(bundle_dir / "metadata.json"),
        "feature_names": str(bundle_dir / "feature_names.json"),
        "feature_importance_csv": str(bundle_dir / "feature_importance_top25.csv"),
        "feature_importance_png": str(bundle_dir / "feature_importance_top25.png"),
        "predictions_dir": str(bundle_dir / "predictions"),
        "reports_dir": str(bundle_dir / "reports"),
        "cycle_status": str(bundle_dir / "cycle_status.json"),
    }
    run_summary_path = bundle_dir / "run_summary.json"
    run_summary = read_json_if_exists(run_summary_path)
    if run_summary is not None:
        run_summary.setdefault("artifacts", {}).update(artifact_paths)
        if extra_updates:
            run_summary.update(extra_updates)
        write_json(run_summary_path, run_summary)

    metadata_path = bundle_dir / "metadata.json"
    metadata = read_json_if_exists(metadata_path)
    if metadata is not None:
        metadata["model_version"] = bundle_dir.name
        metadata.setdefault("artifacts", {}).update(artifact_paths)
        if extra_updates:
            metadata.update(extra_updates)
        write_json(metadata_path, metadata)


def copy_bundle(src: Path, dst: Path, extra_updates: dict[str, Any] | None = None) -> None:
    tmp_dst = dst.parent / f".{dst.name}_tmp"
    backup_dst = dst.parent / f".{dst.name}_bak"
    if tmp_dst.exists():
        remove_tree_with_retries(tmp_dst)
    if backup_dst.exists():
        remove_tree_with_retries(backup_dst)
    shutil.copytree(src, tmp_dst)
    rewrite_bundle_artifacts(tmp_dst, extra_updates=extra_updates)
    if dst.exists():
        dst.rename(backup_dst)
    tmp_dst.rename(dst)
    if backup_dst.exists():
        remove_tree_with_retries(backup_dst)


def sanitize_reason(reason: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in reason.lower())
    return token[:80].strip("_") or "failed"


def move_bundle_to_failed(src: Path, failed_root: Path, reason: str) -> Path:
    failed_root.mkdir(parents=True, exist_ok=True)
    target = failed_root / f"{src.name}__{sanitize_reason(reason)}"
    if not src.exists():
        target.mkdir(parents=True, exist_ok=True)
        write_json(
            target / "cycle_status.json",
            {
                "status": "failed",
                "reason": reason,
                "source_candidate": str(src),
                "finished_at": utc_now_str(),
            },
        )
        return target
    if target.exists():
        remove_tree_with_retries(target)
    shutil.move(str(src), str(target))
    return target


def latest_complete_trade_date(dataset: pd.DataFrame, requested_as_of: pd.Timestamp) -> pd.Timestamp:
    eligible = sorted(
        pd.Timestamp(value).normalize()
        for value in dataset["trade_date"].dropna().unique()
        if pd.Timestamp(value).normalize() <= requested_as_of
    )
    if not eligible:
        raise RuntimeError(f"No complete regular session is available on or before {requested_as_of.date()}.")
    return eligible[-1]


def latest_available_origin_bar_ts(dataset: pd.DataFrame, requested_as_of: pd.Timestamp) -> pd.Timestamp:
    eligible = dataset.loc[dataset["bar_ts"] <= requested_as_of, "bar_ts"].dropna()
    if eligible.empty:
        raise RuntimeError(f"No origin bars are available on or before {requested_as_of.date()}.")
    return pd.Timestamp(eligible.max())


def resolve_load_window(mode: str, requested_as_of: pd.Timestamp, args: argparse.Namespace) -> tuple[str, str]:
    end_date = args.end_date or requested_as_of.strftime("%Y-%m-%d")
    if args.start_date:
        return args.start_date, end_date
    if mode in ("bootstrap", "base"):
        start_date = (requested_as_of - pd.DateOffset(months=args.base_window_months)).strftime("%Y-%m-%d")
    elif mode == "refresh":
        start_date = (requested_as_of - pd.Timedelta(days=max(int(args.fast_refresh_days), 5))).strftime("%Y-%m-%d")
    else:
        start_date = (requested_as_of - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    return start_date, end_date


def load_pipeline_context(engine, args: argparse.Namespace, mode: str) -> dict[str, Any]:
    requested_as_of = resolve_requested_as_of(engine, args)
    load_start, load_end = resolve_load_window(mode, requested_as_of, args)
    bars = load_bars(
        engine,
        load_start,
        load_end,
        args.symbols,
        source_table=args.source_table,
        include_provisional=False,
    )
    bars = clip_bars_to_requested_as_of(bars, requested_as_of)
    if bars.empty:
        raise RuntimeError(f"No bars found between {load_start} and {load_end}.")

    dataset, regular = build_symbol_day_dataset(
        bars=bars,
        daily_windows=args.daily_windows,
        slot_windows=args.slot_windows,
        winsor_pct=args.winsor_pct,
    )
    if dataset.empty:
        raise RuntimeError("Feature dataset is empty after session preparation.")
    train_ready = dataset.dropna(subset=TARGET_COLUMNS).copy()
    if train_ready.empty:
        raise RuntimeError("No train-ready rows remain after feature generation.")
    effective_as_of = pd.Timestamp(train_ready["trade_date"].max()).normalize()
    train_ready = train_ready[train_ready["trade_date"] <= effective_as_of].copy()
    latest_bar_date = pd.Timestamp(bars["trade_date"].max()).normalize()
    return {
        "requested_as_of": requested_as_of,
        "effective_as_of": effective_as_of,
        "load_start": load_start,
        "load_end": load_end,
        "latest_bar_date": latest_bar_date,
        "bars": bars,
        "dataset": dataset,
        "regular": regular,
        "train_ready": train_ready,
    }


def sanitize_training_targets(train_df: pd.DataFrame, min_train_rows: int) -> pd.DataFrame:
    target_values = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    finite_mask = np.isfinite(target_values).all(axis=1)
    row_max_abs = np.max(np.abs(target_values), axis=1)
    anomaly_mask = row_max_abs > MAX_ABS_TARGET
    keep_mask = finite_mask & ~anomaly_mask

    dropped_rows = int((~keep_mask).sum())
    if dropped_rows == 0:
        return train_df

    drop_fraction = dropped_rows / float(len(train_df))
    max_abs_target = float(np.max(row_max_abs))
    LOGGER.warning(
        "Dropping %s/%s training rows with invalid or extreme targets (max abs target=%.4f, threshold=%.2f).",
        dropped_rows,
        len(train_df),
        max_abs_target,
        MAX_ABS_TARGET,
    )
    cleaned = train_df.loc[keep_mask].copy()
    if len(cleaned) < min_train_rows:
        raise RuntimeError(
            f"Training rows after target anomaly filtering are {len(cleaned)} < {min_train_rows}."
        )
    if drop_fraction > MAX_ANOMALY_DROP_FRACTION:
        raise RuntimeError(
            "Training target anomalies exceed the allowed fraction "
            f"({drop_fraction:.2%} > {MAX_ANOMALY_DROP_FRACTION:.2%})."
        )
    return cleaned


def resolve_training_setup(args: argparse.Namespace, profile_name: str) -> tuple[dict[str, Any], str, bool]:
    params, resolved_device = resolve_xgb_params(args)
    if profile_name == "refresh" and args.device == "auto" and resolved_device != "cuda":
        LOGGER.warning("Refresh mode did not detect a GPU; continuing on CPU.")
    allow_gpu_fallback = profile_name == "base" and args.device == "auto"
    return params, resolved_device, allow_gpu_fallback


def current_feature_names(run_root: Path) -> list[str] | None:
    feature_path = run_root / "current" / "feature_names.json"
    if not feature_path.exists():
        return None
    return json.loads(feature_path.read_text(encoding="utf-8"))


def validate_candidate_bundle(
    bundle_dir: Path,
    smoke_df: pd.DataFrame,
    reference_feature_names: list[str] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    try:
        manifest = load_model_manifest(bundle_dir)
        if int(manifest.get("horizon_count", -1)) != EXPECTED_REGULAR_BARS:
            return False, "invalid_horizon_count", {"horizon_count": manifest.get("horizon_count")}

        missing_models = []
        for target_col in TARGET_COLUMNS:
            model_ref = Path(manifest["models"].get(target_col, ""))
            model_path = model_ref if model_ref.is_absolute() else bundle_dir / "models" / model_ref
            if not model_path.exists():
                missing_models.append(str(model_path))
        if missing_models:
            return False, "missing_model_files", {"missing_models": missing_models[:5]}

        feature_names = json.loads((bundle_dir / "feature_names.json").read_text(encoding="utf-8"))
        if not feature_names:
            return False, "empty_feature_names", {}
        if reference_feature_names is not None and feature_names != reference_feature_names:
            return False, "feature_schema_drift", {
                "reference_count": len(reference_feature_names),
                "candidate_count": len(feature_names),
            }

        metadata = read_json_if_exists(bundle_dir / "metadata.json") or {}
        if metadata.get("feature_count_check") not in {None, len(feature_names)}:
            return False, "feature_count_mismatch", {
                "metadata_feature_count": metadata.get("feature_count_check"),
                "feature_names_count": len(feature_names),
            }

        smoke_rows = smoke_df.tail(min(32, len(smoke_df))).copy()
        if smoke_rows.empty:
            return False, "no_smoke_rows", {}

        X_smoke = align_features_for_inference(smoke_rows, feature_names)
        preds = predict_path_matrix(bundle_dir, X_smoke)
        if preds.shape != (len(X_smoke), EXPECTED_REGULAR_BARS):
            return False, "invalid_prediction_shape", {"shape": list(preds.shape)}
        if not np.isfinite(preds).all():
            return False, "non_finite_predictions", {}
        final_std = float(np.std(preds[:, -1]))
        if len(preds) > 1 and final_std < 1e-10:
            return False, "degenerate_final_return_distribution", {"final_return_std": final_std}

        return True, "ok", {
            "smoke_rows": int(len(smoke_rows)),
            "feature_count": len(feature_names),
            "final_return_std": final_std,
        }
    except Exception as exc:
        return False, "bundle_validation_exception", {"error": str(exc)}


def resolve_direct_or_registry_bundle(run_root: Path, preferred_name: str) -> Path:
    if (run_root / "run_summary.json").exists() and (run_root / "models" / "model_manifest.json").exists():
        return run_root
    bundle_dir = run_root / preferred_name
    if bundle_dir.exists():
        return bundle_dir
    raise RuntimeError(f"Could not resolve bundle '{preferred_name}' under {run_root}.")


def train_candidate_bundle(
    run_root: Path,
    args: argparse.Namespace,
    engine,
    db_url: str,
    profile_name: str,
    init_model_paths: list[str | None] | None = None,
) -> dict[str, Any]:
    ctx = load_pipeline_context(engine, args, mode=profile_name)
    if ctx["train_ready"].empty:
        raise RuntimeError("No train-ready rows remain after feature generation.")

    if profile_name == "base":
        train_df = restrict_df_to_recent_months(ctx["train_ready"], args.base_window_months)
        variant = BEST_FIXED_VARIANT
        validation_mode = "expanding_cv"
        window_months = int(args.base_window_months)
        training_window_days = None
    elif profile_name == "refresh":
        train_df = restrict_df_to_recent_trading_days(ctx["train_ready"], int(args.fast_refresh_days))
        variant = FAST_REFRESH_VARIANT
        validation_mode = "last_block_holdout"
        window_months = max(1, int(np.ceil(float(args.fast_refresh_days) / 21.0)))
        training_window_days = int(args.fast_refresh_days)
    elif profile_name == "warm_refresh":
        warm_days = int(getattr(args, "warm_refresh_days", 1))
        train_df = restrict_df_to_recent_trading_days(ctx["train_ready"], warm_days)
        warm_trees = int(getattr(args, "warm_trees", 30))
        variant = {**FAST_REFRESH_VARIANT, "name": "warm_refresh", "n_estimators": warm_trees}
        validation_mode = "last_block_holdout"
        window_months = 1
        training_window_days = warm_days
    else:
        raise ValueError(f"Unsupported profile for training: {profile_name}")

    if len(train_df) < args.min_train_rows:
        raise RuntimeError(f"Training rows for {profile_name} are {len(train_df)} < {args.min_train_rows}.")
    train_df = sanitize_training_targets(train_df, min_train_rows=args.min_train_rows)

    base_params, resolved_device, allow_gpu_fallback = resolve_training_setup(args, profile_name=profile_name)
    candidate_id = f"{profile_name}_{ctx['effective_as_of'].strftime('%Y%m%d')}_{utc_compact_id()}"
    candidate_dir = run_root / "staging" / candidate_id
    started_at = utc_now_str()
    t0 = time.time()
    try:
        run_summary = train_profile_run(
            output_dir=candidate_dir,
            train_df=train_df,
            regular=ctx["regular"],
            base_params=base_params,
            requested_device=resolved_device,
            args=args,
            allow_gpu_fallback=allow_gpu_fallback,
            start_date=ctx["load_start"],
            end_date=ctx["load_end"],
            db_url=db_url,
            profile_name=profile_name,
            variant=variant,
            validation_mode=validation_mode,
            window_months=window_months,
            training_window_days=training_window_days,
            init_model_paths=init_model_paths,
        )
    except Exception as exc:
        runtime_sec = float(time.time() - t0)
        if candidate_dir.exists():
            failure_payload = {
                "mode": profile_name,
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now_str(),
                "runtime_sec": runtime_sec,
                "requested_as_of_date": str(ctx["requested_as_of"].date()),
                "effective_as_of_date": str(ctx["effective_as_of"].date()),
                "latest_bar_date": str(ctx["latest_bar_date"].date()),
                "data_status": "latest_incomplete_ignored" if ctx["latest_bar_date"] > ctx["effective_as_of"] else "complete",
                "model_id": candidate_id,
                "bundle_role": "staging",
                "reason": "training_exception",
                "details": {"error": str(exc)},
                "resolved_device": resolved_device,
            }
            write_cycle_status(candidate_dir, failure_payload)
            rewrite_bundle_artifacts(
                candidate_dir,
                extra_updates={
                    "model_id": candidate_id,
                    "bundle_role": "staging",
                    "production_mode": profile_name,
                    "requested_as_of_date": str(ctx["requested_as_of"].date()),
                    "effective_as_of_date": str(ctx["effective_as_of"].date()),
                    "load_start_date": ctx["load_start"],
                    "load_end_date": ctx["load_end"],
                    "latest_bar_date": str(ctx["latest_bar_date"].date()),
                    "training_runtime_sec": runtime_sec,
                    "last_cycle_status": failure_payload,
                },
            )
        return {
            "success": False,
            "reason": "training_exception",
            "details": {"error": str(exc)},
            "candidate_dir": candidate_dir,
            "candidate_id": candidate_id,
            "runtime_sec": runtime_sec,
            "context": ctx,
            "run_summary": None,
            "status_payload": {
                "mode": profile_name,
                "status": "failed",
                "reason": "training_exception",
                "details": {"error": str(exc)},
            },
        }
    runtime_sec = float(time.time() - t0)
    extra = {
        "model_id": candidate_id,
        "bundle_role": "staging",
        "production_mode": profile_name,
        "requested_as_of_date": str(ctx["requested_as_of"].date()),
        "effective_as_of_date": str(ctx["effective_as_of"].date()),
        "load_start_date": ctx["load_start"],
        "load_end_date": ctx["load_end"],
        "latest_bar_date": str(ctx["latest_bar_date"].date()),
        "training_runtime_sec": runtime_sec,
        "refresh_budget_sec": int(args.refresh_budget_sec),
    }
    rewrite_bundle_artifacts(candidate_dir, extra_updates=extra)

    validation_ref = current_feature_names(run_root) if profile_name == "refresh" and (run_root / "current").exists() else None
    valid, reason, details = validate_candidate_bundle(candidate_dir, train_df, reference_feature_names=validation_ref)
    if profile_name == "refresh" and runtime_sec > float(args.refresh_budget_sec):
        valid = False
        reason = "refresh_budget_exceeded"
        details = {"runtime_sec": runtime_sec, "refresh_budget_sec": float(args.refresh_budget_sec)}

    status_payload = {
        "mode": profile_name,
        "status": "success" if valid else "failed",
        "started_at": started_at,
        "finished_at": utc_now_str(),
        "runtime_sec": runtime_sec,
        "requested_as_of_date": str(ctx["requested_as_of"].date()),
        "effective_as_of_date": str(ctx["effective_as_of"].date()),
        "latest_bar_date": str(ctx["latest_bar_date"].date()),
        "data_status": "latest_incomplete_ignored" if ctx["latest_bar_date"] > ctx["effective_as_of"] else "complete",
        "model_id": candidate_id,
        "bundle_role": "staging",
        "reason": reason,
        "details": details,
        "resolved_device": resolved_device,
    }
    write_cycle_status(candidate_dir, status_payload)
    rewrite_bundle_artifacts(candidate_dir, extra_updates={"last_cycle_status": status_payload})

    return {
        "success": valid,
        "reason": reason,
        "details": details,
        "candidate_dir": candidate_dir,
        "candidate_id": candidate_id,
        "runtime_sec": runtime_sec,
        "context": ctx,
        "run_summary": run_summary,
        "status_payload": status_payload,
    }


def promote_candidate(run_root: Path, candidate_dir: Path, profile_name: str, model_id: str) -> dict[str, Path]:
    registry = ensure_registry_layout(run_root)
    promoted_at = utc_now_str()
    targets = ["current", "base"] if profile_name == "base" else ["current", "refresh"]
    promoted: dict[str, Path] = {}
    for target_name in targets:
        target_dir = registry[target_name]
        copy_bundle(
            candidate_dir,
            target_dir,
            extra_updates={
                "bundle_role": target_name,
                "promoted_at": promoted_at,
                "model_id": model_id,
                "production_mode": profile_name,
            },
        )
        write_cycle_status(
            target_dir,
            {
                "mode": profile_name,
                "status": "promoted",
                "promoted_at": promoted_at,
                "model_id": model_id,
                "bundle_role": target_name,
                "source_candidate": candidate_dir.name,
            },
        )
        rewrite_bundle_artifacts(target_dir, extra_updates={"last_promoted_at": promoted_at})
        promoted[target_name] = target_dir
    if candidate_dir.exists():
        try:
            remove_tree_with_retries(candidate_dir)
        except OSError as exc:
            LOGGER.warning("Candidate cleanup failed for %s: %s", candidate_dir, exc)
    return promoted


def run_prediction_bundle(
    bundle_dir: Path,
    engine,
    args: argparse.Namespace,
    prediction_mode: str,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    run_summary = load_run_summary(bundle_dir)
    requested_as_of = resolve_requested_as_of(engine, args)
    load_start = args.start_date or (requested_as_of - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    load_end = args.end_date or requested_as_of.strftime("%Y-%m-%d")

    bars = load_bars(
        engine,
        load_start,
        load_end,
        run_summary.get("symbols"),
        source_table=args.source_table,
        include_provisional=True,
    )
    bars = clip_bars_to_requested_as_of(bars, requested_as_of)
    if bars.empty:
        raise RuntimeError(f"No bars available for prediction between {load_start} and {load_end}.")
    dataset, regular = build_symbol_day_dataset(
        bars=bars,
        daily_windows=list(run_summary["daily_windows"]),
        slot_windows=list(run_summary["slot_windows"]),
        winsor_pct=float(run_summary["winsor_pct"]),
    )
    origin_bar_ts = latest_available_origin_bar_ts(dataset, requested_as_of)
    effective_as_of = pd.Timestamp(origin_bar_ts).normalize()
    as_of_rows = dataset[dataset["bar_ts"] == origin_bar_ts].copy()
    if as_of_rows.empty:
        raise RuntimeError(f"No origin rows found for {origin_bar_ts}.")

    feature_names = json.loads((bundle_dir / "feature_names.json").read_text(encoding="utf-8"))
    X_infer = align_features_for_inference(as_of_rows, feature_names)
    preds = predict_path_matrix(bundle_dir, X_infer)
    if preds.shape != (len(X_infer), EXPECTED_REGULAR_BARS):
        raise RuntimeError(f"Prediction matrix has invalid shape {preds.shape}.")
    if not np.isfinite(preds).all():
        raise RuntimeError("Prediction matrix contains NaN or inf values.")

    base_close = as_of_rows["close"].to_numpy(dtype=float)
    predicted_close = np.exp(preds) * base_close[:, None]
    predicted_full_day_return = preds[:, -1]
    if len(predicted_full_day_return) > 1 and float(np.std(predicted_full_day_return)) < 1e-10:
        raise RuntimeError("Predicted final-return distribution is degenerate.")

    predictions_dir = bundle_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = bundle_dir / "reports"
    if args.write_reports:
        reports_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    slot_labels = regular_slot_labels()

    for row_idx, (_, row) in enumerate(as_of_rows.iterrows()):
        symbol = str(row["symbol"])
        entry = {
            "symbol": symbol,
            "forecast_origin_date": str(pd.Timestamp(row["trade_date"]).date()),
            "forecast_origin_ts": str(pd.Timestamp(row["bar_ts"])),
            "predicted_session": "next_regular_session",
            "origin_close": float(row["close"]),
            "predicted_full_day_return": float(predicted_full_day_return[row_idx]),
            "predicted_direction": predicted_direction_label(np.array([predicted_full_day_return[row_idx]]))[0],
        }
        for horizon_idx in range(EXPECTED_REGULAR_BARS):
            entry[f"pred_log_return_h{horizon_idx:02d}"] = float(preds[row_idx, horizon_idx])
            entry[f"pred_close_h{horizon_idx:02d}"] = float(predicted_close[row_idx, horizon_idx])
            path_rows.append(
                {
                    "symbol": symbol,
                    "forecast_origin_date": str(pd.Timestamp(row["trade_date"]).date()),
                    "forecast_origin_ts": str(pd.Timestamp(row["bar_ts"])),
                    "predicted_session": "next_regular_session",
                    "bar_idx": horizon_idx,
                    "bar_time": slot_labels[horizon_idx],
                    "pred_log_return": float(preds[row_idx, horizon_idx]),
                    "pred_close": float(predicted_close[row_idx, horizon_idx]),
                }
            )
        prediction_rows.append(entry)
        if args.write_reports:
            plot_symbol_forecast(
                symbol=symbol,
                regular_history=regular,
                as_of_date=effective_as_of,
                predicted_close=predicted_close[row_idx],
                out_path=reports_dir / f"{symbol}_nextday_path.png",
                days_back=args.report_days_back,
            )

    predictions_df = pd.DataFrame(prediction_rows).sort_values("predicted_full_day_return", ascending=False).reset_index(drop=True)
    path_df = pd.DataFrame(path_rows)
    predictions_path = predictions_dir / "predictions.csv"
    path_output_path = predictions_dir / "predicted_path.csv"
    predictions_df.to_csv(predictions_path, index=False)
    path_df.to_csv(path_output_path, index=False)

    extra_artifacts = {
        "predictions_csv": str(predictions_path),
        "predicted_path_csv": str(path_output_path),
    }
    if args.write_reports:
        report_summary_path = reports_dir / "report_summary.md"
        with open(report_summary_path, "w", encoding="utf-8") as fh:
            fh.write("# Next-Session 15-Minute Path Forecast\n\n")
            fh.write(f"- Forecast origin date: {effective_as_of.date()}\n")
            fh.write(f"- Forecast origin bar: {origin_bar_ts}\n")
            fh.write(f"- Symbols scored: {len(predictions_df)}\n")
            fh.write(f"- Predictions file: `{predictions_path}`\n")
            fh.write(f"- Path file: `{path_output_path}`\n\n")
            fh.write("## Top predicted next-session returns\n\n")
            fh.write(
                predictions_df[["symbol", "predicted_full_day_return", "predicted_direction"]]
                .head(20)
                .to_markdown(index=False)
            )
            fh.write("\n")
        extra_artifacts.update(
            {
                "reports_dir": str(reports_dir),
                "report_summary": str(report_summary_path),
            }
        )

    update_run_summary_with_inference(bundle_dir, extra_artifacts)
    latest_bar_date = pd.Timestamp(bars["trade_date"].max()).normalize()
    model_id = (read_json_if_exists(bundle_dir / "metadata.json") or {}).get("model_id", bundle_dir.name)
    status_payload = {
        "mode": prediction_mode,
        "status": "success",
        "started_at": utc_now_str(),
        "finished_at": utc_now_str(),
        "requested_as_of_date": str(requested_as_of.date()),
        "effective_as_of_date": str(effective_as_of.date()),
        "effective_as_of_bar_ts": str(origin_bar_ts),
        "latest_bar_date": str(latest_bar_date.date()),
        "data_status": "latest_incomplete_ignored" if latest_bar_date > effective_as_of else "complete",
        "model_id": model_id,
        "predicted_symbols": int(len(predictions_df)),
        "predictions_csv": str(predictions_path),
        "predicted_path_csv": str(path_output_path),
        "fallback_reason": fallback_reason,
        "reports_written": bool(args.write_reports),
    }
    write_cycle_status(bundle_dir, status_payload)
    rewrite_bundle_artifacts(
        bundle_dir,
        extra_updates={
            "last_prediction_as_of_date": str(effective_as_of.date()),
            "last_prediction_as_of_bar_ts": str(origin_bar_ts),
        },
    )
    LOGGER.info("Saved prediction outputs to %s using bundle %s", predictions_path, bundle_dir)
    return status_payload


def production_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    run_root = resolve_registry_root(args)
    ensure_registry_layout(run_root)
    configure_logging(run_root, args.log_level)
    LOGGER.info("Using source: %s", describe_source(args))
    engine = open_source(args)

    result = train_candidate_bundle(run_root, args, engine, describe_source(args), profile_name="base")
    if not result["success"]:
        failed_path = move_bundle_to_failed(result["candidate_dir"], run_root / "failed", result["reason"])
        raise RuntimeError(f"Bootstrap training failed validation: {result['reason']} ({failed_path})")
    promoted = promote_candidate(run_root, result["candidate_dir"], profile_name="base", model_id=result["candidate_id"])
    root_status = {
        "mode": "bootstrap",
        "status": "success",
        "model_id": result["candidate_id"],
        "promoted_targets": {key: str(value) for key, value in promoted.items()},
        "finished_at": utc_now_str(),
    }
    write_json(run_root / "cycle_status.json", root_status)
    return root_status


def production_refresh(args: argparse.Namespace) -> dict[str, Any]:
    run_root = resolve_registry_root(args)
    ensure_registry_layout(run_root)
    configure_logging(run_root, args.log_level)
    LOGGER.info("Using source: %s", describe_source(args))
    engine = open_source(args)

    result = train_candidate_bundle(run_root, args, engine, describe_source(args), profile_name="refresh")
    if not result["success"]:
        failed_path = move_bundle_to_failed(result["candidate_dir"], run_root / "failed", result["reason"])
        raise RuntimeError(f"Refresh training failed validation: {result['reason']} ({failed_path})")
    promoted = promote_candidate(run_root, result["candidate_dir"], profile_name="refresh", model_id=result["candidate_id"])
    root_status = {
        "mode": "refresh",
        "status": "success",
        "model_id": result["candidate_id"],
        "promoted_targets": {key: str(value) for key, value in promoted.items()},
        "finished_at": utc_now_str(),
    }
    write_json(run_root / "cycle_status.json", root_status)
    return root_status


def production_predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = resolve_registry_root(args)
    ensure_registry_layout(run_root)
    configure_logging(run_root, args.log_level)
    LOGGER.info("Using source: %s", describe_source(args))
    engine = open_source(args)
    bundle_dir = resolve_direct_or_registry_bundle(run_root, "current")
    status = run_prediction_bundle(bundle_dir, engine, args, prediction_mode="predict")
    write_json(run_root / "cycle_status.json", status)
    return status


def production_cycle(args: argparse.Namespace) -> dict[str, Any]:
    run_root = resolve_registry_root(args)
    ensure_registry_layout(run_root)
    configure_logging(run_root, args.log_level)
    LOGGER.info("Using source: %s", describe_source(args))
    engine = open_source(args)

    refresh_status: dict[str, Any] | None = None
    fallback_reason: str | None = None
    try:
        result = train_candidate_bundle(run_root, args, engine, describe_source(args), profile_name="refresh")
        if result["success"]:
            promote_candidate(run_root, result["candidate_dir"], profile_name="refresh", model_id=result["candidate_id"])
            refresh_status = {
                "status": "success",
                "model_id": result["candidate_id"],
                "runtime_sec": result["runtime_sec"],
            }
        else:
            failed_path = move_bundle_to_failed(result["candidate_dir"], run_root / "failed", result["reason"])
            refresh_status = {
                "status": "failed",
                "reason": result["reason"],
                "failed_path": str(failed_path),
            }
            fallback_reason = result["reason"]
    except Exception as exc:
        refresh_status = {
            "status": "failed",
            "reason": str(exc),
        }
        fallback_reason = str(exc)

    current_bundle = run_root / "current"
    if not current_bundle.exists():
        raise RuntimeError("Cycle cannot continue because no promoted current bundle is available.")

    prediction_status = run_prediction_bundle(
        current_bundle,
        engine,
        args,
        prediction_mode="cycle",
        fallback_reason=fallback_reason,
    )
    root_status = {
        "mode": "cycle",
        "status": "success",
        "finished_at": utc_now_str(),
        "refresh": refresh_status,
        "prediction": prediction_status,
    }
    write_json(run_root / "cycle_status.json", root_status)
    return root_status


def resolve_replay_root(args: argparse.Namespace) -> Path:
    preferred = args.run_root
    if (
        preferred == default_run_root()
        and args.output_dir is None
        and args.run_dir is None
    ):
        preferred = None
    return resolve_runtime_path(preferred, fallback=PROJECT_ROOT / "artifacts" / "replay_nextday")


def resolve_replay_dates(replay_dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    replay_value = args.replay_date or args.as_of_date or args.end_date
    if replay_value is None:
        raise RuntimeError("Replay mode requires --replay-date, --as-of-date, or --end-date.")
    replay_date = pd.Timestamp(replay_value).normalize()
    replay_rows = replay_dataset[replay_dataset["trade_date"] == replay_date].copy()
    if replay_rows.empty:
        raise RuntimeError(f"No replay rows are available for {replay_date.date()}.")

    truth_candidates = sorted(
        {
            pd.Timestamp(value).normalize()
            for value in replay_rows["next_trade_date"].dropna().unique()
        }
    )
    if not truth_candidates:
        raise RuntimeError(
            f"Replay date {replay_date.date()} does not have a complete next-trading-day truth session."
        )
    if args.truth_date:
        truth_date = pd.Timestamp(args.truth_date).normalize()
        if truth_date not in truth_candidates:
            candidate_str = ", ".join(str(value.date()) for value in truth_candidates)
            raise RuntimeError(
                f"Requested truth date {truth_date.date()} is not valid for replay date {replay_date.date()}. "
                f"Valid truth dates: {candidate_str}."
            )
    else:
        truth_date = truth_candidates[0]
    if truth_date <= replay_date:
        raise RuntimeError(
            f"Truth date {truth_date.date()} must be after replay date {replay_date.date()}."
        )
    return replay_date, truth_date


def replay_step_name(cutoff_slot_idx: int) -> str:
    return f"slot_{cutoff_slot_idx:02d}"


def prepare_replay_root(run_root: Path, replay_date: pd.Timestamp) -> Path:
    replay_dir = run_root / replay_date.strftime("%Y-%m-%d")
    if replay_dir.exists():
        remove_tree_with_retries(replay_dir)
    for path in [
        replay_dir / "metrics",
        replay_dir / "predictions" / "base",
        replay_dir / "predictions" / "refresh",
        replay_dir / "paths" / "base",
        replay_dir / "paths" / "refresh",
        replay_dir / "models" / "static_base",
        replay_dir / "models" / "refresh_steps",
        replay_dir / "plots",
        replay_dir / "plots" / "symbols",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return replay_dir


def train_replay_bundle(
    bundle_dir: Path,
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    params: dict[str, Any],
    model_id: str,
    bundle_role: str,
    replay_date: pd.Timestamp,
    truth_date: pd.Timestamp,
    training_scope: str,
    cutoff_slot_idx: int | None = None,
) -> dict[str, Any]:
    if bundle_dir.exists():
        remove_tree_with_retries(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    models_dir = bundle_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    allow_gpu_fallback = args.device == "auto"
    feature_names, feature_types, final_params, feature_importance = train_full_model_set(
        df=train_df,
        params=params,
        allow_gpu_fallback=allow_gpu_fallback,
        model_dir=models_dir,
        parallel_horizons=args.parallel_horizons,
        target_definition=REPLAY_TARGET_DEFINITION,
    )
    save_feature_names(bundle_dir, feature_names)
    feature_csv_path = bundle_dir / "feature_importance_top25.csv"
    feature_importance.to_csv(feature_csv_path, index=False)
    feature_plot_path = bundle_dir / "feature_importance_top25.png"
    if bundle_role == "static_base" and not feature_importance.empty:
        save_feature_importance_plot(feature_importance, feature_plot_path)

    train_target = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    artifacts = {
        "models_dir": str(models_dir),
        "model_manifest": str(models_dir / "model_manifest.json"),
        "feature_names": str(bundle_dir / "feature_names.json"),
        "feature_importance_csv": str(feature_csv_path),
        "log_file": str(bundle_dir / "cycle_status.json"),
    }
    if feature_plot_path.exists():
        artifacts["feature_importance_png"] = str(feature_plot_path)

    run_summary = {
        "mode": "replay_intraday",
        "bundle_role": bundle_role,
        "model_id": model_id,
        "replay_date": str(replay_date.date()),
        "truth_date": str(truth_date.date()),
        "training_scope": training_scope,
        "target_mode": "next_regular_session_close_path_from_cutoff",
        "target_definition": REPLAY_TARGET_DEFINITION,
        "train_rows": int(len(train_df)),
        "train_symbols": int(train_df["symbol"].nunique()),
        "train_dates": int(train_df["trade_date"].nunique()),
        "train_start_date": str(pd.Timestamp(train_df["trade_date"].min()).date()),
        "train_end_date": str(pd.Timestamp(train_df["trade_date"].max()).date()),
        "cutoff_slot_idx": int(cutoff_slot_idx) if cutoff_slot_idx is not None else None,
        "daily_windows": args.daily_windows,
        "slot_windows": args.slot_windows,
        "winsor_pct": args.winsor_pct,
        "requested_device": args.device,
        "final_model_device": final_params.get("device"),
        "final_model_params": final_params,
        "feature_count": len(feature_names),
        "target_abs_max": float(np.nanmax(np.abs(train_target))) if train_target.size else float("nan"),
        "artifacts": artifacts,
    }
    metadata = {
        "model_type": "XGBoost_multi_horizon_intraday_replay",
        "model_version": model_id,
        "bundle_role": bundle_role,
        "replay_date": str(replay_date.date()),
        "truth_date": str(truth_date.date()),
        "cutoff_slot_idx": int(cutoff_slot_idx) if cutoff_slot_idx is not None else None,
        "prediction_horizon": "next_regular_session_15m_close_path",
        "target_definition": REPLAY_TARGET_DEFINITION,
        "horizon_count": EXPECTED_REGULAR_BARS,
        "n_features": len(feature_names),
        "n_rows": int(len(train_df)),
        "n_symbols": int(train_df["symbol"].nunique()),
        "n_dates": int(train_df["trade_date"].nunique()),
        "daily_windows": args.daily_windows,
        "slot_windows": args.slot_windows,
        "winsor_pct": args.winsor_pct,
        "device": final_params.get("device"),
        "model_id": model_id,
        "training_scope": training_scope,
        "feature_types": feature_types,
        "feature_count_check": len(feature_names),
        "artifacts": artifacts,
    }
    write_json(bundle_dir / "run_summary.json", run_summary)
    write_json(bundle_dir / "metadata.json", metadata)
    write_cycle_status(
        bundle_dir,
        {
            "mode": "replay_intraday",
            "status": "trained",
            "model_id": model_id,
            "bundle_role": bundle_role,
            "replay_date": str(replay_date.date()),
            "truth_date": str(truth_date.date()),
            "cutoff_slot_idx": int(cutoff_slot_idx) if cutoff_slot_idx is not None else None,
            "finished_at": utc_now_str(),
        },
    )
    return {
        "bundle_dir": bundle_dir,
        "model_id": model_id,
        "feature_names": feature_names,
        "final_model_device": final_params.get("device"),
    }


def load_bundle_model_id(bundle_dir: Path) -> str:
    metadata = read_json_if_exists(bundle_dir / "metadata.json") or {}
    return str(metadata.get("model_id", bundle_dir.name))


def compute_replay_metrics(
    actual_log_returns: np.ndarray,
    predicted_log_returns: np.ndarray,
    actual_close: np.ndarray,
    predicted_close: np.ndarray,
) -> dict[str, float]:
    if actual_log_returns.size == 0 or predicted_log_returns.size == 0:
        return {
            "path_rmse": float("nan"),
            "path_mae": float("nan"),
            "final_horizon_rmse": float("nan"),
            "final_horizon_mae": float("nan"),
            "direction_accuracy": float("nan"),
            "price_path_rmse": float("nan"),
            "final_price_rmse": float("nan"),
        }
    final_true = actual_log_returns[:, -1]
    final_pred = predicted_log_returns[:, -1]
    return {
        "path_rmse": float(np.sqrt(np.mean(np.square(actual_log_returns - predicted_log_returns)))),
        "path_mae": float(np.mean(np.abs(actual_log_returns - predicted_log_returns))),
        "final_horizon_rmse": rmse_no_kw(final_true, final_pred),
        "final_horizon_mae": float(mean_absolute_error(final_true, final_pred)),
        "direction_accuracy": summarize_direction(final_true, final_pred),
        "price_path_rmse": float(np.sqrt(np.mean(np.square(actual_close - predicted_close)))),
        "final_price_rmse": rmse_no_kw(actual_close[:, -1], predicted_close[:, -1]),
    }


def build_replay_prediction_frames(
    step_rows: pd.DataFrame,
    predicted_log_returns: np.ndarray,
    bundle_dir: Path,
    track_name: str,
    replay_date: pd.Timestamp,
    truth_date: pd.Timestamp,
    cutoff_slot_idx: int,
    fallback_used: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ordered_rows = step_rows.sort_values("symbol").reset_index(drop=True)
    actual_log_returns = ordered_rows[TARGET_COLUMNS].to_numpy(dtype=float)
    actual_close = ordered_rows[[f"next_close_h{idx:02d}" for idx in range(EXPECTED_REGULAR_BARS)]].to_numpy(dtype=float)
    cutoff_close = ordered_rows["cutoff_close"].to_numpy(dtype=float)
    predicted_close = np.exp(predicted_log_returns) * cutoff_close[:, None]
    model_id = load_bundle_model_id(bundle_dir)
    cutoff_time = str(ordered_rows["cutoff_time"].iloc[0])
    slot_labels = regular_slot_labels()

    summary_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(ordered_rows.iterrows()):
        final_actual = float(actual_log_returns[row_idx, -1])
        final_pred = float(predicted_log_returns[row_idx, -1])
        summary_rows.append(
            {
                "symbol": str(row["symbol"]),
                "replay_date": str(replay_date.date()),
                "truth_date": str(truth_date.date()),
                "cutoff_slot_idx": int(cutoff_slot_idx),
                "cutoff_time": cutoff_time,
                "model_track": track_name,
                "model_id": model_id,
                "fallback_used": bool(fallback_used),
                "cutoff_close": float(cutoff_close[row_idx]),
                "actual_final_close": float(actual_close[row_idx, -1]),
                "pred_final_close": float(predicted_close[row_idx, -1]),
                "actual_final_return": final_actual,
                "predicted_full_day_return": final_pred,
                "actual_direction": predicted_direction_label(np.array([final_actual]))[0],
                "predicted_direction": predicted_direction_label(np.array([final_pred]))[0],
                "final_abs_error": float(abs(final_actual - final_pred)),
            }
        )
        for horizon_idx in range(EXPECTED_REGULAR_BARS):
            path_rows.append(
                {
                    "symbol": str(row["symbol"]),
                    "replay_date": str(replay_date.date()),
                    "truth_date": str(truth_date.date()),
                    "cutoff_slot_idx": int(cutoff_slot_idx),
                    "cutoff_time": cutoff_time,
                    "model_track": track_name,
                    "model_id": model_id,
                    "fallback_used": bool(fallback_used),
                    "bar_idx": int(horizon_idx),
                    "bar_time": slot_labels[horizon_idx],
                    "cutoff_close": float(cutoff_close[row_idx]),
                    "actual_close": float(actual_close[row_idx, horizon_idx]),
                    "pred_close": float(predicted_close[row_idx, horizon_idx]),
                    "actual_log_return": float(actual_log_returns[row_idx, horizon_idx]),
                    "pred_log_return": float(predicted_log_returns[row_idx, horizon_idx]),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("predicted_full_day_return", ascending=False).reset_index(drop=True)
    path_df = pd.DataFrame(path_rows).sort_values(["symbol", "bar_idx"]).reset_index(drop=True)
    metrics = compute_replay_metrics(actual_log_returns, predicted_log_returns, actual_close, predicted_close)
    metrics["model_id"] = model_id
    return summary_df, path_df, metrics


def write_replay_step_outputs(
    replay_dir: Path,
    track_name: str,
    cutoff_slot_idx: int,
    predictions_df: pd.DataFrame,
    path_df: pd.DataFrame,
) -> tuple[Path, Path]:
    prediction_path = replay_dir / "predictions" / track_name / f"{replay_step_name(cutoff_slot_idx)}_predictions.csv"
    path_output_path = replay_dir / "paths" / track_name / f"{replay_step_name(cutoff_slot_idx)}_path.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    path_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(prediction_path, index=False)
    path_df.to_csv(path_output_path, index=False)
    return prediction_path, path_output_path


def plot_replay_series(
    step_metrics_df: pd.DataFrame,
    series: list[tuple[str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if step_metrics_df.empty:
        return
    plt.figure(figsize=(12, 5))
    plotted = False
    x_values = step_metrics_df["cutoff_slot_idx"].to_numpy(dtype=int)
    for column, label in series:
        if column not in step_metrics_df.columns:
            continue
        values = pd.to_numeric(step_metrics_df[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        plt.plot(x_values, values.to_numpy(dtype=float), marker="o", linewidth=1.8, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xticks(x_values, step_metrics_df["cutoff_time"].tolist(), rotation=45)
    plt.title(title)
    plt.xlabel("March 2 cutoff")
    plt.ylabel(ylabel)
    plt.tight_layout()
    if len(series) > 1:
        plt.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_symbol_cutoff_overlays(
    paths_df: pd.DataFrame,
    symbol: str,
    cutoff_indices: list[int],
    output_path: Path,
    title: str,
) -> None:
    symbol_df = paths_df[paths_df["symbol"] == symbol].copy()
    if symbol_df.empty:
        return
    actual_df = (
        symbol_df.sort_values(["cutoff_slot_idx", "bar_idx"])
        .drop_duplicates(subset=["bar_idx"])[["bar_idx", "bar_time", "actual_close"]]
        .sort_values("bar_idx")
    )
    plt.figure(figsize=(12, 5))
    plt.plot(
        actual_df["bar_idx"].to_numpy(dtype=int),
        actual_df["actual_close"].to_numpy(dtype=float),
        color="black",
        linewidth=2.5,
        label="Actual",
    )
    palette = sns.color_palette("viridis", n_colors=max(len(cutoff_indices), 2))
    plotted = False
    for idx, cutoff_idx in enumerate(cutoff_indices):
        cutoff_df = symbol_df[symbol_df["cutoff_slot_idx"] == cutoff_idx].sort_values("bar_idx")
        if cutoff_df.empty:
            continue
        plt.plot(
            cutoff_df["bar_idx"].to_numpy(dtype=int),
            cutoff_df["pred_close"].to_numpy(dtype=float),
            color=palette[idx],
            linewidth=1.6,
            label=f"Cutoff {str(cutoff_df['cutoff_time'].iloc[0])}",
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    tick_positions = list(range(0, EXPECTED_REGULAR_BARS, 5))
    tick_labels = [regular_slot_labels()[idx] for idx in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title(title)
    plt.ylabel("Close price")
    plt.xlabel("Next-day bar")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_symbol_evolution_heatmap(paths_df: pd.DataFrame, symbol: str, output_path: Path, title: str) -> None:
    symbol_df = paths_df[paths_df["symbol"] == symbol].copy()
    if symbol_df.empty:
        return
    pivot = (
        symbol_df.pivot_table(
            index="cutoff_time",
            columns="bar_idx",
            values="pred_close",
            aggfunc="mean",
        )
        .sort_index()
    )
    if pivot.empty:
        return
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="viridis")
    plt.title(title)
    plt.xlabel("Next-day bar index")
    plt.ylabel("March 2 cutoff")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def generate_replay_plots(
    replay_dir: Path,
    step_metrics_df: pd.DataFrame,
    base_paths_df: pd.DataFrame,
    refresh_paths_df: pd.DataFrame,
    top_symbols: list[str],
) -> None:
    plots_dir = replay_dir / "plots"
    step_metrics_df = step_metrics_df.copy()
    if "base_path_rmse" in step_metrics_df.columns and "refresh_path_rmse" in step_metrics_df.columns:
        step_metrics_df["refresh_path_rmse_uplift"] = step_metrics_df["base_path_rmse"] - step_metrics_df["refresh_path_rmse"]

    plot_replay_series(
        step_metrics_df,
        [("base_path_rmse", "Base"), ("refresh_path_rmse", "Refresh")],
        title="Path RMSE by March 2 cutoff",
        ylabel="RMSE",
        output_path=plots_dir / "path_rmse_vs_cutoff.png",
    )
    plot_replay_series(
        step_metrics_df,
        [("base_final_horizon_rmse", "Base"), ("refresh_final_horizon_rmse", "Refresh")],
        title="Final-horizon RMSE by March 2 cutoff",
        ylabel="RMSE",
        output_path=plots_dir / "final_horizon_rmse_vs_cutoff.png",
    )
    plot_replay_series(
        step_metrics_df,
        [("base_direction_accuracy", "Base"), ("refresh_direction_accuracy", "Refresh")],
        title="Final-horizon direction accuracy by March 2 cutoff",
        ylabel="Accuracy",
        output_path=plots_dir / "direction_accuracy_vs_cutoff.png",
    )
    plot_replay_series(
        step_metrics_df,
        [("refresh_train_runtime_sec", "Refresh train runtime")],
        title="Refresh retrain runtime by March 2 cutoff",
        ylabel="Seconds",
        output_path=plots_dir / "refresh_runtime_vs_cutoff.png",
    )
    plot_replay_series(
        step_metrics_df,
        [("refresh_path_rmse_uplift", "Base RMSE - Refresh RMSE")],
        title="Refresh improvement over static base",
        ylabel="RMSE uplift",
        output_path=plots_dir / "refresh_minus_base_improvement.png",
    )

    key_cutoffs = sorted({0, 5, 10, 15, 20, EXPECTED_REGULAR_BARS - 1})
    for symbol in top_symbols:
        plot_symbol_cutoff_overlays(
            base_paths_df,
            symbol,
            cutoff_indices=key_cutoffs,
            output_path=plots_dir / "symbols" / f"{symbol}_base_overlays.png",
            title=f"{symbol}: static base vs actual March 3 path",
        )
        plot_symbol_cutoff_overlays(
            refresh_paths_df,
            symbol,
            cutoff_indices=key_cutoffs,
            output_path=plots_dir / "symbols" / f"{symbol}_refresh_overlays.png",
            title=f"{symbol}: refresh predictions vs actual March 3 path",
        )
        plot_symbol_evolution_heatmap(
            refresh_paths_df,
            symbol,
            output_path=plots_dir / "symbols" / f"{symbol}_refresh_evolution.png",
            title=f"{symbol}: refresh path evolution across March 2 cutoffs",
        )


def replay_intraday(args: argparse.Namespace) -> dict[str, Any]:
    run_root = resolve_replay_root(args)
    LOGGER.info("Using source: %s", describe_source(args))
    engine = open_source(args)

    load_start, load_end = determine_dates(engine, args)
    bars = load_bars(
        engine,
        load_start,
        load_end,
        args.symbols,
        source_table=args.source_table,
        include_provisional=False,
    )
    if bars.empty:
        raise RuntimeError(f"No bars found between {load_start} and {load_end}.")

    replay_dataset, regular = build_cutoff_replay_dataset(
        bars=bars,
        daily_windows=args.daily_windows,
        slot_windows=args.slot_windows,
        winsor_pct=args.winsor_pct,
    )
    replay_date, truth_date = resolve_replay_dates(replay_dataset, args)
    replay_dir = prepare_replay_root(run_root, replay_date)
    configure_logging(replay_dir, args.log_level, log_filename="replay.log")

    replay_rows = replay_dataset[
        (replay_dataset["trade_date"] == replay_date)
        & (replay_dataset["next_trade_date"] == truth_date)
    ].copy()
    replay_rows = replay_rows.dropna(subset=TARGET_COLUMNS).sort_values(["cutoff_slot_idx", "symbol"]).reset_index(drop=True)
    if replay_rows["cutoff_slot_idx"].nunique() != EXPECTED_REGULAR_BARS:
        raise RuntimeError(
            f"Replay date {replay_date.date()} does not contain all {EXPECTED_REGULAR_BARS} cutoff steps."
        )
    if replay_rows[[f"next_close_h{idx:02d}" for idx in range(EXPECTED_REGULAR_BARS)]].isna().any().any():
        raise RuntimeError(f"Truth path for {truth_date.date()} is incomplete.")

    history_rows = replay_dataset[
        (replay_dataset["trade_date"] < replay_date)
        & replay_dataset["next_trade_date"].notna()
    ].dropna(subset=TARGET_COLUMNS).copy()
    if history_rows.empty:
        raise RuntimeError(f"No historical training rows exist before replay date {replay_date.date()}.")

    base_train_df = restrict_df_to_recent_months(history_rows, args.base_window_months)
    if len(base_train_df) < args.min_train_rows:
        raise RuntimeError(
            f"Static base training rows are {len(base_train_df)} < {args.min_train_rows}."
        )
    base_train_df = sanitize_training_targets(base_train_df, min_train_rows=args.min_train_rows)
    base_params, requested_device = resolve_xgb_params(args)
    base_params.update({k: v for k, v in BEST_FIXED_VARIANT.items() if k != "name"})
    static_base_model_id = f"static_base_{replay_date.strftime('%Y%m%d')}_{utc_compact_id()}"
    static_base_info = train_replay_bundle(
        bundle_dir=replay_dir / "models" / "static_base",
        train_df=base_train_df,
        args=args,
        params=base_params,
        model_id=static_base_model_id,
        bundle_role="static_base",
        replay_date=replay_date,
        truth_date=truth_date,
        training_scope=f"18m history through {pd.Timestamp(base_train_df['trade_date'].max()).date()}",
    )

    base_paths_all: list[pd.DataFrame] = []
    refresh_paths_all: list[pd.DataFrame] = []
    step_metrics: list[dict[str, Any]] = []
    last_successful_refresh_dir: Path | None = None
    last_successful_refresh_model_id: str | None = None

    for cutoff_slot_idx in range(EXPECTED_REGULAR_BARS):
        step_rows = replay_rows[replay_rows["cutoff_slot_idx"] == cutoff_slot_idx].copy()
        if step_rows.empty:
            raise RuntimeError(f"Replay step {cutoff_slot_idx} has no rows for {replay_date.date()}.")
        cutoff_time = str(step_rows["cutoff_time"].iloc[0])

        base_pred_t0 = time.time()
        base_feature_names = json.loads((static_base_info["bundle_dir"] / "feature_names.json").read_text(encoding="utf-8"))
        base_X = align_features_for_inference(step_rows.sort_values("symbol").reset_index(drop=True), base_feature_names)
        base_preds = predict_path_matrix(static_base_info["bundle_dir"], base_X)
        base_pred_runtime = float(time.time() - base_pred_t0)
        base_summary_df, base_path_df, base_metrics = build_replay_prediction_frames(
            step_rows=step_rows,
            predicted_log_returns=base_preds,
            bundle_dir=static_base_info["bundle_dir"],
            track_name="base",
            replay_date=replay_date,
            truth_date=truth_date,
            cutoff_slot_idx=cutoff_slot_idx,
            fallback_used=False,
        )
        write_replay_step_outputs(replay_dir, "base", cutoff_slot_idx, base_summary_df, base_path_df)
        base_paths_all.append(base_path_df)

        refresh_train_runtime_sec = float("nan")
        refresh_prediction_runtime_sec = float("nan")
        refresh_success = False
        refresh_fallback_used = False
        refresh_failure_reason: str | None = None
        refresh_rows_used = 0
        refresh_model_dir = replay_dir / "models" / "refresh_steps" / replay_step_name(cutoff_slot_idx)
        refresh_train_df = history_rows[history_rows["cutoff_slot_idx"] <= cutoff_slot_idx].copy()
        refresh_train_df = restrict_df_to_recent_trading_days(refresh_train_df, args.fast_refresh_days)
        refresh_rows_used = int(len(refresh_train_df))

        active_refresh_bundle: Path | None = None
        active_refresh_model_id: str | None = None

        if refresh_rows_used < args.min_train_rows:
            refresh_failure_reason = f"insufficient_rows_{refresh_rows_used}"
        else:
            refresh_train_df = sanitize_training_targets(refresh_train_df, min_train_rows=args.min_train_rows)
            refresh_params, _ = resolve_xgb_params(args)
            refresh_params.update({k: v for k, v in FAST_REFRESH_VARIANT.items() if k != "name"})
            refresh_model_id = f"refresh_{replay_date.strftime('%Y%m%d')}_{cutoff_slot_idx:02d}_{utc_compact_id()}"
            train_t0 = time.time()
            try:
                refresh_info = train_replay_bundle(
                    bundle_dir=refresh_model_dir,
                    train_df=refresh_train_df,
                    args=args,
                    params=refresh_params,
                    model_id=refresh_model_id,
                    bundle_role="refresh_step",
                    replay_date=replay_date,
                    truth_date=truth_date,
                    training_scope=f"{args.fast_refresh_days}d history up to cutoff {cutoff_time}",
                    cutoff_slot_idx=cutoff_slot_idx,
                )
                refresh_train_runtime_sec = float(time.time() - train_t0)
                active_refresh_bundle = refresh_info["bundle_dir"]
                active_refresh_model_id = refresh_info["model_id"]
                last_successful_refresh_dir = active_refresh_bundle
                last_successful_refresh_model_id = active_refresh_model_id
                refresh_success = True
            except Exception as exc:
                refresh_train_runtime_sec = float(time.time() - train_t0)
                refresh_failure_reason = str(exc)
                refresh_model_dir.mkdir(parents=True, exist_ok=True)
                write_cycle_status(
                    refresh_model_dir,
                    {
                        "mode": "replay_intraday",
                        "status": "failed",
                        "cutoff_slot_idx": cutoff_slot_idx,
                        "cutoff_time": cutoff_time,
                        "reason": refresh_failure_reason,
                        "runtime_sec": refresh_train_runtime_sec,
                        "rows_used": refresh_rows_used,
                        "finished_at": utc_now_str(),
                    },
                )

        if active_refresh_bundle is None and last_successful_refresh_dir is not None:
            active_refresh_bundle = last_successful_refresh_dir
            active_refresh_model_id = last_successful_refresh_model_id
            refresh_fallback_used = True

        if active_refresh_bundle is not None:
            refresh_pred_t0 = time.time()
            refresh_feature_names = json.loads((active_refresh_bundle / "feature_names.json").read_text(encoding="utf-8"))
            refresh_X = align_features_for_inference(step_rows.sort_values("symbol").reset_index(drop=True), refresh_feature_names)
            refresh_preds = predict_path_matrix(active_refresh_bundle, refresh_X)
            refresh_prediction_runtime_sec = float(time.time() - refresh_pred_t0)
            refresh_summary_df, refresh_path_df, refresh_metrics = build_replay_prediction_frames(
                step_rows=step_rows,
                predicted_log_returns=refresh_preds,
                bundle_dir=active_refresh_bundle,
                track_name="refresh",
                replay_date=replay_date,
                truth_date=truth_date,
                cutoff_slot_idx=cutoff_slot_idx,
                fallback_used=refresh_fallback_used,
            )
        else:
            refresh_summary_df = pd.DataFrame()
            refresh_path_df = pd.DataFrame()
            refresh_metrics = {
                "path_rmse": float("nan"),
                "path_mae": float("nan"),
                "final_horizon_rmse": float("nan"),
                "final_horizon_mae": float("nan"),
                "direction_accuracy": float("nan"),
                "price_path_rmse": float("nan"),
                "final_price_rmse": float("nan"),
                "model_id": None,
            }

        write_replay_step_outputs(replay_dir, "refresh", cutoff_slot_idx, refresh_summary_df, refresh_path_df)
        if not refresh_path_df.empty:
            refresh_paths_all.append(refresh_path_df)

        step_row = {
            "replay_date": str(replay_date.date()),
            "truth_date": str(truth_date.date()),
            "cutoff_slot_idx": int(cutoff_slot_idx),
            "cutoff_time": cutoff_time,
            "rows_used_for_refresh": int(refresh_rows_used),
            "refresh_train_runtime_sec": refresh_train_runtime_sec,
            "base_prediction_runtime_sec": base_pred_runtime,
            "refresh_prediction_runtime_sec": refresh_prediction_runtime_sec,
            "static_base_model_id": base_metrics["model_id"],
            "refresh_model_id": refresh_metrics.get("model_id") or active_refresh_model_id,
            "refresh_success": bool(refresh_success),
            "refresh_fallback_used": bool(refresh_fallback_used),
            "refresh_failure_reason": refresh_failure_reason,
            "base_path_rmse": base_metrics["path_rmse"],
            "base_path_mae": base_metrics["path_mae"],
            "base_final_horizon_rmse": base_metrics["final_horizon_rmse"],
            "base_final_horizon_mae": base_metrics["final_horizon_mae"],
            "base_direction_accuracy": base_metrics["direction_accuracy"],
            "base_price_path_rmse": base_metrics["price_path_rmse"],
            "refresh_path_rmse": refresh_metrics["path_rmse"],
            "refresh_path_mae": refresh_metrics["path_mae"],
            "refresh_final_horizon_rmse": refresh_metrics["final_horizon_rmse"],
            "refresh_final_horizon_mae": refresh_metrics["final_horizon_mae"],
            "refresh_direction_accuracy": refresh_metrics["direction_accuracy"],
            "refresh_price_path_rmse": refresh_metrics["price_path_rmse"],
        }
        step_metrics.append(step_row)
        LOGGER.info(
            "Replay cutoff=%s slot=%02d base_rmse=%.6f refresh_rmse=%s refresh_success=%s fallback=%s",
            cutoff_time,
            cutoff_slot_idx,
            base_metrics["path_rmse"],
            (
                f"{refresh_metrics['path_rmse']:.6f}"
                if np.isfinite(refresh_metrics["path_rmse"])
                else "nan"
            ),
            refresh_success,
            refresh_fallback_used,
        )

    step_metrics_df = pd.DataFrame(step_metrics).sort_values("cutoff_slot_idx").reset_index(drop=True)
    metrics_csv_path = replay_dir / "metrics" / "step_metrics.csv"
    metrics_json_path = replay_dir / "metrics" / "step_metrics.json"
    step_metrics_df.to_csv(metrics_csv_path, index=False)
    write_json(metrics_json_path, {"steps": step_metrics_df.to_dict(orient="records")})

    base_paths_df = pd.concat(base_paths_all, ignore_index=True) if base_paths_all else pd.DataFrame()
    refresh_paths_df = pd.concat(refresh_paths_all, ignore_index=True) if refresh_paths_all else pd.DataFrame()

    volume_rank = (
        regular[regular["trade_date"] == replay_date]
        .groupby("symbol", observed=True)["volume"]
        .sum()
        .sort_values(ascending=False)
    )
    top_symbols = volume_rank.head(max(0, int(args.visual_symbol_count))).index.tolist()
    generate_replay_plots(
        replay_dir=replay_dir,
        step_metrics_df=step_metrics_df,
        base_paths_df=base_paths_df,
        refresh_paths_df=refresh_paths_df,
        top_symbols=top_symbols,
    )

    replay_summary = {
        "mode": "replay_intraday",
        "status": "success",
        "replay_date": str(replay_date.date()),
        "truth_date": str(truth_date.date()),
        "load_start_date": load_start,
        "load_end_date": load_end,
        "requested_device": args.device,
        "resolved_device": requested_device,
        "base_model_id": static_base_info["model_id"],
        "static_base_bundle": str(static_base_info["bundle_dir"]),
        "refresh_successful_steps": int(step_metrics_df["refresh_success"].sum()),
        "refresh_fallback_steps": int(step_metrics_df["refresh_fallback_used"].sum()),
        "refresh_unavailable_steps": int(
            (~step_metrics_df["refresh_success"] & ~step_metrics_df["refresh_fallback_used"]).sum()
        ),
        "processed_steps": int(len(step_metrics_df)),
        "top_visual_symbols": top_symbols,
        "artifacts": {
            "replay_log": str(replay_dir / "replay.log"),
            "metrics_csv": str(metrics_csv_path),
            "metrics_json": str(metrics_json_path),
            "predictions_base_dir": str(replay_dir / "predictions" / "base"),
            "predictions_refresh_dir": str(replay_dir / "predictions" / "refresh"),
            "paths_base_dir": str(replay_dir / "paths" / "base"),
            "paths_refresh_dir": str(replay_dir / "paths" / "refresh"),
            "models_static_base_dir": str(replay_dir / "models" / "static_base"),
            "models_refresh_steps_dir": str(replay_dir / "models" / "refresh_steps"),
            "plots_dir": str(replay_dir / "plots"),
        },
        "finished_at": utc_now_str(),
    }
    write_json(replay_dir / "replay_summary.json", replay_summary)
    LOGGER.info("Saved replay artifacts to %s", replay_dir)
    return replay_summary


def simulate_warm_refresh(args: argparse.Namespace) -> dict[str, Any]:
    sim_wall_t0 = time.time()
    run_root = resolve_registry_root(args)
    ensure_registry_layout(run_root)
    configure_logging(run_root, args.log_level)
    LOGGER.info("=== simulate_warm_refresh | replay_date=%s ===", args.replay_date)

    if not args.replay_date:
        raise ValueError("--replay-date is required for simulate_warm_refresh.")
    replay_date = pd.Timestamp(args.replay_date).normalize()
    engine = open_source(args)

    # --- Data retrieval ---
    t_load = time.time()
    load_start = (replay_date - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    load_end = replay_date.strftime("%Y-%m-%d")
    bars = load_bars(engine, load_start, load_end, args.symbols, source_table=args.source_table)
    retrieval_sec = time.time() - t_load
    LOGGER.info("[TIMER] Data retrieval: %.2fs  rows=%d", retrieval_sec, len(bars))

    if bars.empty:
        raise RuntimeError(f"No bars loaded between {load_start} and {load_end}. Check DB tunnel and data.")

    # --- Feature build ---
    t_feat = time.time()
    dataset, _ = build_symbol_day_dataset(
        bars=bars,
        daily_windows=args.daily_windows,
        slot_windows=args.slot_windows,
        winsor_pct=args.winsor_pct,
    )
    feature_build_sec = time.time() - t_feat
    LOGGER.info("[TIMER] Feature build: %.2fs  rows=%d", feature_build_sec, len(dataset))

    # Filter to replay date for slot timestamps
    replay_day_mask = pd.to_datetime(dataset["trade_date"]).dt.normalize() == replay_date
    replay_day_rows = dataset[replay_day_mask].copy()
    if replay_day_rows.empty:
        raise RuntimeError(f"No feature rows for replay_date {replay_date.date()} — check DB data.")

    replay_slots = sorted(pd.to_datetime(replay_day_rows["window_ts"]).unique())
    LOGGER.info("Replay date %s: %d bar slots found", replay_date.date(), len(replay_slots))

    # --- Starting init_model_paths from current/ bundle (base model) ---
    current_dir = registry_paths(run_root)["current"]
    init_model_paths: list[str | None] = [None] * EXPECTED_REGULAR_BARS
    if current_dir.exists():
        for idx in range(EXPECTED_REGULAR_BARS):
            p = current_dir / "models" / f"horizon_{idx:02d}.json"
            if p.exists():
                init_model_paths[idx] = str(p)
        found = sum(1 for p in init_model_paths if p is not None)
        LOGGER.info("Warm-start: %d/%d base models loaded from %s", found, EXPECTED_REGULAR_BARS, current_dir)
    else:
        LOGGER.warning("No current/ bundle found at %s — training from scratch.", current_dir)

    # --- Simulation output directory ---
    if getattr(args, "simulation_out", None):
        sim_dir = Path(args.simulation_out)
    else:
        sim_dir = run_root / "simulations" / replay_date.strftime("%Y%m%d")
    sim_dir.mkdir(parents=True, exist_ok=True)

    base_params, resolved_device, allow_gpu_fallback = resolve_training_setup(args, profile_name="refresh")
    warm_trees = int(getattr(args, "warm_trees", 30))
    warm_days = int(getattr(args, "warm_refresh_days", 1))
    slot_labels = regular_slot_labels()
    base_n_estimators = int(BEST_FIXED_VARIANT["n_estimators"])

    step_records: list[dict[str, Any]] = []

    for step_idx, slot_ts in enumerate(replay_slots):
        as_of_ts = pd.Timestamp(slot_ts)
        slot_label = slot_labels[step_idx] if step_idx < len(slot_labels) else str(as_of_ts.time())
        step_dir = sim_dir / f"step_{step_idx:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "predictions").mkdir(exist_ok=True)

        # Clip full dataset to this bar's timestamp
        clipped = dataset[pd.to_datetime(dataset["window_ts"]) <= as_of_ts].copy()
        train_ready = clipped.dropna(subset=TARGET_COLUMNS).copy()
        train_df = restrict_df_to_recent_trading_days(train_ready, warm_days)

        if len(train_df) < 10:
            LOGGER.warning("Step %02d | %s | only %d training rows — skipping.", step_idx, slot_label, len(train_df))
            step_records.append({"step": step_idx, "as_of_ts": str(as_of_ts), "slot_label": slot_label, "skipped": True})
            continue

        step_params = base_params.copy()
        step_params.update({k: v for k, v in FAST_REFRESH_VARIANT.items() if k != "name"})
        step_params["n_estimators"] = warm_trees

        # --- Warm-start training ---
        t_train = time.time()
        feature_names, _feature_types, _final_params, _ = train_full_model_set(
            df=train_df,
            params=step_params,
            allow_gpu_fallback=allow_gpu_fallback,
            model_dir=step_dir / "models",
            parallel_horizons=getattr(args, "parallel_horizons", 1),
            init_model_paths=init_model_paths,
        )
        train_sec = time.time() - t_train

        # Save feature_names.json (required by predict_path_matrix)
        save_feature_names(step_dir, feature_names)

        # Update init_model_paths for next step
        init_model_paths = [
            str(step_dir / "models" / f"horizon_{idx:02d}.json")
            for idx in range(EXPECTED_REGULAR_BARS)
        ]

        # --- Inference ---
        t_infer = time.time()
        as_of_rows = replay_day_rows[pd.to_datetime(replay_day_rows["window_ts"]) == as_of_ts].copy()
        if as_of_rows.empty:
            as_of_rows = (
                clipped.sort_values("window_ts")
                .groupby("symbol", observed=True)
                .tail(1)
                .reset_index(drop=True)
            )
        as_of_rows = as_of_rows.sort_values("symbol").reset_index(drop=True)
        X_inf = align_features_for_inference(as_of_rows, feature_names)
        preds = predict_path_matrix(step_dir, X_inf)
        infer_sec = time.time() - t_infer

        predicted_full_day_return = preds[:, -1]
        prediction_rows_out = []
        for row_idx, (_, row) in enumerate(as_of_rows.iterrows()):
            entry: dict[str, Any] = {
                "symbol": str(row["symbol"]),
                "step": step_idx,
                "as_of_ts": str(as_of_ts),
                "slot_label": slot_label,
                "predicted_full_day_return": float(predicted_full_day_return[row_idx]),
                "predicted_direction": predicted_direction_label(np.array([predicted_full_day_return[row_idx]]))[0],
            }
            for h_idx in range(EXPECTED_REGULAR_BARS):
                entry[f"pred_log_return_h{h_idx:02d}"] = float(preds[row_idx, h_idx])
            prediction_rows_out.append(entry)

        predictions_df = (
            pd.DataFrame(prediction_rows_out)
            .sort_values("predicted_full_day_return", ascending=False)
            .reset_index(drop=True)
        )
        predictions_df.to_csv(step_dir / "predictions" / "predictions.csv", index=False)

        meta: dict[str, Any] = {
            "step": step_idx,
            "as_of_ts": str(as_of_ts),
            "slot_label": slot_label,
            "warm_trees_added": warm_trees,
            "base_trees": base_n_estimators,
            "total_trees_estimate": base_n_estimators + (step_idx + 1) * warm_trees,
            "train_rows": int(len(train_df)),
            "inference_symbols": int(len(predictions_df)),
            "train_sec": round(train_sec, 3),
            "inference_sec": round(infer_sec, 3),
            "total_step_sec": round(train_sec + infer_sec, 3),
            "device": resolved_device,
        }
        write_json(step_dir / "metadata.json", meta)
        LOGGER.info(
            "[TIMER] Step %02d | %s | train=%.2fs | infer=%.2fs | symbols=%d",
            step_idx, slot_label, train_sec, infer_sec, len(predictions_df),
        )
        step_records.append(meta)

    total_wall_sec = time.time() - sim_wall_t0
    LOGGER.info("[TIMER] Total simulation wall time: %.2fs", total_wall_sec)

    summary: dict[str, Any] = {
        "mode": "simulate_warm_refresh",
        "status": "success",
        "replay_date": str(replay_date.date()),
        "run_root": str(run_root),
        "sim_dir": str(sim_dir),
        "warm_trees_per_step": warm_trees,
        "warm_refresh_days": warm_days,
        "steps_completed": sum(1 for r in step_records if not r.get("skipped")),
        "data_retrieval_sec": round(retrieval_sec, 3),
        "feature_build_sec": round(feature_build_sec, 3),
        "total_wall_sec": round(total_wall_sec, 3),
        "steps": step_records,
        "finished_at": utc_now_str(),
    }
    write_json(sim_dir / "simulation_summary.json", summary)
    LOGGER.info("Simulation complete. Artifacts at %s", sim_dir)
    return summary


def production_warm_refresh(args: argparse.Namespace) -> dict[str, Any]:
    run_root = resolve_registry_root(args)
    ensure_registry_layout(run_root)
    configure_logging(run_root, args.log_level)
    LOGGER.info("Using source: %s", describe_source(args))
    engine = open_source(args)

    # Locate existing horizon models in current/ to warm-start from
    current_dir = registry_paths(run_root)["current"]
    init_model_paths: list[str | None] = [None] * EXPECTED_REGULAR_BARS
    if current_dir.exists():
        models_dir = current_dir / "models"
        for idx in range(EXPECTED_REGULAR_BARS):
            p = models_dir / f"horizon_{idx:02d}.json"
            if p.exists():
                init_model_paths[idx] = str(p)
        found = sum(1 for p in init_model_paths if p is not None)
        LOGGER.info("Warm-start: found %d/%d horizon models in %s", found, EXPECTED_REGULAR_BARS, current_dir)
        if found == 0:
            LOGGER.warning("No existing models found — warm_refresh will train from scratch.")
    else:
        LOGGER.warning("No current/ bundle at %s — training from scratch.", current_dir)

    result = train_candidate_bundle(
        run_root, args, engine, describe_source(args),
        profile_name="warm_refresh",
        init_model_paths=init_model_paths,
    )
    if not result["success"]:
        failed_path = move_bundle_to_failed(result["candidate_dir"], run_root / "failed", result["reason"])
        raise RuntimeError(f"Warm refresh failed: {result['reason']} ({failed_path})")
    promoted = promote_candidate(run_root, result["candidate_dir"], profile_name="refresh", model_id=result["candidate_id"])
    root_status = {
        "mode": "warm_refresh",
        "status": "success",
        "model_id": result["candidate_id"],
        "warm_trees": int(getattr(args, "warm_trees", 30)),
        "warm_refresh_days": int(getattr(args, "warm_refresh_days", 1)),
        "promoted_targets": {key: str(value) for key, value in promoted.items()},
        "finished_at": utc_now_str(),
    }
    write_json(run_root / "cycle_status.json", root_status)
    return root_status


def main() -> None:
    args = parse_args()
    args.mode = canonical_mode(args.mode)
    args.windows_months = [int(args.base_window_months)]
    if args.mode == "bootstrap":
        production_bootstrap(args)
    elif args.mode == "refresh":
        production_refresh(args)
    elif args.mode == "predict":
        production_predict(args)
    elif args.mode == "cycle":
        production_cycle(args)
    elif args.mode == "replay_intraday":
        replay_intraday(args)
    elif args.mode == "warm_refresh":
        production_warm_refresh(args)
    elif args.mode == "simulate_warm_refresh":
        simulate_warm_refresh(args)


if __name__ == "__main__":
    main()
