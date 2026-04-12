"""
High-quality refetch for regular-session 15-minute market data.

This script fetches:
1. Current S&P 500 constituents from FMP.
2. Regular-session 15-minute bars (09:30–16:00 America/New_York) for:
   - all current S&P 500 stocks
   - gold (`GCUSD`)
   - crude oil (`CLUSD`)
3. Daily Treasury rates for:
   - 5-year
   - 10-year
   - 13-week (assumed from the user's "13 year" request because 13-year is not
     a standard US Treasury maturity series in FMP/UST datasets)

Data-quality rules:
- The script derives the expected regular-session timestamps for each trade date
  from a dedicated SPY reference fetch. This handles shortened sessions and DST
  transitions without fabricating bars.
- Every target symbol must match the reference schedule exactly for every
  reference trade date in the requested range.
- Missing dates are re-fetched day-by-day up to `--max-retries`.
- If any symbol still has unresolved gaps after retries, the script exits
  non-zero and reports them instead of silently loading partial data.
- Duplicate timestamps and null OHLCV rows are rejected.

Rows are written into `ml.market_data_15m`, and the existing warehouse feature
  columns are recomputed using the same logic as the targeted backfill script.
Treasury rates are stored in `ml.macro_indicator_daily`.

Examples:
    python ml/scripts/refetch_market_data_15m_quality.py --start-date 2024-03-25 --end-date 2026-03-23
    python ml/scripts/refetch_market_data_15m_quality.py --limit-symbols 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from backfill_market_data_15m_from_fmp import (
    connect_db,
    recompute_features,
    update_features,
    upsert_core_rows,
)


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml"
load_dotenv(ROOT / ".env")
load_dotenv(ML_ROOT / ".env")

FMP_API_KEY = os.getenv("FMP_API_KEY")
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
NY_TZ = "America/New_York"
REGULAR_OPEN = "09:30"
REGULAR_CLOSE = "16:00"
REGULAR_BARS = 26
REQUEST_TIMEOUT = 60
REQUEST_SLEEP_SECONDS = 0.35
RETRY_BACKOFF_SECONDS = (2, 5, 15)
DEFAULT_CHUNK_DAYS = 5
DEFAULT_SIMPLE_CHUNK_DAYS = 30
MASSIVE_CHUNK_DAYS = 45
MAX_ABS_PRICE = 999_999_999_999.0

SP500_CACHE = ML_ROOT / "data" / ".sp500_tickers_cache.json"
REPORT_DIR = ML_ROOT / "data" / "quality_refetch_reports"
TREASURY_TABLE = "ml.macro_indicator_daily"
SIMPLE_FMP_CHECKPOINT = REPORT_DIR / "simple_fmp_checkpoint.json"
FILTERED_ROWS_REPORT = REPORT_DIR / "filtered_intraday_rows.ndjson"

STOCK_INTRADAY_URLS = (
    "https://financialmodelingprep.com/stable/historical-chart/15min",
    "https://financialmodelingprep.com/api/v3/historical-chart/15min/{symbol}",
)
SP500_URLS = (
    "https://financialmodelingprep.com/stable/sp500-constituent",
    "https://financialmodelingprep.com/api/v3/sp500_constituent",
)
TREASURY_URLS = (
    "https://financialmodelingprep.com/stable/treasury-rates",
    "https://financialmodelingprep.com/api/v4/treasury",
)

COMMODITY_SYMBOLS = {
    "GCUSD": "gold",
    "CLUSD": "crude_oil",
}

MASSIVE_AGGS_URL = "https://api.massive.com/v2/aggs/ticker/{symbol}/range/15/minute/{start}/{end}"

TREASURY_SERIES = {
    "US_TREASURY_13W": "month3",
    "US_TREASURY_5Y": "year5",
    "US_TREASURY_10Y": "year10",
}


@dataclass(frozen=True)
class MissingDay:
    symbol: str
    trade_date: date
    missing_count: int
    expected_count: int
    actual_count: int
    missing_times_local: tuple[str, ...]


@dataclass(frozen=True)
class FallbackFill:
    symbol: str
    provider: str
    provider_symbol: str
    trade_date: date
    rows_added: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refetch high-quality regular-session 15-minute market data.")
    parser.add_argument(
        "--mode",
        choices=("quality", "simple-fmp"),
        default="quality",
        help="`quality` enforces full schedule coverage with fallback; `simple-fmp` bulk-loads FMP data without gap blocking.",
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS, help="Days per intraday API request.")
    parser.add_argument("--max-retries", type=int, default=3, help="Day-level refetch passes for unresolved gaps.")
    parser.add_argument("--limit-symbols", type=int, default=None, help="Only fetch the first N target symbols.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated explicit symbol list. When set, overrides the default S&P 500 universe selection.",
    )
    parser.add_argument(
        "--symbols-file",
        default=None,
        help="Path to a newline-delimited symbol list. Merged with --symbols when both are provided.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume simple-FMP mode from the last committed symbol in the checkpoint file when available.",
    )
    parser.add_argument("--start-from-symbol", default=None, help="Start processing at this symbol (inclusive).")
    parser.add_argument(
        "--allow-unresolved-symbol",
        action="append",
        dest="allow_unresolved_symbols",
        default=[],
        help="Symbol allowed to remain incomplete without failing the run. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and validate without writing to Postgres.")
    parser.add_argument("--skip-treasuries", action="store_true", help="Skip Treasury-rate fetch/storage.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def ensure_api_key() -> None:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not found. Set it in .env or ml/.env.")


def iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "maverick-refetch/1.0"})
    return session


def build_massive_url(symbol: str, start_date: date, end_date: date) -> str:
    return MASSIVE_AGGS_URL.format(symbol=symbol, start=start_date.isoformat(), end=end_date.isoformat())


def request_json(session: requests.Session, url: str, params: dict[str, Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt >= len(RETRY_BACKOFF_SECONDS):
                break
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    assert last_error is not None
    raise last_error


def fetch_sp500_tickers(session: requests.Session, force_refresh: bool = False) -> list[str]:
    if SP500_CACHE.exists() and not force_refresh:
        cached = json.loads(SP500_CACHE.read_text())
        tickers = cached.get("tickers", [])
        if tickers:
            return sorted(set(str(t).upper() for t in tickers))

    payload: Any = None
    for url in SP500_URLS:
        try:
            payload = request_json(session, url, {"apikey": FMP_API_KEY})
            if isinstance(payload, list):
                break
        except Exception:
            payload = None
    if not isinstance(payload, list):
        raise RuntimeError("Unable to fetch the S&P 500 constituent list from FMP.")
    tickers = sorted({str(item["symbol"]).upper() for item in payload if item.get("symbol")})
    SP500_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SP500_CACHE.write_text(
        json.dumps(
            {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "tickers": tickers},
            indent=2,
        )
    )
    return tickers


def fetch_intraday_chunk(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    payload: Any = None
    for url in STOCK_INTRADAY_URLS:
        try:
            endpoint = url.format(symbol=symbol)
            params = {"from": start_date.isoformat(), "to": end_date.isoformat(), "apikey": FMP_API_KEY}
            if "{symbol}" not in url:
                params["symbol"] = symbol
            payload = request_json(session, endpoint, params)
            if isinstance(payload, list):
                break
        except Exception:
            payload = None
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["symbol", "trade_date", "window_ts", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(payload)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{symbol} intraday response missing columns: {sorted(missing)}")

    frame = frame[list(required)].copy()
    local_ts = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(
        NY_TZ, nonexistent="shift_forward", ambiguous="NaT"
    )
    frame["window_ts"] = local_ts.dt.tz_convert("UTC")
    frame["trade_date"] = local_ts.dt.date
    frame["local_time"] = local_ts.dt.strftime("%H:%M")
    frame["symbol"] = symbol
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").round().astype("Int64")

    frame = frame[
        (frame["local_time"] >= REGULAR_OPEN)
        & (frame["local_time"] < REGULAR_CLOSE)
        & frame["window_ts"].notna()
    ].copy()
    frame = drop_out_of_range_ohlc_rows(symbol, frame)
    frame = frame.drop(columns=["date", "local_time"]).drop_duplicates(subset=["symbol", "window_ts"])
    return frame.sort_values("window_ts").reset_index(drop=True)


def normalize_intraday_frame(symbol: str, frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    required = {timestamp_col, "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{symbol} intraday response missing columns: {sorted(missing)}")

    normalized = frame[[timestamp_col, "open", "high", "low", "close", "volume"]].copy()
    local_ts = pd.to_datetime(normalized[timestamp_col], errors="coerce", utc=True).dt.tz_convert(NY_TZ)
    normalized["window_ts"] = local_ts.dt.tz_convert("UTC")
    normalized["trade_date"] = local_ts.dt.date
    normalized["local_time"] = local_ts.dt.strftime("%H:%M")
    normalized["symbol"] = symbol
    normalized["open"] = pd.to_numeric(normalized["open"], errors="coerce")
    normalized["high"] = pd.to_numeric(normalized["high"], errors="coerce")
    normalized["low"] = pd.to_numeric(normalized["low"], errors="coerce")
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce").round().astype("Int64")

    normalized = normalized[
        (normalized["local_time"] >= REGULAR_OPEN)
        & (normalized["local_time"] < REGULAR_CLOSE)
        & normalized["window_ts"].notna()
    ].copy()
    normalized = drop_out_of_range_ohlc_rows(symbol, normalized)
    normalized = normalized.drop(columns=[timestamp_col, "local_time"]).drop_duplicates(subset=["symbol", "window_ts"])
    return normalized.sort_values("window_ts").reset_index(drop=True)


def fetch_massive_intraday_chunk(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if not MASSIVE_API_KEY:
        raise RuntimeError("MASSIVE_API_KEY or POLYGON_API_KEY not found. Set one to enable Massive fallback.")
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": MASSIVE_API_KEY,
    }
    payload = request_json(session, build_massive_url(symbol, start_date, end_date), params)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Massive response type for {symbol}: {type(payload)}")
    if payload.get("status") not in {None, "OK"}:
        raise RuntimeError(f"Massive aggregates error for {symbol}: {payload}")
    results = payload.get("results") or []
    if not results:
        return pd.DataFrame(columns=["symbol", "trade_date", "window_ts", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.Series([row.get("t") for row in results], dtype="float64"), unit="ms", utc=True),
            "open": [row.get("o") for row in results],
            "high": [row.get("h") for row in results],
            "low": [row.get("l") for row in results],
            "close": [row.get("c") for row in results],
            "volume": [row.get("v") for row in results],
        }
    )
    return normalize_intraday_frame(symbol, frame, "timestamp")


def fetch_massive_intraday_range(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
    verbose: bool = False,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=MASSIVE_CHUNK_DAYS - 1), end_date)
        if verbose:
            print(f"Massive fallback {symbol}: {current} -> {chunk_end}")
        frame = fetch_massive_intraday_chunk(session, symbol, current, chunk_end)
        if not frame.empty:
            chunks.append(frame)
        time.sleep(REQUEST_SLEEP_SECONDS)
        current = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["symbol", "trade_date", "window_ts", "open", "high", "low", "close", "volume"])
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["symbol", "window_ts"]).sort_values("window_ts")


def fetch_intraday_range(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
    chunk_days: int,
    verbose: bool = False,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        if verbose:
            print(f"Fetching {symbol}: {current} -> {chunk_end}")
        frame = fetch_intraday_chunk(session, symbol, current, chunk_end)
        if not frame.empty:
            chunks.append(frame)
        time.sleep(REQUEST_SLEEP_SECONDS)
        current = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["symbol", "trade_date", "window_ts", "open", "high", "low", "close", "volume"])
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["symbol", "window_ts"]).sort_values("window_ts")


def write_report(name: str, payload: Any) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / name).write_text(json.dumps(payload, indent=2, default=str))


def append_filtered_rows_report(symbol: str, reason: str, rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with FILTERED_ROWS_REPORT.open("a", encoding="utf-8") as handle:
        for rec in rows.sort_values("window_ts").itertuples(index=False):
            handle.write(
                json.dumps(
                    {
                        "symbol": symbol,
                        "reason": reason,
                        "trade_date": rec.trade_date.isoformat() if rec.trade_date else None,
                        "window_ts": rec.window_ts.isoformat() if rec.window_ts is not None else None,
                        "open": None if pd.isna(rec.open) else float(rec.open),
                        "high": None if pd.isna(rec.high) else float(rec.high),
                        "low": None if pd.isna(rec.low) else float(rec.low),
                        "close": None if pd.isna(rec.close) else float(rec.close),
                        "volume": None if pd.isna(rec.volume) else int(rec.volume),
                    },
                    default=str,
                )
                + "\n"
            )


def drop_out_of_range_ohlc_rows(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    overflow_mask = frame[["open", "high", "low", "close"]].abs().gt(MAX_ABS_PRICE).any(axis=1)
    if not overflow_mask.any():
        return frame
    bad_rows = frame.loc[overflow_mask].copy()
    append_filtered_rows_report(symbol, "ohlc_out_of_range", bad_rows)
    print(f"  filtered_rows={len(bad_rows):,} reason=ohlc_out_of_range")
    return frame.loc[~overflow_mask].copy()


def serialize_missing_days(items: list[MissingDay]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": item.symbol,
            "trade_date": item.trade_date.isoformat(),
            "missing_count": item.missing_count,
            "expected_count": item.expected_count,
            "actual_count": item.actual_count,
            "missing_times_local": list(item.missing_times_local),
        }
        for item in items
    ]


def read_simple_fmp_checkpoint() -> dict[str, Any] | None:
    if not SIMPLE_FMP_CHECKPOINT.exists():
        return None
    return json.loads(SIMPLE_FMP_CHECKPOINT.read_text())


def write_simple_fmp_checkpoint(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIMPLE_FMP_CHECKPOINT.write_text(json.dumps(payload, indent=2, default=str))


def remove_simple_fmp_checkpoint() -> None:
    if SIMPLE_FMP_CHECKPOINT.exists():
        SIMPLE_FMP_CHECKPOINT.unlink()


def determine_effective_chunk_days(args: argparse.Namespace) -> int:
    if args.mode == "simple-fmp" and args.chunk_days == DEFAULT_CHUNK_DAYS:
        return DEFAULT_SIMPLE_CHUNK_DAYS
    return args.chunk_days


def resolve_requested_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.symbols:
        symbols.extend(item.strip().upper() for item in args.symbols.split(",") if item.strip())
    if args.symbols_file:
        for line in Path(args.symbols_file).read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                symbols.append(stripped.upper())
    return sorted(dict.fromkeys(symbols))


def apply_simple_fmp_resume(
    symbols: list[str],
    args: argparse.Namespace,
    start_date: date,
    end_date: date,
) -> tuple[list[str], str | None]:
    if args.start_from_symbol:
        start_symbol = args.start_from_symbol.upper()
        return [symbol for symbol in symbols if symbol >= start_symbol], start_symbol
    if args.mode != "simple-fmp" or not args.resume_from_checkpoint:
        return symbols, None

    checkpoint = read_simple_fmp_checkpoint()
    if not checkpoint:
        return symbols, None
    if checkpoint.get("mode") != "simple-fmp":
        return symbols, None
    if checkpoint.get("start_date") != start_date.isoformat() or checkpoint.get("end_date") != end_date.isoformat():
        return symbols, None

    last_symbol = checkpoint.get("last_completed_symbol")
    if not last_symbol:
        return symbols, None
    return [symbol for symbol in symbols if symbol > last_symbol], last_symbol


def persist_symbol_frame(conn, symbol: str, frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    upsert_rows = build_upsert_rows(frame)
    upsert_core_rows(conn, upsert_rows)
    target_dates = set(frame["trade_date"].tolist())
    symbol_frame = frame.copy()
    symbol_frame["status"] = "provisional"
    recomputed = recompute_features(symbol_frame)
    feature_updates = update_features(conn, symbol, recomputed, target_dates)
    return len(upsert_rows), feature_updates


def validate_core_frame(symbol: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    dupes = frame.duplicated(subset=["symbol", "window_ts"]).sum()
    if dupes:
        raise RuntimeError(f"{symbol} contains {dupes} duplicate (symbol, window_ts) rows after normalization.")
    null_rows = frame[["open", "high", "low", "close", "volume", "window_ts", "trade_date"]].isna().any(axis=1).sum()
    if null_rows:
        raise RuntimeError(f"{symbol} contains {null_rows} rows with null OHLCV/timestamp fields.")


def build_reference_schedule(spy_frame: pd.DataFrame) -> dict[date, tuple[pd.Timestamp, ...]]:
    validate_core_frame("SPY", spy_frame)
    if spy_frame.empty:
        raise RuntimeError("SPY reference fetch returned no regular-session bars.")

    schedule: dict[date, tuple[pd.Timestamp, ...]] = {}
    for trade_date, day in spy_frame.groupby("trade_date", observed=True):
        ordered = tuple(day.sort_values("window_ts")["window_ts"].tolist())
        if not ordered:
            continue
        if len(set(ordered)) != len(ordered):
            raise RuntimeError(f"SPY reference schedule has duplicate timestamps on {trade_date}.")
        if len(ordered) > REGULAR_BARS:
            raise RuntimeError(f"SPY reference schedule has {len(ordered)} bars on {trade_date}; expected <= {REGULAR_BARS}.")
        schedule[trade_date] = ordered
    if not schedule:
        raise RuntimeError("SPY reference fetch produced no usable trade dates.")
    return schedule


def find_missing_days(frame: pd.DataFrame, schedule: dict[date, tuple[pd.Timestamp, ...]], symbol: str) -> list[MissingDay]:
    validate_core_frame(symbol, frame)
    existing_by_date: dict[date, set[pd.Timestamp]] = defaultdict(set)
    for rec in frame[["trade_date", "window_ts"]].itertuples(index=False):
        existing_by_date[rec.trade_date].add(pd.Timestamp(rec.window_ts))

    missing: list[MissingDay] = []
    for trade_date, expected in schedule.items():
        actual = existing_by_date.get(trade_date, set())
        missing_times = tuple(
            pd.Timestamp(ts).tz_convert(NY_TZ).strftime("%H:%M")
            for ts in expected
            if pd.Timestamp(ts) not in actual
        )
        if missing_times:
            missing.append(
                MissingDay(
                    symbol=symbol,
                    trade_date=trade_date,
                    missing_count=len(missing_times),
                    expected_count=len(expected),
                    actual_count=len(actual),
                    missing_times_local=missing_times,
                )
            )
    return missing


def repair_symbol_gaps(
    session: requests.Session,
    symbol: str,
    base_frame: pd.DataFrame,
    missing_days: list[MissingDay],
    max_retries: int,
    verbose: bool = False,
) -> pd.DataFrame:
    merged = base_frame.copy()
    unresolved = missing_days
    for attempt in range(1, max_retries + 1):
        if not unresolved:
            break
        retry_dates = sorted({item.trade_date for item in unresolved})
        if verbose:
            print(f"Retry {attempt}/{max_retries} for {symbol}: {len(retry_dates)} trade dates")
        retry_chunks: list[pd.DataFrame] = []
        for trade_date in retry_dates:
            retry_frame = fetch_intraday_chunk(session, symbol, trade_date, trade_date)
            if not retry_frame.empty:
                retry_chunks.append(retry_frame)
            time.sleep(REQUEST_SLEEP_SECONDS)
        if retry_chunks:
            merged = (
                pd.concat([merged, *retry_chunks], ignore_index=True)
                .drop_duplicates(subset=["symbol", "window_ts"])
                .sort_values("window_ts")
                .reset_index(drop=True)
            )
        schedule = {item.trade_date: tuple() for item in missing_days}
        unresolved = find_missing_days(merged, schedule, symbol)
    return merged


def merge_frames(base_frame: pd.DataFrame, new_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not new_frames:
        return base_frame.copy()
    merged = pd.concat([base_frame, *new_frames], ignore_index=True)
    merged = merged.drop_duplicates(subset=["symbol", "window_ts"], keep="first")
    return merged.sort_values("window_ts").reset_index(drop=True)


def should_use_massive_fallback(symbol: str) -> bool:
    return symbol not in COMMODITY_SYMBOLS and bool(MASSIVE_API_KEY)


def apply_massive_fallback(
    session: requests.Session,
    symbol: str,
    base_frame: pd.DataFrame,
    unresolved: list[MissingDay],
    schedule: dict[date, tuple[pd.Timestamp, ...]],
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[MissingDay], list[FallbackFill]]:
    if not unresolved or not should_use_massive_fallback(symbol):
        return base_frame, unresolved, []

    retry_dates = sorted({item.trade_date for item in unresolved})
    if verbose:
        print(f"Massive fallback for {symbol}: {len(retry_dates)} trade dates")

    before_index = set()
    for rec in base_frame[["trade_date", "window_ts"]].itertuples(index=False):
        before_index.add((rec.trade_date, pd.Timestamp(rec.window_ts)))

    fallback_frame = fetch_massive_intraday_range(
        session=session,
        symbol=symbol,
        start_date=retry_dates[0],
        end_date=retry_dates[-1],
        verbose=verbose,
    )
    merged = merge_frames(base_frame, [fallback_frame] if not fallback_frame.empty else [])
    unresolved_after = find_missing_days(merged, schedule, symbol)

    added_by_day: dict[date, int] = defaultdict(int)
    for rec in merged[["trade_date", "window_ts"]].itertuples(index=False):
        key = (rec.trade_date, pd.Timestamp(rec.window_ts))
        if key not in before_index:
            added_by_day[rec.trade_date] += 1

    fills = [
        FallbackFill(
            symbol=symbol,
            provider="massive",
            provider_symbol=symbol,
            trade_date=trade_date,
            rows_added=rows_added,
        )
        for trade_date, rows_added in sorted(added_by_day.items())
        if rows_added > 0
    ]
    return merged, unresolved_after, fills


def repair_with_schedule(
    session: requests.Session,
    symbol: str,
    base_frame: pd.DataFrame,
    schedule: dict[date, tuple[pd.Timestamp, ...]],
    max_retries: int,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[MissingDay]]:
    merged = base_frame.copy()
    unresolved = find_missing_days(merged, schedule, symbol)
    for attempt in range(1, max_retries + 1):
        if not unresolved:
            break
        retry_dates = sorted({item.trade_date for item in unresolved})
        if verbose:
            print(f"Retry {attempt}/{max_retries} for {symbol}: {len(retry_dates)} trade dates")
        retry_chunks: list[pd.DataFrame] = []
        for trade_date in retry_dates:
            retry_frame = fetch_intraday_chunk(session, symbol, trade_date, trade_date)
            if not retry_frame.empty:
                retry_chunks.append(retry_frame)
            time.sleep(REQUEST_SLEEP_SECONDS)
        if retry_chunks:
            merged = (
                pd.concat([merged, *retry_chunks], ignore_index=True)
                .drop_duplicates(subset=["symbol", "window_ts"])
                .sort_values("window_ts")
                .reset_index(drop=True)
            )
        unresolved = find_missing_days(merged, schedule, symbol)
    return merged, unresolved


def build_upsert_rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    now_utc = datetime.now(timezone.utc)
    for rec in frame.sort_values(["trade_date", "window_ts"]).itertuples(index=False):
        rows.append(
            (
                rec.symbol,
                rec.trade_date,
                rec.window_ts.to_pydatetime() if hasattr(rec.window_ts, "to_pydatetime") else rec.window_ts,
                None if pd.isna(rec.open) else float(rec.open),
                None if pd.isna(rec.high) else float(rec.high),
                None if pd.isna(rec.low) else float(rec.low),
                None if pd.isna(rec.close) else float(rec.close),
                None if pd.isna(rec.volume) else int(rec.volume),
                0,
                "provisional",
                now_utc,
            )
        )
    return rows


def ensure_treasury_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TREASURY_TABLE} (
                series_code text NOT NULL,
                trade_date date NOT NULL,
                value numeric(18, 6),
                source text NOT NULL DEFAULT 'FMP',
                created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (series_code, trade_date)
            )
            """
        )


def fetch_treasury_rates(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    payload: Any = None
    for url in TREASURY_URLS:
        try:
            payload = request_json(
                session,
                url,
                {
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat(),
                    "apikey": FMP_API_KEY,
                },
            )
            if isinstance(payload, list):
                break
        except Exception:
            payload = None
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Treasury response type: {type(payload)}")
    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(columns=["series_code", "trade_date", "value", "source"])
    if "date" not in frame.columns:
        raise RuntimeError("Treasury response is missing the `date` column.")

    frame["trade_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    records: list[dict[str, Any]] = []
    for series_code, field_name in TREASURY_SERIES.items():
        if field_name not in frame.columns:
            raise RuntimeError(f"Treasury response is missing expected field `{field_name}` for {series_code}.")
        values = pd.to_numeric(frame[field_name], errors="coerce")
        for trade_date, value in zip(frame["trade_date"], values, strict=False):
            if trade_date is None or pd.isna(value):
                continue
            records.append(
                {
                    "series_code": series_code,
                    "trade_date": trade_date,
                    "value": float(value),
                    "source": "FMP",
                }
            )
    return pd.DataFrame(records)


def upsert_treasury_rates(conn, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    ensure_treasury_table(conn)
    rows = [tuple(rec) for rec in frame[["series_code", "trade_date", "value", "source"]].itertuples(index=False, name=None)]
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {TREASURY_TABLE} (series_code, trade_date, value, source)
            VALUES %s
            ON CONFLICT (series_code, trade_date) DO UPDATE SET
                value = EXCLUDED.value,
                source = EXCLUDED.source
            """,
            rows,
            page_size=500,
        )
    return len(rows)


def main() -> None:
    args = parse_args()
    ensure_api_key()
    start_date = iso_date(args.start_date)
    end_date = iso_date(args.end_date)
    if start_date > end_date:
        raise ValueError("--start-date must be <= --end-date.")
    allowed_unresolved = {symbol.upper() for symbol in args.allow_unresolved_symbols}
    chunk_days = determine_effective_chunk_days(args)

    session = build_session()
    all_sp500 = fetch_sp500_tickers(session)
    symbols = resolve_requested_symbols(args)
    explicit_symbols = bool(symbols)
    if not explicit_symbols:
        symbols = list(all_sp500)
        if args.limit_symbols is not None:
            symbols = symbols[: max(0, args.limit_symbols)]
        symbols.extend(COMMODITY_SYMBOLS.keys())
        symbols = sorted(set(symbols))
    symbols, resume_marker = apply_simple_fmp_resume(symbols, args, start_date, end_date)

    if explicit_symbols:
        print(f"Target symbols: {len(symbols)} (explicit selection)")
    else:
        print(f"Target symbols: {len(symbols)} ({len(all_sp500)} S&P 500 + {len(COMMODITY_SYMBOLS)} commodities)")
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Chunk days: {chunk_days}")
    if resume_marker:
        print(f"Resuming after: {resume_marker}")

    schedule: dict[date, tuple[pd.Timestamp, ...]] = {}
    if args.mode == "quality":
        print("Fetching SPY reference schedule...")
        spy_frame = fetch_intraday_range(session, "SPY", start_date, end_date, args.chunk_days, verbose=args.verbose)
        schedule = build_reference_schedule(spy_frame)
        print(f"Reference trade dates: {len(schedule)}")
    else:
        print("Simple FMP mode: skipping reference schedule, retries, and fallback.")

    fetched_frames: dict[str, pd.DataFrame] = {}
    unresolved_all: list[MissingDay] = []
    tolerated_unresolved: list[MissingDay] = []
    fallback_fills: list[FallbackFill] = []
    total_upserts = 0
    total_feature_updates = 0

    if args.mode == "simple-fmp":
        if args.dry_run:
            for idx, symbol in enumerate(symbols, start=1):
                print(f"[{idx}/{len(symbols)}] {symbol}")
                frame = fetch_intraday_range(session, symbol, start_date, end_date, chunk_days, verbose=args.verbose)
                validate_core_frame(symbol, frame)
                total_upserts += len(frame)
                print(f"  rows={len(frame):,} unresolved_days=0 fallback_days=0")
            if not args.skip_treasuries:
                treasury_frame = fetch_treasury_rates(session, start_date, end_date)
                write_report("dry_run_treasury_summary.json", {"rows": len(treasury_frame)})
            write_report("dry_run_intraday_summary.json", {"rows": total_upserts, "mode": args.mode, "chunk_days": chunk_days})
            print("Dry run completed successfully.")
            return

        conn = connect_db()
        conn.autocommit = False
        try:
            for idx, symbol in enumerate(symbols, start=1):
                print(f"[{idx}/{len(symbols)}] {symbol}")
                frame = fetch_intraday_range(session, symbol, start_date, end_date, chunk_days, verbose=args.verbose)
                validate_core_frame(symbol, frame)
                symbol_upserts, symbol_feature_updates = persist_symbol_frame(conn, symbol, frame)
                conn.commit()
                total_upserts += symbol_upserts
                total_feature_updates += symbol_feature_updates
                write_simple_fmp_checkpoint(
                    {
                        "mode": args.mode,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "chunk_days": chunk_days,
                        "last_completed_symbol": symbol,
                        "completed_count": idx,
                        "remaining_count": len(symbols) - idx,
                        "intraday_rows_upserted": total_upserts,
                        "feature_rows_updated": total_feature_updates,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f"  rows={len(frame):,} unresolved_days=0 fallback_days=0")

            treasury_rows = 0
            if not args.skip_treasuries:
                treasury_frame = fetch_treasury_rates(session, start_date, end_date)
                treasury_rows = upsert_treasury_rates(conn, treasury_frame)
                conn.commit()

            write_report(
                "last_success_summary.json",
                {
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "symbols_loaded": len(symbols),
                    "intraday_rows_upserted": total_upserts,
                    "feature_rows_updated": total_feature_updates,
                    "treasury_rows_upserted": treasury_rows,
                    "mode": args.mode,
                    "chunk_days": chunk_days,
                    "allowed_unresolved_symbols": sorted(allowed_unresolved),
                    "tolerated_unresolved_symbol_days": 0,
                    "treasury_series": TREASURY_SERIES,
                    "commodity_symbols": COMMODITY_SYMBOLS,
                },
            )
            remove_simple_fmp_checkpoint()
            print(f"Intraday rows upserted: {total_upserts:,}")
            print(f"Feature rows updated: {total_feature_updates:,}")
            print(f"Treasury rows upserted: {treasury_rows:,}")
            return
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] {symbol}")
        frame = fetch_intraday_range(session, symbol, start_date, end_date, chunk_days, verbose=args.verbose)
        unresolved: list[MissingDay] = []
        symbol_fills: list[FallbackFill] = []
        frame, unresolved = repair_with_schedule(
            session=session,
            symbol=symbol,
            base_frame=frame,
            schedule=schedule,
            max_retries=max(0, args.max_retries),
            verbose=args.verbose,
        )
        frame, unresolved, symbol_fills = apply_massive_fallback(
            session=session,
            symbol=symbol,
            base_frame=frame,
            unresolved=unresolved,
            schedule=schedule,
            verbose=args.verbose,
        )
        validate_core_frame(symbol, frame)
        fetched_frames[symbol] = frame
        if unresolved:
            if symbol in allowed_unresolved:
                tolerated_unresolved.extend(unresolved)
            else:
                unresolved_all.extend(unresolved)
        fallback_fills.extend(symbol_fills)
        print(f"  rows={len(frame):,} unresolved_days={len(unresolved)} fallback_days={len(symbol_fills)}")

    if args.mode == "quality" and unresolved_all:
        write_report("unresolved_intraday_gaps.json", serialize_missing_days(unresolved_all))
        raise RuntimeError(
            f"Unresolved intraday gaps remain for {len(unresolved_all)} symbol-days. "
            f"See {REPORT_DIR / 'unresolved_intraday_gaps.json'}."
        )

    if args.mode == "quality" and tolerated_unresolved:
        write_report("tolerated_intraday_gaps.json", serialize_missing_days(tolerated_unresolved))

    if args.mode == "quality" and fallback_fills:
        write_report(
            "massive_fallback_usage.json",
            [
                {
                    "symbol": item.symbol,
                    "provider": item.provider,
                    "provider_symbol": item.provider_symbol,
                    "trade_date": item.trade_date.isoformat(),
                    "rows_added": item.rows_added,
                }
                for item in fallback_fills
            ],
        )

    if args.dry_run:
        summary = {symbol: len(frame) for symbol, frame in fetched_frames.items()}
        write_report("dry_run_intraday_summary.json", summary)
        if not args.skip_treasuries:
            treasury_frame = fetch_treasury_rates(session, start_date, end_date)
            write_report("dry_run_treasury_summary.json", {"rows": len(treasury_frame)})
        print("Dry run completed successfully.")
        return

    conn = connect_db()
    conn.autocommit = False
    try:
        for symbol, frame in fetched_frames.items():
            if frame.empty:
                continue
            symbol_upserts, symbol_feature_updates = persist_symbol_frame(conn, symbol, frame)
            total_upserts += symbol_upserts
            total_feature_updates += symbol_feature_updates

        treasury_rows = 0
        if not args.skip_treasuries:
            treasury_frame = fetch_treasury_rates(session, start_date, end_date)
            treasury_rows = upsert_treasury_rates(conn, treasury_frame)

        conn.commit()
        write_report(
            "last_success_summary.json",
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbols_loaded": len(symbols),
                "intraday_rows_upserted": total_upserts,
                "feature_rows_updated": total_feature_updates,
                "treasury_rows_upserted": treasury_rows,
                "mode": args.mode,
                "chunk_days": chunk_days,
                "allowed_unresolved_symbols": sorted(allowed_unresolved),
                "tolerated_unresolved_symbol_days": len(tolerated_unresolved),
                "treasury_series": TREASURY_SERIES,
                "commodity_symbols": COMMODITY_SYMBOLS,
            },
        )
        print(f"Intraday rows upserted: {total_upserts:,}")
        print(f"Feature rows updated: {total_feature_updates:,}")
        print(f"Treasury rows upserted: {treasury_rows:,}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
