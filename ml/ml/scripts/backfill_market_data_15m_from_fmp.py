"""
Targeted FMP backfill for ml.market_data_15m.

This script:
1. Detects missing regular-session 15-minute bars and rows with missing core OHLCV.
2. Fetches the affected bars from FMP.
3. Upserts the core row data into ml.market_data_15m on (symbol, window_ts).
4. Recomputes row-level engineered columns for the affected symbol/date window.

It is intentionally narrow. It does not rebuild the whole table.

Examples:
    python ml/scripts/backfill_market_data_15m_from_fmp.py --limit-symbol-days 25 --dry-run
    python ml/scripts/backfill_market_data_15m_from_fmp.py --symbol TER --start-date 2026-01-27 --end-date 2026-01-27
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import execute_values


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"))
load_dotenv(os.path.join(ROOT, "ml", ".env"))

DB_HOST = os.getenv("POSTGRES_SERVER", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "app")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "changethis")
FMP_API_KEY = os.getenv("FMP_API_KEY")

FMP_BASE = "https://financialmodelingprep.com/api/v3/historical-chart/15min"
NY_TZ = "America/New_York"
REGULAR_BAR_STARTS = pd.date_range("09:30", "15:45", freq="15min").time
FETCH_SLEEP_SECONDS = 0.35
DEFAULT_LOOKBACK_DAYS = 45

CORE_COLUMNS = ["open", "high", "low", "close", "volume"]
UPSERT_COLUMNS = [
    "symbol",
    "trade_date",
    "window_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "slot_count",
    "status",
    "created_at",
]
FEATURE_UPDATE_COLUMNS = [
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

FEATURE_UPDATE_COLUMN_TYPES = [
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "numeric",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "smallint",
    "numeric",
    "numeric",
    "numeric",
    "smallint",
    "smallint",
    "smallint",
]


@dataclass(frozen=True)
class Candidate:
    symbol: str
    trade_date: date
    expected_ts_utc: datetime
    reason: str


@dataclass(frozen=True)
class FetchWindow:
    start_date: date
    end_date: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing 15-minute bars from FMP into ml.market_data_15m.")
    parser.add_argument("--symbol", action="append", dest="symbols", default=None, help="Symbol to target. Repeatable.")
    parser.add_argument("--start-date", default=None, help="Inclusive trade_date lower bound (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Inclusive trade_date upper bound (YYYY-MM-DD).")
    parser.add_argument("--limit-symbol-days", type=int, default=20, help="Max incomplete symbol-days to scan from DB.")
    parser.add_argument("--chunk-days", type=int, default=10, help="FMP request size in days.")
    parser.add_argument("--dry-run", action="store_true", help="Detect and fetch without writing to Postgres.")
    parser.add_argument("--fix-core-nulls-only", action="store_true", help="Skip missing-bar discovery and repair only existing rows with null OHLCV.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        connect_timeout=10,
    )


def build_filters(args: argparse.Namespace) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if args.symbols:
        clauses.append("symbol = ANY(%s)")
        params.append(args.symbols)
    if args.start_date:
        clauses.append("trade_date >= %s")
        params.append(args.start_date)
    if args.end_date:
        clauses.append("trade_date <= %s")
        params.append(args.end_date)
    return (" AND ".join(clauses), params)


def fetch_core_null_candidates(conn, args: argparse.Namespace) -> list[Candidate]:
    extra_filter, params = build_filters(args)
    where = "WHERE (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL)"
    if extra_filter:
        where += f" AND {extra_filter}"
    sql = f"""
        SELECT symbol, trade_date, window_ts
        FROM ml.market_data_15m
        {where}
        ORDER BY trade_date, symbol, window_ts
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [Candidate(symbol=r[0], trade_date=r[1], expected_ts_utc=r[2], reason="core_null") for r in rows]


def fetch_missing_regular_bar_candidates(conn, args: argparse.Namespace) -> list[Candidate]:
    extra_filter, params = build_filters(args)
    where = "WHERE TRUE"
    if extra_filter:
        where += f" AND {extra_filter}"
    sql = f"""
        WITH day_counts AS (
            SELECT symbol, trade_date,
                   COUNT(*) FILTER (
                       WHERE (window_ts AT TIME ZONE 'America/New_York')::time >= TIME '09:30'
                         AND (window_ts AT TIME ZONE 'America/New_York')::time < TIME '16:00'
                   ) AS bars_930_1600
            FROM ml.market_data_15m
            {where}
            GROUP BY 1,2
        )
        SELECT symbol, trade_date
        FROM day_counts
        WHERE bars_930_1600 < 26
        ORDER BY trade_date, symbol
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, [*params, args.limit_symbol_days])
        symbol_days = cur.fetchall()

    candidates: list[Candidate] = []
    for symbol, trade_date in symbol_days:
        expected_times = [
            pd.Timestamp.combine(pd.Timestamp(trade_date).date(), bar_time)
            .tz_localize(NY_TZ)
            .tz_convert("UTC")
            .to_pydatetime()
            for bar_time in REGULAR_BAR_STARTS
        ]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT window_ts
                FROM ml.market_data_15m
                WHERE symbol = %s
                  AND trade_date = %s
                  AND (window_ts AT TIME ZONE 'America/New_York')::time >= TIME '09:30'
                  AND (window_ts AT TIME ZONE 'America/New_York')::time < TIME '16:00'
                """,
                (symbol, trade_date),
            )
            existing = {row[0] for row in cur.fetchall()}
        for ts_utc in expected_times:
            if ts_utc not in existing:
                candidates.append(
                    Candidate(
                        symbol=symbol,
                        trade_date=trade_date,
                        expected_ts_utc=ts_utc,
                        reason="missing_regular_bar",
                    )
                )
    return candidates


def fetch_fmp_symbol_range(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not found in environment.")
    chunks: list[pd.DataFrame] = []
    cur = start_date
    chunk_days = max(1, getattr(fetch_fmp_symbol_range, "chunk_days", 10))
    while cur <= end_date:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_date)
        url = (
            f"{FMP_BASE}/{symbol}"
            f"?from={cur.isoformat()}&to={chunk_end.isoformat()}"
            f"&extended=true&apikey={FMP_API_KEY}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list) and payload:
            frame = pd.DataFrame(payload)[["date", "open", "high", "low", "close", "volume"]]
            frame["symbol"] = symbol
            chunks.append(frame)
        time.sleep(FETCH_SLEEP_SECONDS)
        cur = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["symbol", "window_ts", "trade_date", "open", "high", "low", "close", "volume"])

    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["date"]).copy()
    ts_local = pd.to_datetime(df["date"]).dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
    df["window_ts"] = ts_local.dt.tz_convert("UTC")
    df["trade_date"] = ts_local.dt.date
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    volume_num = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = volume_num.round().astype("Int64")
    return df[["symbol", "window_ts", "trade_date", "open", "high", "low", "close", "volume"]]


def collapse_dates_to_windows(dates: list[date], max_span_days: int) -> list[FetchWindow]:
    if not dates:
        return []
    sorted_dates = sorted(set(dates))
    windows: list[FetchWindow] = []
    start = sorted_dates[0]
    end = sorted_dates[0]
    for current in sorted_dates[1:]:
        next_span = (current - start).days + 1
        if (current - end).days <= 1 and next_span <= max_span_days:
            end = current
            continue
        windows.append(FetchWindow(start_date=start, end_date=end))
        start = current
        end = current
    windows.append(FetchWindow(start_date=start, end_date=end))
    return windows


def fetch_fmp_symbol_dates(symbol: str, dates: list[date], chunk_days: int, verbose: bool = False) -> pd.DataFrame:
    windows = collapse_dates_to_windows(dates, max_span_days=max(1, chunk_days))
    chunks: list[pd.DataFrame] = []
    for window in windows:
        if verbose:
            print(f"Fetching {symbol}: {window.start_date} -> {window.end_date}")
        frame = fetch_fmp_symbol_range(symbol, window.start_date, window.end_date)
        if not frame.empty:
            chunks.append(frame)
    if not chunks:
        return pd.DataFrame(columns=["symbol", "window_ts", "trade_date", "open", "high", "low", "close", "volume"])
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["symbol", "window_ts"])


def prepare_upsert_rows(candidates: list[Candidate], fetched_by_symbol: dict[str, pd.DataFrame]) -> tuple[list[tuple], dict[str, set[date]], list[Candidate]]:
    rows: list[tuple] = []
    affected_dates: dict[str, set[date]] = defaultdict(set)
    unresolved: list[Candidate] = []
    now_utc = datetime.now(timezone.utc)
    for candidate in candidates:
        fdf = fetched_by_symbol.get(candidate.symbol)
        if fdf is None or fdf.empty:
            unresolved.append(candidate)
            continue
        matched = fdf[fdf["window_ts"] == pd.Timestamp(candidate.expected_ts_utc)]
        if matched.empty:
            unresolved.append(candidate)
            continue
        rec = matched.iloc[0]
        rows.append(
            (
                candidate.symbol,
                candidate.trade_date,
                candidate.expected_ts_utc,
                none_if_nan(rec["open"]),
                none_if_nan(rec["high"]),
                none_if_nan(rec["low"]),
                none_if_nan(rec["close"]),
                none_if_nan(rec["volume"]),
                0,
                "provisional",
                now_utc,
            )
        )
        affected_dates[candidate.symbol].add(candidate.trade_date)
        affected_dates[candidate.symbol].add(candidate.trade_date + timedelta(days=1))
    return rows, affected_dates, unresolved


def none_if_nan(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def upsert_core_rows(conn, rows: list[tuple]) -> None:
    if not rows:
        return
    sql = f"""
        INSERT INTO ml.market_data_15m ({", ".join(UPSERT_COLUMNS)})
        VALUES %s
        ON CONFLICT (symbol, window_ts) DO UPDATE SET
            trade_date = EXCLUDED.trade_date,
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            status = EXCLUDED.status
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=500)


def load_symbol_window(conn, symbol: str, target_dates: set[date]) -> pd.DataFrame:
    start = min(target_dates) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    end = max(target_dates) + timedelta(days=2)
    sql = """
        SELECT symbol, trade_date, window_ts, open, high, low, close, volume, status
        FROM ml.market_data_15m
        WHERE symbol = %s
          AND trade_date BETWEEN %s AND %s
        ORDER BY window_ts
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, start, end))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    frame = pd.DataFrame(rows, columns=cols)
    if frame.empty:
        return frame
    frame["window_ts"] = pd.to_datetime(frame["window_ts"], utc=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    return frame


def recompute_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("window_ts").copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    out["lag_close_1"] = out["close"].shift(1)
    out["lag_close_5"] = out["close"].shift(5)
    out["lag_close_10"] = out["close"].shift(10)
    out["close_diff_1"] = out["close"] - out["lag_close_1"]
    out["close_diff_5"] = out["close"] - out["lag_close_5"]
    out["pct_change_1"] = out["close"] / out["lag_close_1"] - 1.0
    out["pct_change_5"] = out["close"] / out["lag_close_5"] - 1.0
    out["log_return_1"] = safe_log_ratio(out["close"], out["lag_close_1"])
    out["sma_close_5"] = out["close"].rolling(5, min_periods=5).mean()
    out["sma_close_10"] = out["close"].rolling(10, min_periods=10).mean()
    out["sma_close_20"] = out["close"].rolling(20, min_periods=20).mean()
    out["sma_volume_5"] = out["volume"].rolling(5, min_periods=5).mean()

    session_close = out.groupby("trade_date", observed=True)["close"].last().shift(1)
    out["previous_close"] = out["trade_date"].map(session_close)
    out["overnight_gap_pct"] = out["open"] / out["previous_close"] - 1.0
    out["overnight_log_return"] = safe_log_ratio(out["open"], out["previous_close"])
    out["is_gap_up"] = (out["overnight_gap_pct"] > 0).astype("Int64")
    out["is_gap_down"] = (out["overnight_gap_pct"] < 0).astype("Int64")

    local_ts = out["window_ts"].dt.tz_convert(NY_TZ)
    out["day_of_week"] = local_ts.dt.dayofweek.astype("Int64")
    out["hour_of_day"] = local_ts.dt.hour.astype("Int64")
    out["month_of_year"] = local_ts.dt.month.astype("Int64")
    out["day_monday"] = (out["day_of_week"] == 0).astype("Int64")
    out["day_tuesday"] = (out["day_of_week"] == 1).astype("Int64")
    out["day_wednesday"] = (out["day_of_week"] == 2).astype("Int64")
    out["day_thursday"] = (out["day_of_week"] == 3).astype("Int64")
    out["day_friday"] = (out["day_of_week"] == 4).astype("Int64")
    quarter = local_ts.dt.quarter.astype("Int64")
    out["quarter_1"] = (quarter == 1).astype("Int64")
    out["quarter_2"] = (quarter == 2).astype("Int64")
    out["quarter_3"] = (quarter == 3).astype("Int64")
    out["quarter_4"] = (quarter == 4).astype("Int64")

    hour = out["hour_of_day"]
    out["hour_early_morning"] = hour.isin([4, 5, 6, 7, 8, 9]).astype("Int64")
    out["hour_mid_morning"] = hour.isin([10, 11]).astype("Int64")
    out["hour_afternoon"] = hour.isin([12, 13, 14]).astype("Int64")
    out["hour_late_afternoon"] = hour.isin([15, 16, 17, 18, 19]).astype("Int64")
    return out


def safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    valid = (a_num > 0) & (b_num > 0)
    out = pd.Series(index=a_num.index, dtype="float64")
    out.loc[valid] = np_log(a_num.loc[valid] / b_num.loc[valid])
    return out


def np_log(values: pd.Series) -> pd.Series:
    return pd.Series(np.log(values), index=values.index)


def update_features(conn, symbol: str, recomputed: pd.DataFrame, target_dates: set[date]) -> int:
    target = recomputed[recomputed["trade_date"].isin(target_dates)].copy()
    if target.empty:
        return 0
    raw_columns = ["symbol", "window_ts", *FEATURE_UPDATE_COLUMNS]
    casted_select = ", ".join(
        [
            "raw.symbol::text AS symbol",
            "raw.window_ts::timestamptz AS window_ts",
            *[
                f"raw.{col}::{col_type} AS {col}"
                for col, col_type in zip(FEATURE_UPDATE_COLUMNS, FEATURE_UPDATE_COLUMN_TYPES, strict=True)
            ],
        ]
    )
    sql = f"""
        UPDATE ml.market_data_15m AS t
        SET {", ".join(f"{col} = data.{col}" for col in FEATURE_UPDATE_COLUMNS)}
        FROM (
            SELECT {casted_select}
            FROM (VALUES %s) AS raw ({", ".join(raw_columns)})
        ) AS data
        WHERE t.symbol = data.symbol
          AND t.window_ts = data.window_ts
    """
    values = []
    for rec in target[["symbol", "window_ts", *FEATURE_UPDATE_COLUMNS]].itertuples(index=False, name=None):
        values.append(tuple(none_if_nan(v) for v in rec))
    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=500)
    return len(values)


def main() -> None:
    args = parse_args()
    fetch_fmp_symbol_range.chunk_days = args.chunk_days
    conn = connect_db()
    conn.autocommit = False
    try:
        core_nulls = fetch_core_null_candidates(conn, args)
        missing_bars = [] if args.fix_core_nulls_only else fetch_missing_regular_bar_candidates(conn, args)
        candidates = {(c.symbol, c.trade_date, c.expected_ts_utc, c.reason): c for c in [*core_nulls, *missing_bars]}
        candidate_list = sorted(candidates.values(), key=lambda c: (c.trade_date, c.symbol, c.expected_ts_utc, c.reason))
        print(f"Candidates: {len(candidate_list)}")
        if not candidate_list:
            conn.rollback()
            return

        by_symbol_dates: dict[str, list[date]] = defaultdict(list)
        for candidate in candidate_list:
            by_symbol_dates[candidate.symbol].append(candidate.trade_date)

        fetched_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol, dates in by_symbol_dates.items():
            fetched_by_symbol[symbol] = fetch_fmp_symbol_dates(
                symbol,
                dates,
                chunk_days=args.chunk_days,
                verbose=args.verbose,
            )

        upsert_rows, affected_dates, unresolved = prepare_upsert_rows(candidate_list, fetched_by_symbol)
        print(f"Resolvable candidates: {len(upsert_rows)}")
        print(f"Unresolved candidates: {len(unresolved)}")
        if unresolved and args.verbose:
            for item in unresolved[:20]:
                print(f"  unresolved {item.symbol} {item.trade_date} {item.expected_ts_utc} {item.reason}")

        if args.dry_run:
            conn.rollback()
            return

        upsert_core_rows(conn, upsert_rows)
        feature_updates = 0
        for symbol, target_dates in affected_dates.items():
            symbol_frame = load_symbol_window(conn, symbol, target_dates)
            if symbol_frame.empty:
                continue
            recomputed = recompute_features(symbol_frame)
            feature_updates += update_features(conn, symbol, recomputed, target_dates)

        conn.commit()
        print(f"Upserted rows: {len(upsert_rows)}")
        print(f"Feature-updated rows: {feature_updates}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
