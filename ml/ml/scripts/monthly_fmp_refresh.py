"""Monthly FMP refresh for ml.market_data_15m.

This job replaces continuous intraday collection with a historical monthly
refresh. It fetches regular-session 15-minute OHLCV bars for the target date
range, validates coverage against an SPY-derived reference schedule, upserts
bars into ml.market_data_15m, refreshes market.daily_prices for overnight-gap
features, and writes machine-readable reports.
"""

from __future__ import annotations

import argparse
import calendar
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from backfill_market_data_15m_from_fmp import connect_db
from refetch_market_data_15m_quality import (
    FMP_API_KEY,
    build_reference_schedule,
    build_session,
    ensure_api_key,
    fetch_intraday_range,
    find_missing_days,
    persist_symbol_frame,
    repair_with_schedule,
    request_json,
    serialize_missing_days,
    validate_core_frame,
)


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml"
load_dotenv(ROOT / ".env")
load_dotenv(ML_ROOT / ".env")

REPORT_DIR = ML_ROOT / "data" / "monthly_refresh_reports"
CHECKPOINT_FILE = REPORT_DIR / "resume_checkpoint.json"
SUMMARY_FILE = REPORT_DIR / "last_success_summary.json"
DRY_RUN_FILE = REPORT_DIR / "dry_run_summary.json"
UNRESOLVED_FILE = REPORT_DIR / "unresolved_intraday_gaps.json"
NULLS_FILE = REPORT_DIR / "null_ohlcv_rows.json"
DAILY_PRICE_FAILURES_FILE = REPORT_DIR / "daily_price_failures.json"

DEFAULT_CHUNK_DAYS = 5
DEFAULT_MAX_RETRIES = 2
DEFAULT_HISTORY_YEARS = 5
DAILY_PRICE_LOOKBACK_DAYS = 7
DAILY_PRICE_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"


@dataclass(frozen=True)
class FrameIssue:
    symbol: str
    trade_date: str | None
    window_ts: str | None
    issue: str


def previous_calendar_month(today: date | None = None) -> tuple[date, date]:
    today = today or datetime.now(timezone.utc).date()
    first_of_month = today.replace(day=1)
    prev_end = first_of_month - timedelta(days=1)
    prev_start = prev_end.replace(day=1)
    return prev_start, prev_end


def previous_calendar_month_label(today: date | None = None) -> str:
    start, _ = previous_calendar_month(today=today)
    return start.strftime("%Y-%m")


def parse_month(value: str) -> tuple[date, date]:
    try:
        parsed = datetime.strptime(value, "%Y-%m").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("month must be in YYYY-MM format") from exc
    month_end = date(parsed.year, parsed.month, calendar.monthrange(parsed.year, parsed.month)[1])
    return parsed.replace(day=1), month_end


def subtract_years(value: date, years: int) -> date:
    target_year = value.year - years
    target_day = min(value.day, calendar.monthrange(target_year, value.month)[1])
    return value.replace(year=target_year, day=target_day)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly FMP refresh for ml.market_data_15m.")
    parser.add_argument("--mode", choices=("monthly-refresh", "bootstrap"), default="monthly-refresh")
    parser.add_argument("--month", default=None, help="Override month in YYYY-MM format.")
    parser.add_argument("--start-date", default=None, help="Explicit inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Explicit inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--history-years", type=int, default=DEFAULT_HISTORY_YEARS)
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--symbols", default=None, help="Comma-separated explicit symbol list.")
    parser.add_argument("--limit-symbols", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-daily-prices", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def write_report(path: Path, payload: Any) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def remove_stale_reports() -> None:
    for path in (SUMMARY_FILE, DRY_RUN_FILE, UNRESOLVED_FILE, NULLS_FILE, DAILY_PRICE_FAILURES_FILE):
        if path.exists():
            path.unlink()


def resolve_date_range(args: argparse.Namespace) -> tuple[date, date]:
    if args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            raise ValueError("--start-date and --end-date must be provided together.")
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        if start > end:
            raise ValueError("--start-date must be <= --end-date.")
        return start, end
    if args.month:
        return parse_month(args.month)
    if args.mode == "bootstrap":
        _, prev_end = previous_calendar_month()
        return subtract_years(prev_end, args.history_years), prev_end
    return previous_calendar_month()


def get_active_symbols(conn, explicit_symbols: list[str] | None, limit_symbols: int | None) -> list[str]:
    if explicit_symbols:
        symbols = sorted(dict.fromkeys(symbol.upper() for symbol in explicit_symbols))
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol FROM market.stocks WHERE is_active = true ORDER BY symbol")
            symbols = [row[0].upper() for row in cur.fetchall()]
    if limit_symbols is not None:
        symbols = symbols[: max(0, limit_symbols)]
    if not symbols:
        raise RuntimeError("No symbols available. Populate market.stocks or pass --symbols.")
    return symbols


def read_checkpoint() -> dict[str, Any] | None:
    if not CHECKPOINT_FILE.exists():
        return None
    return json.loads(CHECKPOINT_FILE.read_text())


def write_checkpoint(payload: dict[str, Any]) -> None:
    write_report(CHECKPOINT_FILE, payload)


def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def apply_checkpoint(symbols: list[str], args: argparse.Namespace, start_date: date, end_date: date) -> tuple[list[str], str | None]:
    if not args.resume_from_checkpoint:
        return symbols, None
    checkpoint = read_checkpoint()
    if not checkpoint:
        return symbols, None
    if checkpoint.get("mode") != args.mode:
        return symbols, None
    if checkpoint.get("start_date") != start_date.isoformat() or checkpoint.get("end_date") != end_date.isoformat():
        return symbols, None
    last_symbol = checkpoint.get("last_completed_symbol")
    if not last_symbol:
        return symbols, None
    return [symbol for symbol in symbols if symbol > last_symbol], last_symbol


def inspect_frame_issues(symbol: str, frame: pd.DataFrame) -> list[FrameIssue]:
    if frame.empty:
        return []
    issues: list[FrameIssue] = []
    null_mask = frame[["open", "high", "low", "close", "volume", "window_ts", "trade_date"]].isna().any(axis=1)
    for rec in frame.loc[null_mask, ["trade_date", "window_ts"]].itertuples(index=False):
        issues.append(
            FrameIssue(
                symbol=symbol,
                trade_date=None if pd.isna(rec.trade_date) else str(rec.trade_date),
                window_ts=None if pd.isna(rec.window_ts) else pd.Timestamp(rec.window_ts).isoformat(),
                issue="null_ohlcv_or_timestamp",
            )
        )
    return issues


def fetch_daily_price_rows(session, symbol: str, start_date: date, end_date: date) -> list[tuple[Any, ...]]:
    payload = request_json(
        session,
        DAILY_PRICE_URL.format(symbol=symbol),
        {"from": start_date.isoformat(), "to": end_date.isoformat(), "apikey": FMP_API_KEY},
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected daily price response type for {symbol}: {type(payload)}")
    historical = payload.get("historical") or []
    rows: list[tuple[Any, ...]] = []
    for item in historical:
        try:
            close = float(item["close"])
            change = float(item.get("change") or 0.0)
            rows.append(
                (
                    symbol,
                    item["date"],
                    float(item["open"]),
                    float(item["high"]),
                    float(item["low"]),
                    close,
                    int(item.get("volume") or 0),
                    round(close - change, 6),
                    round(change, 6),
                    round(float(item.get("changePercent") or 0.0), 6),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def refresh_daily_prices(session, conn, symbols: list[str], start_date: date, end_date: date, verbose: bool = False) -> tuple[int, list[str]]:
    rows_written = 0
    failed_symbols: list[str] = []
    daily_start = start_date - timedelta(days=DAILY_PRICE_LOOKBACK_DAYS)
    with conn.cursor() as cur:
        for idx, symbol in enumerate(symbols, start=1):
            if verbose:
                print(f"[daily {idx}/{len(symbols)}] {symbol}: {daily_start} -> {end_date}")
            try:
                rows = fetch_daily_price_rows(session, symbol, daily_start, end_date)
            except Exception:
                failed_symbols.append(symbol)
                continue
            if not rows:
                failed_symbols.append(symbol)
                continue
            execute_values(
                cur,
                """
                INSERT INTO market.daily_prices
                    (symbol, date, open, high, low, close, volume, previous_close, change, change_pct)
                VALUES %s
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    previous_close = EXCLUDED.previous_close,
                    change = EXCLUDED.change,
                    change_pct = EXCLUDED.change_pct
                """,
                rows,
                page_size=500,
            )
            rows_written += len(rows)
    return rows_written, failed_symbols


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_api_key()
    start_date, end_date = resolve_date_range(args)
    remove_stale_reports()

    explicit_symbols = [item.strip().upper() for item in (args.symbols or "").split(",") if item.strip()] or None
    session = build_session()
    symbol_conn = connect_db()
    try:
        symbols = get_active_symbols(symbol_conn, explicit_symbols, args.limit_symbols)
    finally:
        symbol_conn.close()
    symbols, resume_marker = apply_checkpoint(symbols, args, start_date, end_date)
    if not symbols:
        raise RuntimeError("No symbols remain after applying the checkpoint.")

    print(f"Mode: {args.mode}")
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Target symbols: {len(symbols)}")
    print(f"Chunk days: {args.chunk_days}")
    if resume_marker:
        print(f"Resuming after: {resume_marker}")

    print("Fetching SPY reference schedule...")
    spy_frame = fetch_intraday_range(session, "SPY", start_date, end_date, args.chunk_days, verbose=args.verbose)
    schedule = build_reference_schedule(spy_frame)
    print(f"Reference trade dates: {len(schedule)}")

    fetched_frames: dict[str, pd.DataFrame] = {}
    unresolved_all = []
    null_issues: list[FrameIssue] = []

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] {symbol}")
        frame = fetch_intraday_range(session, symbol, start_date, end_date, args.chunk_days, verbose=args.verbose)
        frame, unresolved = repair_with_schedule(
            session=session,
            symbol=symbol,
            base_frame=frame,
            schedule=schedule,
            max_retries=max(0, args.max_retries),
            verbose=args.verbose,
        )
        issues = inspect_frame_issues(symbol, frame)
        null_issues.extend(issues)
        validate_core_frame(symbol, frame)
        unresolved = find_missing_days(frame, schedule, symbol)
        fetched_frames[symbol] = frame
        unresolved_all.extend(unresolved)
        print(f"  rows={len(frame):,} unresolved_days={len(unresolved)} null_rows={len(issues)}")

    if unresolved_all:
        write_report(UNRESOLVED_FILE, serialize_missing_days(unresolved_all))
    if null_issues:
        write_report(NULLS_FILE, [issue.__dict__ for issue in null_issues])
    if unresolved_all or null_issues:
        raise RuntimeError(
            "Validation failed for monthly refresh. "
            f"unresolved_days={len(unresolved_all)} null_rows={len(null_issues)}"
        )

    if args.dry_run:
        write_report(
            DRY_RUN_FILE,
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": args.mode,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbols_loaded": len(symbols),
                "intraday_rows_detected": sum(len(frame) for frame in fetched_frames.values()),
                "chunk_days": args.chunk_days,
                "report_dir": str(REPORT_DIR),
            },
        )
        print("Dry run completed successfully.")
        return

    conn = connect_db()
    conn.autocommit = False
    try:
        daily_price_rows = 0
        if not args.skip_daily_prices:
            daily_price_rows, failed_daily_symbols = refresh_daily_prices(
                session=session,
                conn=conn,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                verbose=args.verbose,
            )
            if failed_daily_symbols:
                write_report(DAILY_PRICE_FAILURES_FILE, failed_daily_symbols)
                raise RuntimeError(f"Daily price refresh failed for {len(failed_daily_symbols)} symbols.")

        total_upserts = 0
        total_feature_updates = 0
        for idx, symbol in enumerate(symbols, start=1):
            frame = fetched_frames[symbol]
            symbol_upserts, symbol_feature_updates = persist_symbol_frame(conn, symbol, frame)
            conn.commit()
            total_upserts += symbol_upserts
            total_feature_updates += symbol_feature_updates
            write_checkpoint(
                {
                    "mode": args.mode,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "last_completed_symbol": symbol,
                    "completed_count": idx,
                    "remaining_count": len(symbols) - idx,
                    "intraday_rows_upserted": total_upserts,
                    "feature_rows_updated": total_feature_updates,
                    "daily_price_rows_upserted": daily_price_rows,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

        write_report(
            SUMMARY_FILE,
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": args.mode,
                "month": args.month or previous_calendar_month_label(today=end_date + timedelta(days=1)),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbols_loaded": len(symbols),
                "intraday_rows_upserted": total_upserts,
                "feature_rows_updated": total_feature_updates,
                "daily_price_rows_upserted": daily_price_rows,
                "chunk_days": args.chunk_days,
                "max_retries": args.max_retries,
                "report_dir": str(REPORT_DIR),
            },
        )
        clear_checkpoint()
        print(f"Intraday rows upserted: {total_upserts:,}")
        print(f"Feature rows updated: {total_feature_updates:,}")
        print(f"Daily price rows upserted: {daily_price_rows:,}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
