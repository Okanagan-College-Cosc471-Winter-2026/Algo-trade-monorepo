"""
Fetch raw vendor market data into a clean landing area for later processing.

Outputs under `ml/data/raw_market_vendor/`:
- `reference/sp500_constituents.json`
- `intraday/symbol=<SYMBOL>/bars.parquet`
- `macro/treasury_rates_daily.parquet`
- `reports/fetch_summary.json`
- `reports/intraday_coverage.csv`

What this script fetches:
- Current S&P 500 constituents from FMP
- 15-minute intraday bars for all constituents
- 15-minute intraday bars for:
  - GCUSD (gold)
  - CLUSD (crude oil)
- Daily Treasury rates for:
  - 13-week
  - 5-year
  - 10-year

This script is a raw landing job, not a final quality gate:
- It stores vendor output with normalized timestamps and light validation
- It does not refuse partial symbol-days
- It produces coverage reports so later processing can decide how to handle gaps
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml"
load_dotenv(ROOT / ".env")
load_dotenv(ML_ROOT / ".env")

FMP_API_KEY = os.getenv("FMP_API_KEY")
NY_TZ = "America/New_York"
REQUEST_TIMEOUT = 60
REQUEST_SLEEP_SECONDS = 0.35
RETRY_BACKOFF_SECONDS = (2, 5, 15)

SP500_URLS = (
    "https://financialmodelingprep.com/api/v3/sp500_constituent",
    "https://financialmodelingprep.com/stable/sp500-constituent",
)
INTRADAY_15M_URLS = (
    "https://financialmodelingprep.com/stable/historical-chart/15min",
    "https://financialmodelingprep.com/api/v3/historical-chart/15min/{symbol}",
)
TREASURY_URLS = (
    "https://financialmodelingprep.com/stable/treasury-rates",
    "https://financialmodelingprep.com/api/v4/treasury",
)

COMMODITY_SYMBOLS = {
    "GCUSD": "gold",
    "CLUSD": "crude_oil",
}
TREASURY_SERIES = {
    "US_TREASURY_13W": "month3",
    "US_TREASURY_5Y": "year5",
    "US_TREASURY_10Y": "year10",
}

RAW_ROOT = ML_ROOT / "data" / "raw_market_vendor"
REFERENCE_DIR = RAW_ROOT / "reference"
INTRADAY_DIR = RAW_ROOT / "intraday"
MACRO_DIR = RAW_ROOT / "macro"
REPORT_DIR = RAW_ROOT / "reports"


@dataclass(frozen=True)
class FetchStats:
    symbol: str
    asset_type: str
    rows: int
    first_ts_utc: str | None
    last_ts_utc: str | None
    trading_days: int
    regular_session_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch raw FMP market data into a clean landing area.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--chunk-days", type=int, default=25, help="Days per intraday request.")
    parser.add_argument("--limit-symbols", type=int, default=None, help="Limit the number of S&P 500 symbols.")
    parser.add_argument("--skip-treasuries", action="store_true", help="Skip Treasury daily data.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def ensure_api_key() -> None:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not found. Set it in .env or ml/.env.")


def ensure_dirs() -> None:
    for path in (REFERENCE_DIR, INTRADAY_DIR, MACRO_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "maverick-raw-fetch/1.0"})
    return session


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


def fetch_sp500_tickers(session: requests.Session) -> list[str]:
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
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    (REFERENCE_DIR / "sp500_constituents.json").write_text(
        json.dumps(
            {
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "count": len(tickers),
                "tickers": tickers,
                "raw": payload,
            },
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
    for url in INTRADAY_15M_URLS:
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
        return pd.DataFrame(
            columns=[
                "symbol",
                "asset_type",
                "source",
                "vendor_timestamp_local",
                "timestamp_ny",
                "timestamp_utc",
                "trade_date_ny",
                "bar_start_hhmm_ny",
                "is_regular_session_equity_window",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    frame = pd.DataFrame(payload)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{symbol} intraday response missing columns: {sorted(missing)}")

    frame = frame[list(required)].copy()
    frame["symbol"] = symbol
    frame["vendor_timestamp_local"] = frame["date"].astype(str)
    ts_ny = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(
        NY_TZ, nonexistent="shift_forward", ambiguous="NaT"
    )
    frame["timestamp_ny"] = ts_ny
    frame["timestamp_utc"] = ts_ny.dt.tz_convert("UTC")
    frame["trade_date_ny"] = ts_ny.dt.date
    frame["bar_start_hhmm_ny"] = ts_ny.dt.strftime("%H:%M")
    frame["is_regular_session_equity_window"] = (
        (frame["bar_start_hhmm_ny"] >= "09:30") & (frame["bar_start_hhmm_ny"] < "16:00")
    )
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").round().astype("Int64")
    frame["source"] = "FMP"
    return frame.drop(columns=["date"]).drop_duplicates(subset=["symbol", "timestamp_utc"]).sort_values("timestamp_utc")


def fetch_intraday_range(
    session: requests.Session,
    symbol: str,
    asset_type: str,
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
            frame["asset_type"] = asset_type
            chunks.append(frame)
        time.sleep(REQUEST_SLEEP_SECONDS)
        current = chunk_end + timedelta(days=1)

    if not chunks:
        return fetch_intraday_chunk(session, symbol, start_date, start_date).assign(asset_type=asset_type).iloc[0:0]
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["symbol", "timestamp_utc"]).sort_values("timestamp_utc")


def write_symbol_parquet(frame: pd.DataFrame, symbol: str) -> None:
    symbol_dir = INTRADAY_DIR / f"symbol={symbol}"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(symbol_dir / "bars.parquet", index=False)


def summarize_intraday(frame: pd.DataFrame, symbol: str, asset_type: str) -> FetchStats:
    if frame.empty:
        return FetchStats(symbol, asset_type, 0, None, None, 0, 0)
    return FetchStats(
        symbol=symbol,
        asset_type=asset_type,
        rows=int(len(frame)),
        first_ts_utc=str(frame["timestamp_utc"].min()),
        last_ts_utc=str(frame["timestamp_utc"].max()),
        trading_days=int(frame["trade_date_ny"].nunique()),
        regular_session_rows=int(frame["is_regular_session_equity_window"].sum()),
    )


def fetch_treasury_rates(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    payload: Any = None
    for url in TREASURY_URLS:
        try:
            payload = request_json(
                session,
                url,
                {"from": start_date.isoformat(), "to": end_date.isoformat(), "apikey": FMP_API_KEY},
            )
            if isinstance(payload, list):
                break
        except Exception:
            payload = None
    if not isinstance(payload, list):
        raise RuntimeError("Unable to fetch Treasury rates from FMP.")

    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(columns=["series_code", "trade_date", "value", "source"])
    if "date" not in frame.columns:
        raise RuntimeError("Treasury response is missing the `date` column.")

    frame["trade_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    records: list[dict[str, Any]] = []
    for series_code, field_name in TREASURY_SERIES.items():
        if field_name not in frame.columns:
            raise RuntimeError(f"Treasury response missing `{field_name}` for {series_code}.")
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
    out = pd.DataFrame(records).sort_values(["series_code", "trade_date"]).reset_index(drop=True)
    out.to_parquet(MACRO_DIR / "treasury_rates_daily.parquet", index=False)
    return out


def write_reports(stats: list[FetchStats], treasury_frame: pd.DataFrame | None, start_date: date, end_date: date) -> None:
    coverage = pd.DataFrame([s.__dict__ for s in stats])
    coverage.to_csv(REPORT_DIR / "intraday_coverage.csv", index=False)

    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols_fetched": len(stats),
        "intraday_rows_total": int(sum(s.rows for s in stats)),
        "regular_session_rows_total": int(sum(s.regular_session_rows for s in stats)),
        "treasury_rows_total": int(len(treasury_frame)) if treasury_frame is not None else 0,
        "commodities": COMMODITY_SYMBOLS,
        "treasury_series": TREASURY_SERIES,
    }
    (REPORT_DIR / "fetch_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    ensure_api_key()
    ensure_dirs()

    start_date = iso_date(args.start_date)
    end_date = iso_date(args.end_date)
    if start_date > end_date:
        raise ValueError("--start-date must be <= --end-date")

    session = build_session()
    sp500 = fetch_sp500_tickers(session)
    if args.limit_symbols is not None:
        sp500 = sp500[: max(0, args.limit_symbols)]

    targets: list[tuple[str, str]] = [(symbol, "equity") for symbol in sp500]
    targets.extend((symbol, "commodity") for symbol in COMMODITY_SYMBOLS)

    print(f"Fetching {len(targets)} intraday symbols into {RAW_ROOT}")
    print(f"Date range: {start_date} -> {end_date}")

    stats: list[FetchStats] = []
    for idx, (symbol, asset_type) in enumerate(targets, start=1):
        print(f"[{idx}/{len(targets)}] {symbol} ({asset_type})")
        frame = fetch_intraday_range(
            session=session,
            symbol=symbol,
            asset_type=asset_type,
            start_date=start_date,
            end_date=end_date,
            chunk_days=args.chunk_days,
            verbose=args.verbose,
        )
        write_symbol_parquet(frame, symbol)
        stat = summarize_intraday(frame, symbol, asset_type)
        stats.append(stat)
        print(f"  rows={stat.rows:,} trade_days={stat.trading_days} regular_rows={stat.regular_session_rows:,}")

    treasury_frame: pd.DataFrame | None = None
    if not args.skip_treasuries:
        print("Fetching Treasury daily series...")
        treasury_frame = fetch_treasury_rates(session, start_date, end_date)
        print(f"  treasury_rows={len(treasury_frame):,}")

    write_reports(stats, treasury_frame, start_date, end_date)
    print("Raw vendor fetch complete.")


if __name__ == "__main__":
    main()
