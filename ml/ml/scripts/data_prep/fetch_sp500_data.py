"""
fetch_sp500_data.py — Download 15-min extended-hours OHLCV data for all S&P 500 stocks.

Two modes per ticker:
  - Existing (29 stocks in final_data/): gap-fill from last date → 2026-03-23
  - New stocks: full 2-year range 2024-03-23 → 2026-03-23

Tracks every FMP API call in bandwidth_log.csv and bandwidth_summary.json.
Fully resumable — safe to Ctrl+C and restart at any point.

Usage:
    python ml/scripts/data_prep/fetch_sp500_data.py
    python ml/scripts/data_prep/fetch_sp500_data.py --dry-run
    python ml/scripts/data_prep/fetch_sp500_data.py --limit 5   # test first 5 tickers
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────

ML_ROOT = Path(__file__).resolve().parents[3] / "ml"

FINAL_DATA_DIR    = ML_ROOT / "data" / "final_data"
BANDWIDTH_LOG     = ML_ROOT / "data" / "bandwidth_log.csv"
BANDWIDTH_SUMMARY = ML_ROOT / "data" / "bandwidth_summary.json"
PROGRESS_FILE     = ML_ROOT / "data" / ".fetch_progress.json"
SP500_CACHE       = ML_ROOT / "data" / ".sp500_tickers_cache.json"

# ── Constants ─────────────────────────────────────────────────────────────────

FETCH_START      = "2024-03-23"
FETCH_END        = "2026-03-23"
COVERAGE_CUTOFF  = datetime(2026, 3, 23)

CHUNK_DAYS       = 25
SLEEP_S          = 0.5
REQUEST_TIMEOUT  = 30
MAX_RETRIES      = 3
RETRY_BACKOFF_S  = [2, 5, 15]

FMP_BASE         = "https://financialmodelingprep.com/api/v3"
SP500_CACHE_TTL_HOURS = 24

# ── Env ───────────────────────────────────────────────────────────────────────

load_dotenv(ML_ROOT / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# ── BandwidthTracker ──────────────────────────────────────────────────────────

class BandwidthTracker:
    """Tracks every FMP API call; logs to CSV and updates JSON summary."""

    _CSV_HEADER = [
        "timestamp_utc", "ticker", "chunk_start", "chunk_end",
        "http_status", "response_bytes", "duration_ms",
        "rows_returned", "retry_attempt", "error_message",
    ]

    def __init__(self, log_path: Path, summary_path: Path):
        self._log_path     = log_path
        self._summary_path = summary_path

        log_path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not log_path.exists() or log_path.stat().st_size == 0
        self._fh = open(log_path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if is_new:
            self._writer.writerow(self._CSV_HEADER)
            self._fh.flush()

        now_utc = datetime.now(timezone.utc).isoformat()
        if summary_path.exists():
            with open(summary_path) as f:
                self._summary = json.load(f)
        else:
            self._summary = {
                "script_started_at_utc":  now_utc,
                "last_updated_utc":       now_utc,
                "tickers_completed":      0,
                "tickers_skipped":        0,
                "tickers_failed":         0,
                "total_api_calls":        0,
                "successful_calls":       0,
                "failed_calls":           0,
                "retried_calls":          0,
                "total_bytes_received":   0,
                "total_bytes_received_mb": 0.0,
                "total_rows_written":     0,
                "average_call_duration_ms": 0.0,
                "calls_by_http_status":   {},
                "empty_response_calls":   0,
                "last_ticker_completed":  None,
            }

    def log_call(
        self,
        ticker: str,
        chunk_start: str,
        chunk_end: str,
        http_status: int,
        response_bytes: int,
        duration_ms: float,
        rows_returned: int,
        retry_attempt: int = 0,
        error_message: str = "",
    ):
        ts = datetime.now(timezone.utc).isoformat()
        self._writer.writerow([
            ts, ticker, chunk_start, chunk_end,
            http_status, response_bytes, f"{duration_ms:.1f}",
            rows_returned, retry_attempt, error_message,
        ])
        self._fh.flush()

        s = self._summary
        s["total_api_calls"] += 1
        s["total_bytes_received"] += response_bytes
        s["total_bytes_received_mb"] = round(s["total_bytes_received"] / (1024 * 1024), 2)

        status_key = str(http_status)
        s["calls_by_http_status"][status_key] = s["calls_by_http_status"].get(status_key, 0) + 1

        if 200 <= http_status < 300:
            s["successful_calls"] += 1
        else:
            s["failed_calls"] += 1

        if retry_attempt > 0:
            s["retried_calls"] += 1

        if rows_returned == 0 and not error_message:
            s["empty_response_calls"] += 1

        # Rolling average duration
        prev_avg = s["average_call_duration_ms"]
        n = s["total_api_calls"]
        s["average_call_duration_ms"] = round(prev_avg + (duration_ms - prev_avg) / n, 2)

    def record_ticker_done(
        self,
        ticker: str,
        skipped: bool = False,
        failed: bool = False,
        rows_written: int = 0,
    ):
        s = self._summary
        if skipped:
            s["tickers_skipped"] += 1
        elif failed:
            s["tickers_failed"] += 1
        else:
            s["tickers_completed"] += 1
            s["total_rows_written"] += rows_written
            s["last_ticker_completed"] = ticker
        self._save_summary()

    def _save_summary(self):
        self._summary["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = Path(str(self._summary_path) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._summary, f, indent=2)
        os.replace(tmp, self._summary_path)

    def close(self):
        self._fh.flush()
        self._fh.close()
        self._save_summary()

    def print_summary(self):
        s = self._summary
        mb = s["total_bytes_received_mb"]
        gb = round(mb / 1024, 3)
        print(f"\n{'='*55}")
        print(f"  FMP Bandwidth Summary")
        print(f"{'='*55}")
        print(f"  Tickers completed : {s['tickers_completed']}")
        print(f"  Tickers skipped   : {s['tickers_skipped']}")
        print(f"  Tickers failed    : {s['tickers_failed']}")
        print(f"  Total API calls   : {s['total_api_calls']:,}")
        print(f"  Successful calls  : {s['successful_calls']:,}")
        print(f"  Failed calls      : {s['failed_calls']:,}")
        print(f"  Retried calls     : {s['retried_calls']:,}")
        print(f"  Empty responses   : {s['empty_response_calls']:,}")
        print(f"  Data received     : {mb:,.1f} MB  ({gb} GB)")
        print(f"  Rows written      : {s['total_rows_written']:,}")
        print(f"  Avg call duration : {s['average_call_duration_ms']:.1f} ms")
        print(f"  Status breakdown  : {s['calls_by_http_status']}")
        print(f"{'='*55}")


# ── S&P 500 List ──────────────────────────────────────────────────────────────

def fetch_sp500_tickers(api_key: str, cache_path: Path) -> list[str]:
    """Return S&P 500 tickers from cache (< 24h) or FMP API."""
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < SP500_CACHE_TTL_HOURS:
            with open(cache_path) as f:
                data = json.load(f)
            tickers = data.get("tickers", [])
            if tickers:
                print(f"  S&P 500 list loaded from cache ({len(tickers)} tickers, {age_hours:.1f}h old)")
                return tickers

    print("  Fetching S&P 500 constituent list from FMP...")
    try:
        url = f"{FMP_BASE}/sp500_constituent?apikey={api_key}"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected response format: {type(data)}")
        tickers = sorted(set(item["symbol"] for item in data if "symbol" in item))
        print(f"  Fetched {len(tickers)} S&P 500 tickers")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"tickers": tickers, "fetched_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        return tickers
    except Exception as exc:
        print(f"  WARNING: Could not fetch S&P 500 list: {exc}")
        if cache_path.exists():
            print("  Falling back to stale cache.")
            with open(cache_path) as f:
                return json.load(f).get("tickers", [])
        raise RuntimeError("No S&P 500 ticker list available and no cache to fall back to.") from exc


# ── Gap Detection ─────────────────────────────────────────────────────────────

def detect_gap(csv_path: Path) -> tuple[datetime | None, datetime | None]:
    """
    Returns (gap_start, COVERAGE_CUTOFF) if data is missing before cutoff.
    Returns (None, None) if already covered or CSV is unreadable.
    """
    try:
        df = pd.read_csv(csv_path, usecols=["date"], parse_dates=["date"])
        if df.empty:
            return pd.to_datetime(FETCH_START), COVERAGE_CUTOFF
        max_date = df["date"].max()
        if pd.isna(max_date):
            return pd.to_datetime(FETCH_START), COVERAGE_CUTOFF
        if max_date.date() >= COVERAGE_CUTOFF.date():
            return None, None
        gap_start = (max_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return gap_start, COVERAGE_CUTOFF
    except Exception as exc:
        print(f"    WARNING: Could not read {csv_path.name}: {exc}")
        return None, None


# ── Progress Helpers ──────────────────────────────────────────────────────────

def load_progress(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": [], "failed": []}


def save_progress(path: Path, progress: dict):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, path)


def is_done(ticker: str, progress: dict, final_data_dir: Path) -> bool:
    if ticker not in progress["completed"]:
        return False
    csv_path = final_data_dir / f"{ticker}.csv"
    if not csv_path.exists():
        return False
    gap_start, _ = detect_gap(csv_path)
    return gap_start is None


# ── Chunked FMP Fetch ─────────────────────────────────────────────────────────

def fetch_ticker_chunks(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    api_key: str,
    bw: BandwidthTracker,
) -> pd.DataFrame | None:
    """Fetch 15-min extended-hours data in 25-day chunks with retry logic."""
    all_chunks: list[pd.DataFrame] = []
    current_start = start_dt

    while current_start.date() <= end_dt.date():
        chunk_end = min(current_start + timedelta(days=CHUNK_DAYS), end_dt)
        str_start = current_start.strftime("%Y-%m-%d")
        str_end   = chunk_end.strftime("%Y-%m-%d")

        url = (
            f"{FMP_BASE}/historical-chart/15min/{ticker}"
            f"?from={str_start}&to={str_end}&extended=true&apikey={api_key}"
        )

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
                duration_ms = (time.perf_counter() - t0) * 1000
                response_bytes = len(resp.content)

                if resp.status_code == 429 or resp.status_code >= 500:
                    backoff = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
                    bw.log_call(ticker, str_start, str_end, resp.status_code,
                                response_bytes, duration_ms, 0, attempt,
                                f"HTTP {resp.status_code}, retrying in {backoff}s")
                    print(f"    [{ticker}] HTTP {resp.status_code} on chunk {str_start}→{str_end}, "
                          f"retrying in {backoff}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(backoff)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Guard against non-list responses (e.g. {"message": "..."})
                if not isinstance(data, list):
                    data = []

                bw.log_call(ticker, str_start, str_end, resp.status_code,
                            response_bytes, duration_ms, len(data), attempt)

                if data:
                    df_chunk = pd.DataFrame(data)[["date", "open", "low", "high", "close", "volume"]]
                    df_chunk["date"] = pd.to_datetime(df_chunk["date"])
                    all_chunks.append(df_chunk)

                success = True
                break

            except requests.Timeout:
                backoff = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
                bw.log_call(ticker, str_start, str_end, 0, 0,
                            0, 0, attempt, f"Timeout, retrying in {backoff}s")
                print(f"    [{ticker}] Timeout on chunk {str_start}→{str_end}, "
                      f"retrying in {backoff}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(backoff)

            except requests.ConnectionError as exc:
                backoff = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
                bw.log_call(ticker, str_start, str_end, 0, 0,
                            0, 0, attempt, f"ConnectionError: {exc}")
                print(f"    [{ticker}] Connection error, retrying in {backoff}s")
                time.sleep(backoff)

            except ValueError as exc:
                # JSONDecodeError subclasses ValueError
                bw.log_call(ticker, str_start, str_end, 0, 0,
                            0, 0, attempt, f"JSONDecodeError: {exc}")
                print(f"    [{ticker}] JSON decode error on chunk {str_start}→{str_end}, skipping")
                success = True  # don't retry malformed JSON — just skip chunk
                break

        if not success:
            print(f"    [{ticker}] All {MAX_RETRIES} attempts failed for chunk {str_start}→{str_end}, skipping")

        time.sleep(SLEEP_S)

        if chunk_end.date() >= end_dt.date():
            break
        current_start = chunk_end + timedelta(days=1)

    if not all_chunks:
        return None

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    return df


# ── Save Ticker Data ──────────────────────────────────────────────────────────

def save_ticker_data(
    ticker: str,
    new_df: pd.DataFrame,
    final_data_dir: Path,
    is_gap_fill: bool,
) -> int:
    """Write new_df to CSV. For gap-fill, merges with existing data. Returns new row count."""
    csv_path = final_data_dir / f"{ticker}.csv"
    col_order = ["date", "open", "low", "high", "close", "volume"]

    if is_gap_fill and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date", ascending=True).reset_index(drop=True)
            combined[col_order].to_csv(csv_path, index=False)
            return len(combined) - len(existing)
        except Exception as exc:
            print(f"    WARNING: Could not merge with existing {ticker}.csv: {exc}. Writing new_df only.")

    new_df[col_order].to_csv(csv_path, index=False)
    return len(new_df)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch S&P 500 15-min extended-hours data from FMP.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be fetched without making API calls.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N tickers (for testing).")
    parser.add_argument("--reset-progress", action="store_true",
                        help="Delete progress file and start fresh.")
    parser.add_argument("--existing-only", action="store_true",
                        help="Only process tickers that already have CSVs in final_data/ (gap-fill mode).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY not found. Set it in ml/.env")

    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. S&P 500 ticker list
    print("Step 1: Loading S&P 500 tickers...")
    tickers = fetch_sp500_tickers(FMP_API_KEY, SP500_CACHE)

    if args.existing_only:
        existing_set = {p.stem for p in FINAL_DATA_DIR.glob("*.csv")}
        tickers = [t for t in tickers if t in existing_set]
        print(f"  (--existing-only: restricted to {len(tickers)} tickers with existing CSVs)")

    if args.limit:
        tickers = tickers[: args.limit]
        print(f"  (limited to first {args.limit} tickers)")

    # 2. Progress
    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("  Progress file reset.")
    progress = load_progress(PROGRESS_FILE)

    # 3. Existing CSV set
    existing_csvs = {p.stem for p in FINAL_DATA_DIR.glob("*.csv")}

    # 4. Bandwidth tracker
    bw = BandwidthTracker(BANDWIDTH_LOG, BANDWIDTH_SUMMARY)

    fetch_start_dt = pd.to_datetime(FETCH_START)
    fetch_end_dt   = pd.to_datetime(FETCH_END)

    total = len(tickers)
    completed = skipped = failed = 0

    print(f"\nStep 2: Processing {total} tickers...\n")

    try:
        for i, ticker in enumerate(tickers, 1):
            prefix = f"[{i:>3}/{total}] {ticker:<8}"

            # Resume check
            if is_done(ticker, progress, FINAL_DATA_DIR):
                print(f"{prefix} SKIP (already complete)")
                bw.record_ticker_done(ticker, skipped=True)
                skipped += 1
                continue

            csv_path = FINAL_DATA_DIR / f"{ticker}.csv"

            # Determine fetch range
            if ticker in existing_csvs:
                gap_start, gap_end = detect_gap(csv_path)
                if gap_start is None:
                    # CSV exists and is fully covered — just mark done
                    if ticker not in progress["completed"]:
                        progress["completed"].append(ticker)
                        save_progress(PROGRESS_FILE, progress)
                    bw.record_ticker_done(ticker, skipped=True)
                    skipped += 1
                    print(f"{prefix} SKIP (CSV already covers cutoff)")
                    continue
                start_dt, end_dt = gap_start, gap_end
                is_gap_fill = True
                print(f"{prefix} GAP FILL {start_dt.date()} → {end_dt.date()}")
            else:
                start_dt, end_dt = fetch_start_dt, fetch_end_dt
                is_gap_fill = False
                print(f"{prefix} NEW      {start_dt.date()} → {end_dt.date()}")

            if args.dry_run:
                print(f"           (dry run — skipping actual fetch)")
                continue

            # Fetch
            df = fetch_ticker_chunks(ticker, start_dt, end_dt, FMP_API_KEY, bw)

            if df is None or df.empty:
                print(f"           WARNING: no data returned")
                if ticker not in progress["failed"]:
                    progress["failed"].append(ticker)
                save_progress(PROGRESS_FILE, progress)
                bw.record_ticker_done(ticker, failed=True)
                failed += 1
                continue

            # Save
            rows_written = save_ticker_data(ticker, df, FINAL_DATA_DIR, is_gap_fill)
            print(f"           saved {rows_written:,} rows  "
                  f"(max date: {df['date'].max().strftime('%Y-%m-%d %H:%M')})")

            if ticker not in progress["completed"]:
                progress["completed"].append(ticker)
            if ticker in progress["failed"]:
                progress["failed"].remove(ticker)
            save_progress(PROGRESS_FILE, progress)
            bw.record_ticker_done(ticker, rows_written=rows_written)
            completed += 1

    finally:
        bw.close()

    print(f"\nDone. completed={completed}  skipped={skipped}  failed={failed}")
    bw.print_summary()


if __name__ == "__main__":
    main()
