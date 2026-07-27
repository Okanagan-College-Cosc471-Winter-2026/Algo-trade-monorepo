"""
Shared FMP fetch helpers for the standalone EDA scripts
(fetch_top29_eda.py, fetch_macro_eda.py).

Two data shapes are fetched:

- 15-min intraday bars (`/stable/historical-chart/15min`) — regular session
  only, timestamped by bar *start*. The last bar of each day starts at
  15:45 and covers 15:45-16:00, so its `close` is the last print before the
  feed cuts off, NOT the official 4:00pm closing/settlement price.

- Daily official close (`/api/v3/historical-price-full/{symbol}`) — the same
  endpoint the DB pipeline uses for `market.daily_prices`
  (see monthly_fmp_refresh.py::fetch_daily_price_rows). This carries the
  true 4:00pm print and should be joined on (symbol, date) rather than
  appended as a synthetic 27th intraday bar, since the 26-bars/day shape is
  load-bearing for the model (FEATURE_CONTRACT.md: 26 horizons, one per
  15-min bar).
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "ml" / "ml" / ".env")

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_INTRADAY_BASE = "https://financialmodelingprep.com/stable"
FMP_DAILY_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"

DEFAULT_CHUNK_DAYS = 5   # FMP free tier handles ~5-day intraday chunks well
DEFAULT_DELAY_SEC = 0.4  # polite rate-limit


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    return s


def fetch_15m_chunk(session: requests.Session, symbol: str, from_: date, to: date) -> list[dict]:
    url = f"{FMP_INTRADAY_BASE}/historical-chart/15min"
    params = {"symbol": symbol, "from": from_.isoformat(), "to": to.isoformat(), "apikey": FMP_API_KEY}
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
    except Exception as exc:
        print(f"  WARN {symbol} {from_}→{to}: {exc}")
    return []


def fetch_15m_series(
    session: requests.Session,
    symbol: str,
    start: date,
    end: date,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    delay_sec: float = DEFAULT_DELAY_SEC,
) -> pd.DataFrame:
    """Regular-session 15-min OHLCV bars for one symbol, `start`..`end` inclusive."""
    rows: list[dict] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        rows.extend(fetch_15m_chunk(session, symbol, cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
        time.sleep(delay_sec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["symbol"] = symbol
    df["ts"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[["symbol", "ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)


def fetch_daily_close_series(session: requests.Session, symbol: str, start: date, end: date) -> pd.DataFrame:
    """Official daily OHLCV (true 4:00pm close) for one symbol, `start`..`end` inclusive."""
    url = FMP_DAILY_URL.format(symbol=symbol)
    params = {"from": start.isoformat(), "to": end.isoformat(), "apikey": FMP_API_KEY}
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        print(f"  WARN daily-close {symbol}: {exc}")
        return pd.DataFrame()

    historical = payload.get("historical") if isinstance(payload, dict) else None
    if not historical:
        return pd.DataFrame()

    df = pd.DataFrame(historical)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce")
    df = df.dropna(subset=["close"])
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def fetch_series_for_symbols(
    session: requests.Session,
    symbols: list[str],
    start: date,
    end: date,
    fetch_fn,
    label: str,
) -> pd.DataFrame:
    """Run `fetch_fn(session, symbol, start, end)` over `symbols`, concatenate, and report gaps."""
    frames = []
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {sym} ({label})")
        part = fetch_fn(session, sym, start, end)
        if part.empty:
            print(f"    WARNING: no {label} data for {sym}")
        else:
            print(f"    {len(part):,} rows")
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_or_fetch(
    csv_path: Path,
    symbols: list[str],
    key_col: str,
    start: date,
    end: date,
    fetch_fn,
    session: requests.Session,
    label: str,
    date_col: str = "ts",
) -> pd.DataFrame:
    """Load `csv_path` if present, fetching only symbols missing from the cache; else fetch all."""
    if csv_path.exists():
        print(f"Loading cached {label} from {csv_path.name} ...")
        df = pd.read_csv(csv_path, parse_dates=[date_col])
        already = set(df[key_col].unique())
        missing = [s for s in symbols if s not in already]
        if missing:
            print(f"Fetching {len(missing)} missing symbols: {missing}")
            new = fetch_series_for_symbols(session, missing, start, end, fetch_fn, label)
            if not new.empty:
                df = pd.concat([df, new], ignore_index=True)
                df.to_csv(csv_path, index=False)
        return df

    print(f"Fetching {len(symbols)} symbols for {label} from FMP ({start} → {end}) ...")
    df = fetch_series_for_symbols(session, symbols, start, end, fetch_fn, label)
    if df.empty:
        raise RuntimeError(f"No {label} data fetched at all. Check FMP_API_KEY.")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df):,} rows → {csv_path}")
    return df
