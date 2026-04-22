"""
NYSE session utilities backed by exchange_calendars (XNYS).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import exchange_calendars as xcals


RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)
XNYS_CODE = "XNYS"


@dataclass(frozen=True)
class MarketSessionStatus:
    now_et: dt.datetime
    session_date: dt.date
    is_trading_day: bool
    is_open_now: bool
    is_half_day: bool
    session_open_et: dt.datetime | None
    session_close_et: dt.datetime | None
    reason: str


_CALENDAR = xcals.get_calendar(XNYS_CODE)


def _to_et(ts: dt.datetime, tz: ZoneInfo) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=tz)
    return ts.astimezone(tz)


def _session_for_date(session_date: dt.date, market_tz: ZoneInfo) -> tuple[dt.datetime, dt.datetime] | None:
    schedule = _CALENDAR.schedule.loc[str(session_date) : str(session_date)]
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    open_col = "open" if "open" in row else "market_open"
    close_col = "close" if "close" in row else "market_close"
    open_et = row[open_col].to_pydatetime().astimezone(market_tz)
    close_et = row[close_col].to_pydatetime().astimezone(market_tz)
    return open_et, close_et


def get_market_session_status(now: dt.datetime | None = None, market_tz_name: str = "America/New_York") -> MarketSessionStatus:
    market_tz = ZoneInfo(market_tz_name)
    now_et = _to_et(now or dt.datetime.now(dt.timezone.utc), market_tz)
    session_date = now_et.date()
    session = _session_for_date(session_date, market_tz)
    if session is None:
        return MarketSessionStatus(
            now_et=now_et,
            session_date=session_date,
            is_trading_day=False,
            is_open_now=False,
            is_half_day=False,
            session_open_et=None,
            session_close_et=None,
            reason="holiday",
        )

    session_open_et, session_close_et = session
    is_open_now = session_open_et <= now_et < session_close_et
    is_half_day = session_close_et.timetz().replace(tzinfo=None) < RTH_CLOSE
    reason = "open" if is_open_now else "outside_rth"
    return MarketSessionStatus(
        now_et=now_et,
        session_date=session_date,
        is_trading_day=True,
        is_open_now=is_open_now,
        is_half_day=is_half_day,
        session_open_et=session_open_et,
        session_close_et=session_close_et,
        reason=reason,
    )


def compute_last_closed_rth_window_start_utc(
    now: dt.datetime | None = None,
    market_tz_name: str = "America/New_York",
) -> tuple[dt.datetime | None, MarketSessionStatus]:
    status = get_market_session_status(now=now, market_tz_name=market_tz_name)
    if not status.is_open_now or status.session_open_et is None or status.session_close_et is None:
        return None, status

    now_et = status.now_et.replace(second=0, microsecond=0)
    floor_end = now_et - dt.timedelta(minutes=now_et.minute % 15)
    window_start_et = floor_end - dt.timedelta(minutes=15)
    window_end_et = floor_end
    if window_start_et < status.session_open_et or window_end_et > status.session_close_et:
        return None, status
    return window_start_et.astimezone(dt.timezone.utc), status
