from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class SessionGate:
    now_et: dt.datetime
    session_open_et: dt.datetime | None
    session_close_et: dt.datetime | None
    is_trading_day: bool
    is_open_now: bool
    is_half_day: bool
    reason: str


def _calendar():
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise RuntimeError(
            "exchange_calendars is required for market session gating. "
            "Install it in the Airflow runtime."
        ) from exc
    return xcals.get_calendar("XNYS")


def get_session_gate(now: dt.datetime | None = None, tz_name: str = "America/New_York") -> SessionGate:
    xnys = _calendar()
    market_tz = ZoneInfo(tz_name)
    now_et = (now or dt.datetime.now(dt.timezone.utc)).astimezone(market_tz)
    schedule = xnys.schedule.loc[str(now_et.date()) : str(now_et.date())]
    if schedule.empty:
        return SessionGate(
            now_et=now_et,
            session_open_et=None,
            session_close_et=None,
            is_trading_day=False,
            is_open_now=False,
            is_half_day=False,
            reason="holiday",
        )

    row = schedule.iloc[0]
    open_col = "open" if "open" in row else "market_open"
    close_col = "close" if "close" in row else "market_close"
    session_open_et = row[open_col].to_pydatetime().astimezone(market_tz)
    session_close_et = row[close_col].to_pydatetime().astimezone(market_tz)
    is_open_now = session_open_et <= now_et <= session_close_et
    is_half_day = session_close_et.timetz().replace(tzinfo=None) < dt.time(16, 0)
    return SessionGate(
        now_et=now_et,
        session_open_et=session_open_et,
        session_close_et=session_close_et,
        is_trading_day=True,
        is_open_now=is_open_now,
        is_half_day=is_half_day,
        reason="open" if is_open_now else "outside_rth",
    )


def expected_last_closed_window_start_utc(now: dt.datetime | None = None, tz_name: str = "America/New_York") -> dt.datetime | None:
    gate = get_session_gate(now=now, tz_name=tz_name)
    if not gate.is_open_now or gate.session_open_et is None or gate.session_close_et is None:
        return None
    now_et = gate.now_et.replace(second=0, microsecond=0)
    floor_end = now_et - dt.timedelta(minutes=now_et.minute % 15)
    window_start_et = floor_end - dt.timedelta(minutes=15)
    window_end_et = floor_end
    if window_start_et < gate.session_open_et or window_end_et > gate.session_close_et:
        return None
    return window_start_et.astimezone(dt.timezone.utc)
