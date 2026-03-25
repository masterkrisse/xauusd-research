"""
DST-aware session boundary calculator.

London opens at:
  08:00 UTC  during GMT  (last Sunday of October → last Sunday of March)
  07:00 UTC  during BST  (last Sunday of March   → last Sunday of October)

All output timestamps are UTC-aware pandas Timestamps.

Known limitation (Logic Risk #4):
  On the two clocks-change nights per year, the Asian session may have an
  anomalous candle count (one hour missing or duplicated). The boundaries
  computed here are correct; the caller is responsible for detecting
  sparse-candle days.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

_LONDON_TZ = ZoneInfo("Europe/London")
_UTC = timezone.utc


def _is_bst(d: date) -> bool:
    """
    Return True if the given calendar date falls within British Summer Time (UTC+1).

    Checks the UTC offset of Europe/London at noon on that date (safely away from
    the transition hour at 01:00 UTC).
    """
    noon_utc = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=_UTC)
    noon_london = noon_utc.astimezone(_LONDON_TZ)
    return noon_london.utcoffset().total_seconds() == 3600


@dataclass(frozen=True)
class SessionBoundaries:
    """
    All relevant UTC timestamps for one trading day.

    asia_start:           00:00 UTC  (always)
    london_open:          07:00 UTC  (BST) or 08:00 UTC  (GMT)
    london_window_close:  london_open + london_window_duration_hours
    time_exit:            london_open + time_exit_hours_after_london_open
    trading_date:         The calendar date this boundary set applies to
    is_bst:               True if British Summer Time is in effect
    """
    trading_date: date
    asia_start: pd.Timestamp
    london_open: pd.Timestamp
    london_window_close: pd.Timestamp
    time_exit: pd.Timestamp
    is_bst: bool


def get_session_boundaries(
    trading_date: date,
    london_window_duration_hours: float,
    time_exit_hours_after_london_open: float,
    candle_minutes: int,          # kept for documentation; not used in boundary maths
) -> SessionBoundaries:
    """
    Compute session boundaries for one trading date.

    The Asian range is defined as all candles whose open timestamp falls in
    [asia_start, london_open).  The candle that opens at
    (london_open - candle_minutes) closes exactly at london_open and is the
    last Asian candle.  The candle that opens at london_open is the first
    eligible breakout candle.

    Args:
        trading_date:                     The UTC calendar date being processed.
        london_window_duration_hours:     How many hours after London open to accept signals.
        time_exit_hours_after_london_open: Hard close time relative to London open.
        candle_minutes:                   Candle timeframe in minutes (informational).

    Returns:
        SessionBoundaries with all timestamps as UTC-aware pd.Timestamps.
    """
    bst = _is_bst(trading_date)
    london_open_hour_utc = 7 if bst else 8

    asia_start_dt = datetime(
        trading_date.year, trading_date.month, trading_date.day,
        0, 0, 0, tzinfo=_UTC,
    )
    london_open_dt = datetime(
        trading_date.year, trading_date.month, trading_date.day,
        london_open_hour_utc, 0, 0, tzinfo=_UTC,
    )
    london_window_close_dt = london_open_dt + pd.Timedelta(
        hours=london_window_duration_hours
    )
    time_exit_dt = london_open_dt + pd.Timedelta(
        hours=time_exit_hours_after_london_open
    )

    boundaries = SessionBoundaries(
        trading_date=trading_date,
        asia_start=pd.Timestamp(asia_start_dt),
        london_open=pd.Timestamp(london_open_dt),
        london_window_close=pd.Timestamp(london_window_close_dt),
        time_exit=pd.Timestamp(time_exit_dt),
        is_bst=bst,
    )

    logger.debug(
        "[SESSION] %s | BST=%s | LondonOpen=%s UTC | WindowClose=%s UTC | TimeExit=%s UTC",
        trading_date,
        bst,
        boundaries.london_open.strftime("%H:%M"),
        boundaries.london_window_close.strftime("%H:%M"),
        boundaries.time_exit.strftime("%H:%M"),
    )

    return boundaries
