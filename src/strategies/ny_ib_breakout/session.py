"""
DST-correct NY session boundary computation for the IB Breakout strategy.

NY open is 09:30 America/New_York local time.
In UTC:
  Summer (EDT, roughly Mar–Nov): 09:30 ET = 13:30 UTC
  Winter (EST, roughly Nov–Mar): 09:30 ET = 14:30 UTC

We derive the UTC time directly from zoneinfo so daylight saving transitions
are handled correctly — including the asymmetric US/UK DST crossover weeks.

Session boundaries returned (all UTC-aware pd.Timestamp):
  ny_open           : 09:30 ET in UTC on the given trading date
  ib_close          : ny_open + ib_duration_minutes
  signal_window_close: ny_open + ib_signal_window_hours
  time_exit         : ny_open + time_exit_hours_after_ny_open
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timezone

import pandas as pd
from zoneinfo import ZoneInfo

from .params import NYIBParams

_NY_TZ = ZoneInfo("America/New_York")
_NY_OPEN_LOCAL = time(9, 30)   # 09:30 ET — NYSE open


@dataclass(frozen=True)
class NYSessionBoundaries:
    """
    All session timestamps for one trading day, expressed as UTC-aware pd.Timestamps.

    ny_open             : 09:30 ET in UTC on the given trading date.
    ib_close            : End of the initial balance period.
    signal_window_close : Last candle start that can generate a signal.
    time_exit           : Hard time exit for any open position.
    """
    ny_open: pd.Timestamp
    ib_close: pd.Timestamp
    signal_window_close: pd.Timestamp
    time_exit: pd.Timestamp


def get_ny_session_boundaries(
    trading_date: date,
    params: NYIBParams,
) -> NYSessionBoundaries:
    """
    Compute DST-correct NY session boundaries for a given UTC trading date.

    Note: 'trading_date' is the UTC calendar date.  For gold (24-hour market),
    there are no trading-date vs. settlement-date discrepancies to worry about.

    XAUUSD trades 23 hours/day (closes briefly at 17:00 ET).  Friday sessions
    end at 17:00 ET.  We do not model the Friday early close here — low-volume
    late Friday bars will simply not trigger signals in the signal window.
    """
    # Build 09:30 ET on this date, then convert to UTC
    dt_local = datetime.combine(trading_date, _NY_OPEN_LOCAL, tzinfo=_NY_TZ)
    dt_utc = dt_local.astimezone(timezone.utc)
    ny_open = pd.Timestamp(dt_utc)

    ib_close           = ny_open + pd.Timedelta(minutes=params.ib_duration_minutes)
    signal_window_close = ny_open + pd.Timedelta(hours=params.ib_signal_window_hours)
    time_exit          = ny_open + pd.Timedelta(hours=params.time_exit_hours_after_ny_open)

    return NYSessionBoundaries(
        ny_open=ny_open,
        ib_close=ib_close,
        signal_window_close=signal_window_close,
        time_exit=time_exit,
    )
