"""
20-session MA slope macro filter for the Short-Only Fade strategy.

Definition:
  session_close(t) = last 15-min bar close strictly before 17:00 UTC on day t.
  MA(t)            = mean of session_close values for the 20 sessions ending at t.
  slope(t)         = MA(t) - MA(t - slope_lookback)

  Bearish filter: slope(t) < 0

The slope measures whether the one-month rolling average is lower than it was
one week ago.  A declining MA signals that the medium-term trend is downward.

This is structurally different from the 10-session close-to-close trend used
in combined_fade:
  - Smoothed (MA) vs. raw (point-to-point close)
  - 20-session horizon vs. 10-session
  - Weekly slope comparison vs. single comparison

Design intent: capture sustained macro downtrend with less sensitivity to
single-session noise.

Filter state at session_start S is determined BEFORE that session opens —
using only session closes from prior completed sessions.  No lookahead.
"""

import logging
from typing import Optional

import pandas as pd

from .params import SOFParams

logger = logging.getLogger(__name__)

_SESSION_HOUR_UTC = 17
FILTER_BEARISH    = "bearish"
FILTER_NEUTRAL    = "neutral"   # MA not declining
FILTER_UNDEFINED  = "undefined" # insufficient history


def build_filter_index(df: pd.DataFrame, params: SOFParams) -> pd.Series:
    """
    Pre-compute the macro filter state for every 17:00 UTC boundary in df.

    Returns a Series indexed by pd.Timestamp (17:00 UTC boundaries).
    Values: FILTER_BEARISH, FILTER_NEUTRAL, or FILTER_UNDEFINED.
    """
    # ── Collect session closes ─────────────────────────────────────────────────
    dates = sorted(df.index.normalize().unique())
    session_closes: dict[pd.Timestamp, float] = {}

    for d in dates:
        boundary = d + pd.Timedelta(hours=_SESSION_HOUR_UTC)
        prior_bars = df[df.index < boundary]
        if not prior_bars.empty:
            session_closes[boundary] = float(prior_bars.iloc[-1]["close"])

    if not session_closes:
        return pd.Series(dtype=str)

    close_series = pd.Series(session_closes).sort_index()

    # ── Rolling MA and slope ───────────────────────────────────────────────────
    ma     = close_series.rolling(params.ma_period).mean()
    slope  = ma - ma.shift(params.slope_lookback)

    # ── Map to filter state ────────────────────────────────────────────────────
    filter_states: dict[pd.Timestamp, str] = {}
    for ts in close_series.index:
        if pd.isna(slope.get(ts, float("nan"))):
            filter_states[ts] = FILTER_UNDEFINED
        elif slope[ts] < 0.0:
            filter_states[ts] = FILTER_BEARISH
        else:
            filter_states[ts] = FILTER_NEUTRAL

    result = pd.Series(filter_states)
    n_bearish  = (result == FILTER_BEARISH).sum()
    n_neutral  = (result == FILTER_NEUTRAL).sum()
    n_undef    = (result == FILTER_UNDEFINED).sum()
    logger.info(
        "[SOF FILTER] Built | Sessions=%d | Bearish=%d (%.0f%%) | Neutral=%d | Undefined=%d",
        len(result), n_bearish, n_bearish / max(len(result), 1) * 100,
        n_neutral, n_undef,
    )
    return result


def get_filter_state(
    session_start: pd.Timestamp,
    filter_index: pd.Series,
) -> str:
    """
    Return the filter state for a given session start timestamp.

    The filter state uses session closes PRIOR to session_start.
    (The session_start boundary itself is in the index, valued from prior closes.)
    """
    if session_start not in filter_index.index:
        return FILTER_UNDEFINED
    return str(filter_index[session_start])
