"""
Regime filter for the Combined Prior-Day Fade strategy.

Design rationale (derived from trade analysis, not optimization):

  SHORT trades in prior-day fade:
    10-session uptrend   → WR 21-26%, Exp -0.32 to -0.39R  (both IS and OOS)
    10-session downtrend → WR 43-49%, Exp +0.49 to +0.51R  (both IS and OOS)

  LONG trades in prior-day fade:
    10-session uptrend   → WR 36-43%, Exp -0.00 to +0.34R
    10-session downtrend → WR 25-30%, Exp -0.08 to -0.26R

  The 10-session trend sign cleanly separates profitable from unprofitable
  conditions for both legs.  This is structural: gold's directional momentum
  determines whether a PDH sweep is a false breakout or the start of an
  extension.

Regime definition:
  At the START of each 17:00 UTC session, compute:
    trend_10 = (session_close_now - session_close_10_sessions_ago) / session_close_10_sessions_ago

  Where "session close" = the last 15-minute candle close strictly before
  the 17:00 UTC session boundary.

  Regime:
    UPTREND   : trend_10 > 0  → take LONGS only
    DOWNTREND : trend_10 < 0  → take SHORTS only
    UNDEFINED : fewer than 10 prior sessions available → no trade

No threshold is applied to the trend value — we use the sign only.
This is intentional: introducing a magnitude threshold would be parameter
optimization.  The sign alone is the structural observation.

No-trade zones:
  There is no "both" case and no "neither" case in normal operation.
    - If trend_10 > 0: long signals taken, short signals skipped.
    - If trend_10 < 0: short signals taken, long signals skipped.
    - If trend_10 = 0 or undefined: no trade.
  A session can only have one regime state.  Since LONG and SHORT signals
  can't fire on the same candle (price can't be both above PDH and below PDL),
  there is never a conflict to resolve.
"""

from typing import Optional

import pandas as pd

_SESSION_HOUR_UTC = 17
REGIME_UPTREND   = "uptrend"
REGIME_DOWNTREND = "downtrend"
REGIME_UNDEFINED = "undefined"


def build_session_close_index(df: pd.DataFrame) -> pd.Series:
    """
    Pre-compute 'session closes' for all dates in df.

    A session close is the last 15-minute bar close strictly before 17:00 UTC
    on each UTC calendar date.

    Returns a Series indexed by pd.Timestamp (the 17:00 UTC boundary itself),
    so lookups can be done by session_start timestamp.
    """
    closes: dict[pd.Timestamp, float] = {}
    dates = sorted(df.index.normalize().unique())

    for d in dates:
        session_boundary = d + pd.Timedelta(hours=_SESSION_HOUR_UTC)
        prior_bars = df[df.index < session_boundary]
        if not prior_bars.empty:
            closes[session_boundary] = float(prior_bars.iloc[-1]["close"])

    return pd.Series(closes)


def get_regime(
    session_start: pd.Timestamp,
    session_close_index: pd.Series,
    lookback: int = 10,
) -> str:
    """
    Return the regime for a given session start.

    Args:
        session_start:        The 17:00 UTC timestamp of the current session.
        session_close_index:  Pre-computed series from build_session_close_index.
        lookback:             Number of prior sessions to measure trend over.
                              Default 10 (two calendar weeks).

    Returns:
        REGIME_UPTREND, REGIME_DOWNTREND, or REGIME_UNDEFINED.
    """
    all_ts = session_close_index.index.tolist()

    if session_start not in session_close_index.index:
        return REGIME_UNDEFINED

    try:
        idx = all_ts.index(session_start)
    except ValueError:
        return REGIME_UNDEFINED

    if idx < lookback:
        return REGIME_UNDEFINED

    current_close = session_close_index.iloc[idx]
    past_close    = session_close_index.iloc[idx - lookback]

    if past_close <= 0:
        return REGIME_UNDEFINED

    trend = (current_close - past_close) / past_close

    if trend > 0:
        return REGIME_UPTREND
    elif trend < 0:
        return REGIME_DOWNTREND
    else:
        return REGIME_UNDEFINED   # exactly flat: too rare to worry about, skip
