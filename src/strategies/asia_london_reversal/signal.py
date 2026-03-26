"""
Signal detection for the Asia–London Session Reversal.

Asian session definition:
  Start : first 15-min candle at or after session_start (17:00 UTC)
  End   : last 15-min candle whose timestamp is < london_open_ts (07:00 UTC)
          This captures the 06:45 UTC candle as the final Asian bar.

Open price  : open of the first Asian candle (17:00 UTC bar)
Close price : close of the last Asian candle (06:45 UTC bar)
High / Low  : rolling extremes across all Asian candles

Net move:
  net_pct = (close_price - open_price) / open_price

Signal:
  If net_pct >= +min_asian_move_pct → Asian session went UP → SHORT at London open
  If net_pct <= -min_asian_move_pct → Asian session went DOWN → LONG at London open

TP:
  50% retrace of the net move = (open_price + close_price) / 2
  This is the midpoint of the Asian session open and close — if the Asian session
  retraces halfway, the hypothesis has been confirmed.

Stop:
  LONG : Asian session low  - stop_buffer
  SHORT: Asian session high + stop_buffer
  The stop is placed beyond the Asian extreme, not just beyond the close.
  This correctly reflects the risk if the trend resumes from its worst point.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .params import ALParams

logger = logging.getLogger(__name__)


@dataclass
class AsianSessionInfo:
    """Characterisation of a single Asian trading session."""
    session_start: pd.Timestamp
    open_price: float        # open of first bar at 17:00 UTC
    close_price: float       # close of last bar before London open
    high: float              # session high
    low: float               # session low
    net_pct: float           # (close - open) / open  (signed)
    candle_count: int
    valid: bool
    block_reason: str = ""


@dataclass
class ALSignal:
    """Fully specified reversal signal, ready for trade construction."""
    direction: int               # +1 LONG (Asian fell), -1 SHORT (Asian rose)
    asian: AsianSessionInfo
    tp_level: float              # 50% retrace = (asian.open + asian.close) / 2
    stop_level: float            # Asian extreme + buffer (raw, pre-spread)
    london_open_ts: pd.Timestamp


def compute_asian_session(
    df: pd.DataFrame,
    session_start: pd.Timestamp,
    params: ALParams,
) -> AsianSessionInfo:
    """
    Extract and summarise the Asian session for a given 17:00 UTC session start.

    Captures all 15-min candles in [session_start, session_start + london_open_hours).
    """
    london_open_ts = session_start + pd.Timedelta(hours=params.london_open_hours)
    asian_candles = df[
        (df.index >= session_start) & (df.index < london_open_ts)
    ]

    if len(asian_candles) < params.min_asian_candles:
        return AsianSessionInfo(
            session_start=session_start,
            open_price=0.0, close_price=0.0, high=0.0, low=0.0,
            net_pct=0.0, candle_count=len(asian_candles),
            valid=False,
            block_reason=(
                f"Insufficient Asian candles: {len(asian_candles)} < {params.min_asian_candles}"
            ),
        )

    open_price  = float(asian_candles.iloc[0]["open"])
    close_price = float(asian_candles.iloc[-1]["close"])
    high        = float(asian_candles["high"].max())
    low         = float(asian_candles["low"].min())

    if open_price <= 0.0:
        return AsianSessionInfo(
            session_start=session_start,
            open_price=open_price, close_price=close_price,
            high=high, low=low, net_pct=0.0,
            candle_count=len(asian_candles),
            valid=False, block_reason="Asian open price is zero or negative",
        )

    net_pct = (close_price - open_price) / open_price

    logger.debug(
        "[AL SIGNAL] Asian session %s | Open=%.2f Close=%.2f High=%.2f Low=%.2f "
        "Net=%.4f%% | n=%d",
        session_start.strftime("%Y-%m-%d"),
        open_price, close_price, high, low, net_pct * 100, len(asian_candles),
    )

    return AsianSessionInfo(
        session_start=session_start,
        open_price=open_price, close_price=close_price,
        high=high, low=low, net_pct=net_pct,
        candle_count=len(asian_candles),
        valid=True,
    )


def detect_al_signal(
    asian: AsianSessionInfo,
    params: ALParams,
) -> Optional[ALSignal]:
    """
    Generate a reversal signal if the Asian session moved enough.

    Returns None if the move is below the threshold or the session is invalid.
    """
    if not asian.valid:
        logger.debug("[AL SIGNAL] Skip: invalid session — %s", asian.block_reason)
        return None

    abs_move = abs(asian.net_pct)
    if abs_move < params.min_asian_move_pct:
        logger.debug(
            "[AL SIGNAL] Skip %s: move=%.4f%% < threshold=%.4f%%",
            asian.session_start.date(), abs_move * 100, params.min_asian_move_pct * 100,
        )
        return None

    # Direction: fade the Asian move
    direction = -1 if asian.net_pct > 0 else 1   # Asian up → SHORT; Asian down → LONG

    # TP: 50% retrace of net move (midpoint of open and close)
    tp_level = (asian.open_price + asian.close_price) / 2.0

    # Stop: beyond the Asian session extreme in the direction of the original move
    stop_buffer = params.stop_buffer_pct * asian.close_price
    if direction == 1:   # LONG — Asian fell — stop below Asian low
        stop_level = asian.low - stop_buffer
    else:                # SHORT — Asian rose — stop above Asian high
        stop_level = asian.high + stop_buffer

    london_open_ts = asian.session_start + pd.Timedelta(hours=params.london_open_hours)

    logger.info(
        "[AL SIGNAL] %s | Asian move=%.4f%% | Dir=%s | "
        "Open=%.2f Close=%.2f | TP=%.2f Stop=%.2f | Extreme=%s=%.2f",
        asian.session_start.strftime("%Y-%m-%d"),
        asian.net_pct * 100,
        "LONG" if direction == 1 else "SHORT",
        asian.open_price, asian.close_price,
        tp_level, stop_level,
        ("Low" if direction == 1 else "High"),
        (asian.low if direction == 1 else asian.high),
    )

    return ALSignal(
        direction=direction,
        asian=asian,
        tp_level=tp_level,
        stop_level=stop_level,
        london_open_ts=london_open_ts,
    )
