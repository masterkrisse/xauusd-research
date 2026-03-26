"""
Signal logic for the London Opening Range Breakout strategy.

Step 1 — Compute LOR:
  Scan candles in [session_start + lor_start_hours, session_start + lor_end_hours).
  LOR_H = max(candle.high)  LOR_L = min(candle.low)
  Validate: LOR_size in [min_lor_pct, max_lor_pct] * midprice.

Step 2 — Detect breakout:
  Scan signal window [session_start + signal_start_hours, session_start + signal_end_hours).
  LONG signal:  first candle with close > LOR_H
  SHORT signal: first candle with close < LOR_L
  First trigger wins. One signal per session.

No lookahead: LOR is computed from completed [07:00, 08:00) candles.
Signal uses candle.close (confirmed, no lookahead within candle).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .params import LBParams

logger = logging.getLogger(__name__)


@dataclass
class LBRange:
    """London Opening Range (07:00–08:00 UTC), fully computed."""
    high:      float
    low:       float
    midprice:  float
    size:      float    # LOR_H - LOR_L  (USD)
    size_pct:  float    # size / midprice
    valid:     bool
    n_candles: int


@dataclass
class LBSignal:
    """First close-outside-LOR event in the signal window."""
    direction:       int            # +1 LONG, -1 SHORT
    signal_candle_ts: pd.Timestamp  # timestamp of the breakout candle
    lor_high:        float
    lor_low:         float
    lor_size:        float
    breakout_level:  float          # LOR_H (LONG) or LOR_L (SHORT)
    tp_gross:        float          # structural TP: breakout_level ± size * multiplier


def compute_lor(
    df:            pd.DataFrame,
    session_start: pd.Timestamp,
    params:        LBParams,
) -> LBRange:
    """
    Compute the London Opening Range for a given session.
    """
    lor_start = session_start + pd.Timedelta(hours=params.lor_start_hours)
    lor_end   = session_start + pd.Timedelta(hours=params.lor_end_hours)

    lor_candles = df[(df.index >= lor_start) & (df.index < lor_end)]

    if len(lor_candles) < params.min_lor_candles:
        logger.debug(
            "[LOR] %s | Insufficient candles: %d < %d",
            session_start.strftime("%Y-%m-%d"), len(lor_candles), params.min_lor_candles,
        )
        return LBRange(0.0, 0.0, 0.0, 0.0, 0.0, False, len(lor_candles))

    lor_h    = float(lor_candles["high"].max())
    lor_l    = float(lor_candles["low"].min())
    mid      = (lor_h + lor_l) / 2.0
    lor_size = lor_h - lor_l
    size_pct = lor_size / mid if mid > 0 else 0.0

    if size_pct < params.min_lor_pct:
        logger.info(
            "[LOR] %s | Range too tight: %.4f%% < %.4f%%",
            session_start.strftime("%Y-%m-%d"), size_pct * 100, params.min_lor_pct * 100,
        )
        return LBRange(lor_h, lor_l, mid, lor_size, size_pct, False, len(lor_candles))

    if size_pct > params.max_lor_pct:
        logger.info(
            "[LOR] %s | Range too wide: %.4f%% > %.4f%%",
            session_start.strftime("%Y-%m-%d"), size_pct * 100, params.max_lor_pct * 100,
        )
        return LBRange(lor_h, lor_l, mid, lor_size, size_pct, False, len(lor_candles))

    logger.debug(
        "[LOR] %s | H=%.2f  L=%.2f  Size=%.2f (%.4f%%)",
        session_start.strftime("%Y-%m-%d"), lor_h, lor_l, lor_size, size_pct * 100,
    )
    return LBRange(lor_h, lor_l, mid, lor_size, size_pct, True, len(lor_candles))


def detect_lb_signal(
    window_candles: pd.DataFrame,
    lor:            LBRange,
    params:         LBParams,
) -> Optional[LBSignal]:
    """
    Scan the signal window for the first close outside the LOR.

    Returns the first LONG or SHORT signal found, or None.
    """
    for ts, candle in window_candles.iterrows():
        c = float(candle["close"])

        if c > lor.high:
            tp_gross = lor.high + lor.size * params.tp_lor_multiplier
            logger.info(
                "[LB SIGNAL] LONG | %s | Close=%.2f > LOR_H=%.2f | TP_target=%.2f",
                ts, c, lor.high, tp_gross,
            )
            return LBSignal(
                direction=1,
                signal_candle_ts=ts,
                lor_high=lor.high,
                lor_low=lor.low,
                lor_size=lor.size,
                breakout_level=lor.high,
                tp_gross=tp_gross,
            )

        if c < lor.low:
            tp_gross = lor.low - lor.size * params.tp_lor_multiplier
            logger.info(
                "[LB SIGNAL] SHORT | %s | Close=%.2f < LOR_L=%.2f | TP_target=%.2f",
                ts, c, lor.low, tp_gross,
            )
            return LBSignal(
                direction=-1,
                signal_candle_ts=ts,
                lor_high=lor.high,
                lor_low=lor.low,
                lor_size=lor.size,
                breakout_level=lor.low,
                tp_gross=tp_gross,
            )

    logger.debug("[LB SIGNAL] No breakout in signal window.")
    return None
