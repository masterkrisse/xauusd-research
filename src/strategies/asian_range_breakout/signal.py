"""
Signal detection layer: Asian range calculation and breakout detection.

Responsibilities:
  1. Compute the Asian session range from raw candle data.
  2. Validate the range against filter criteria.
  3. Scan the London window for the first qualifying breakout candle.

This module has no side effects on trade state. It only reads data and
returns typed result objects.

Breakout candle qualification rules (exact):
  Long:  candle_close > ASIA_HIGH  AND  candle_open <= ASIA_HIGH
  Short: candle_close < ASIA_LOW   AND  candle_open >= ASIA_LOW

The open condition rejects candles that gapped entirely outside the range
before the London session interacted with the boundary.  A candle that opens
exactly at the boundary (open == ASIA_HIGH for long) is treated as "inside"
and qualifies.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .params import StrategyParams
from .session import SessionBoundaries

logger = logging.getLogger(__name__)


@dataclass
class AsianRange:
    """
    Computed Asian session range for one trading day.

    Fields:
        high:            Highest high of all Asian session candles.
        low:             Lowest low of all Asian session candles.
        range_price:     high - low in USD.
        range_pct:       range_price / reference_price.
        reference_price: Close of the last Asian session candle (used for pct calc).
        candle_count:    Number of candles used to build the range.
        valid:           True if the range passed all filters.
        block_reason:    Human-readable reason for rejection, or None if valid.
    """
    high: float
    low: float
    range_price: float
    range_pct: float
    reference_price: float
    candle_count: int
    valid: bool
    block_reason: Optional[str]


@dataclass
class BreakoutSignal:
    """
    First qualifying breakout candle detected in the London window.

    Fields:
        direction:               +1 for long, -1 for short.
        breakout_level:          The Asian range boundary that was breached.
                                 ASIA_HIGH for long, ASIA_LOW for short.
        signal_candle_ts:        Open timestamp of the signal candle.
        signal_candle_open:      Open price of the signal candle.
        signal_candle_close:     Close price of the signal candle.
    """
    direction: int
    breakout_level: float
    signal_candle_ts: pd.Timestamp
    signal_candle_open: float
    signal_candle_close: float


def compute_asian_range(
    day_candles: pd.DataFrame,
    session: SessionBoundaries,
    params: StrategyParams,
) -> AsianRange:
    """
    Compute the Asian session range for one trading day.

    Includes all candles whose open timestamp satisfies:
        session.asia_start <= candle_open < session.london_open

    The candle that opens at (london_open - candle_minutes) closes at london_open
    and is included as the last Asian candle.  The first London candle opens at
    london_open and is excluded from the range.

    Returns an AsianRange with valid=False (and a block_reason) if:
      - No Asian candles are found.
      - The reference price is non-positive.
      - range_pct < params.min_range_pct  (too tight).
      - range_pct > params.max_range_pct  (too wide).
    """
    # Candles with open in [asia_start, london_open)
    asian = day_candles[
        (day_candles.index >= session.asia_start)
        & (day_candles.index < session.london_open)
    ]

    date_tag = str(session.trading_date)

    # ── Guard: must have candles ──────────────────────────────────────────────
    if asian.empty:
        reason = "No Asian session candles found in data."
        logger.warning("[SIGNAL] %s | BLOCKED | %s", date_tag, reason)
        return AsianRange(0.0, 0.0, 0.0, 0.0, 0.0, 0, False, reason)

    high = float(asian["high"].max())
    low = float(asian["low"].min())
    range_price = high - low
    reference_price = float(asian["close"].iloc[-1])
    candle_count = len(asian)

    # ── Guard: non-positive reference price ──────────────────────────────────
    if reference_price <= 0.0:
        reason = f"Reference price non-positive: {reference_price:.5f}"
        logger.error("[SIGNAL] %s | BLOCKED | %s", date_tag, reason)
        return AsianRange(
            high, low, range_price, 0.0, reference_price, candle_count, False, reason
        )

    range_pct = range_price / reference_price

    # ── Range too tight ───────────────────────────────────────────────────────
    if range_pct < params.min_range_pct:
        reason = (
            f"Range too tight: {range_pct:.4%} < min {params.min_range_pct:.4%} "
            f"(${range_price:.2f} on ${reference_price:.2f})"
        )
        logger.info("[SIGNAL] %s | BLOCKED | %s", date_tag, reason)
        return AsianRange(
            high, low, range_price, range_pct, reference_price, candle_count, False, reason
        )

    # ── Range too wide ────────────────────────────────────────────────────────
    if range_pct > params.max_range_pct:
        reason = (
            f"Range too wide: {range_pct:.4%} > max {params.max_range_pct:.4%} "
            f"(${range_price:.2f} on ${reference_price:.2f})"
        )
        logger.info("[SIGNAL] %s | BLOCKED | %s", date_tag, reason)
        return AsianRange(
            high, low, range_price, range_pct, reference_price, candle_count, False, reason
        )

    logger.info(
        "[SIGNAL] %s | Range VALID | H=%.2f  L=%.2f  Range=%.2f (%.4f%%) | "
        "Candles=%d | BST=%s",
        date_tag, high, low, range_price, range_pct * 100,
        candle_count, session.is_bst,
    )

    return AsianRange(
        high=high,
        low=low,
        range_price=range_price,
        range_pct=range_pct,
        reference_price=reference_price,
        candle_count=candle_count,
        valid=True,
        block_reason=None,
    )


def detect_breakout(
    window_candles: pd.DataFrame,
    asian_range: AsianRange,
) -> Optional[BreakoutSignal]:
    """
    Scan London window candles for the first qualifying breakout.

    Iterates candles in chronological order.  Returns on the first match.
    If no qualifying candle is found before the window closes, returns None.

    Qualification:
      Long:  candle.close > asian_range.high  AND  candle.open <= asian_range.high
      Short: candle.close < asian_range.low   AND  candle.open >= asian_range.low

    The open condition means:
      - Candles that gap entirely above ASIA_HIGH do NOT qualify as long signals.
      - Candles that gap entirely below ASIA_LOW  do NOT qualify as short signals.
      - A candle opening exactly at the boundary DOES qualify (open == boundary).
    """
    for ts, candle in window_candles.iterrows():
        o = float(candle["open"])
        c = float(candle["close"])

        # ── Long breakout ─────────────────────────────────────────────────────
        if c > asian_range.high and o <= asian_range.high:
            logger.info(
                "[SIGNAL] Breakout LONG | Candle=%s | Open=%.2f  Close=%.2f | "
                "Level=%.2f (ASIA_HIGH)",
                ts, o, c, asian_range.high,
            )
            return BreakoutSignal(
                direction=1,
                breakout_level=asian_range.high,
                signal_candle_ts=ts,
                signal_candle_open=o,
                signal_candle_close=c,
            )

        # ── Short breakout ────────────────────────────────────────────────────
        if c < asian_range.low and o >= asian_range.low:
            logger.info(
                "[SIGNAL] Breakout SHORT | Candle=%s | Open=%.2f  Close=%.2f | "
                "Level=%.2f (ASIA_LOW)",
                ts, o, c, asian_range.low,
            )
            return BreakoutSignal(
                direction=-1,
                breakout_level=asian_range.low,
                signal_candle_ts=ts,
                signal_candle_open=o,
                signal_candle_close=c,
            )

    logger.info("[SIGNAL] No qualifying breakout in London window.")
    return None
