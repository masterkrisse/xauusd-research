"""
Fade signal detection for the London False Breakout strategy.

Hypothesis:
  London session frequently produces false breakouts of the Asian range.
  Price wicks beyond ASIA_HIGH or ASIA_LOW within the London window, then
  closes back inside the range, signalling rejection of the breakout level.
  We fade that rejection back toward the range midpoint.

Signal definition (wick rejection, single candle):
  SHORT FADE (fade upside false break):
    candle.high  > ASIA_HIGH                   (wick pierces above range)
    candle.close < ASIA_HIGH                   (closes back inside range)
    overshoot    >= min_overshoot_pct           (wick is meaningful, not 1 tick)

  LONG FADE (fade downside false break):
    candle.low   < ASIA_LOW                    (wick pierces below range)
    candle.close > ASIA_LOW                    (closes back inside range)
    overshoot    >= min_overshoot_pct

Entry: open of the candle FOLLOWING the signal candle (next-candle-open fill).
  This is rejected if the entry candle gaps through the rejection level
  (i.e. the gap invalidates the reversal premise).

Stop: beyond the breakout wick extreme.
  Short fade: candle.high + stop_buffer
  Long  fade: candle.low  - stop_buffer

Take-profit: range midpoint = (ASIA_HIGH + ASIA_LOW) / 2
  This is structural, not parametric.  If the false break hypothesis is correct,
  price should at minimum return to the centre of the range.

Comparison note: the opposite range boundary (ASIA_LOW for short, ASIA_HIGH for long)
is the more aggressive target and may produce better R:R at lower win rate.
That is not tested here.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..asian_range_breakout.execution import (
    EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP, TradeResult, TradeSetup,
    _build_result, simulate_trade,
)
from ..asian_range_breakout.signal import AsianRange
from .params import FadeParams

logger = logging.getLogger(__name__)


@dataclass
class FadeSignal:
    """
    A wick rejection signal detected in the London window.

    direction:         +1 = long fade (price wicked below ASIA_LOW and closed inside)
                       -1 = short fade (price wicked above ASIA_HIGH and closed inside)
    breakout_extreme:  The wick tip: candle.high for short, candle.low for long.
                       Stop is placed beyond this level.
    rejection_level:   The range boundary that was pierced: ASIA_HIGH (short) or ASIA_LOW (long).
    overshoot_pct:     How far the wick went beyond the boundary as a fraction of boundary price.
    signal_candle_ts:  Timestamp of the rejection candle.
    signal_candle_close: Close price of the rejection candle (approximate entry reference).
    """
    direction: int
    breakout_extreme: float
    rejection_level: float
    overshoot_pct: float
    signal_candle_ts: pd.Timestamp
    signal_candle_close: float


def detect_fade_signal(
    window_candles: pd.DataFrame,
    asian_range: AsianRange,
    min_overshoot_pct: float,
) -> Optional[FadeSignal]:
    """
    Scan London window candles for the first wick rejection signal.

    Iterates in chronological order; returns the first qualifying candle.
    Returns None if no rejection occurs before the window closes.

    A candle qualifies as:
      SHORT FADE: high > ASIA_HIGH  AND  close < ASIA_HIGH  AND  overshoot >= min
      LONG  FADE: low  < ASIA_LOW   AND  close > ASIA_LOW   AND  overshoot >= min

    No open-price condition is applied: gap opens that immediately reverse also qualify.
    """
    for ts, candle in window_candles.iterrows():
        h  = float(candle["high"])
        lo = float(candle["low"])
        c  = float(candle["close"])

        # ── Short fade: wick above ASIA_HIGH, close back inside ───────────────
        if h > asian_range.high and c < asian_range.high:
            overshoot = (h - asian_range.high) / asian_range.high
            if overshoot >= min_overshoot_pct:
                logger.info(
                    "[FADE SIGNAL] SHORT | Candle=%s | High=%.2f > ASIA_HIGH=%.2f | "
                    "Close=%.2f (inside) | Overshoot=%.4f%%",
                    ts, h, asian_range.high, c, overshoot * 100,
                )
                return FadeSignal(
                    direction=-1,
                    breakout_extreme=h,
                    rejection_level=asian_range.high,
                    overshoot_pct=overshoot,
                    signal_candle_ts=ts,
                    signal_candle_close=c,
                )

        # ── Long fade: wick below ASIA_LOW, close back inside ────────────────
        if lo < asian_range.low and c > asian_range.low:
            overshoot = (asian_range.low - lo) / asian_range.low
            if overshoot >= min_overshoot_pct:
                logger.info(
                    "[FADE SIGNAL] LONG | Candle=%s | Low=%.2f < ASIA_LOW=%.2f | "
                    "Close=%.2f (inside) | Overshoot=%.4f%%",
                    ts, lo, asian_range.low, c, overshoot * 100,
                )
                return FadeSignal(
                    direction=1,
                    breakout_extreme=lo,
                    rejection_level=asian_range.low,
                    overshoot_pct=overshoot,
                    signal_candle_ts=ts,
                    signal_candle_close=c,
                )

    logger.info("[FADE SIGNAL] No wick rejection in London window.")
    return None


def compute_fade_trade_setup(
    signal: FadeSignal,
    entry_candle: pd.Series,
    entry_timestamp: pd.Timestamp,
    asian_range: AsianRange,
    equity: float,
    params: FadeParams,
) -> Optional[TradeSetup]:
    """
    Build a TradeSetup for the fade entry.

    Entry: open of the candle after the signal candle.

    Entry validation:
      If the entry candle gaps through the rejection level (invalidating the reversal),
      the trade is rejected.
        Short fade: entry_open > rejection_level (gap up past ASIA_HIGH = reversal failed)
        Long  fade: entry_open < rejection_level (gap down past ASIA_LOW = reversal failed)

    Stop: wick extreme ± stop_buffer (beyond the rejected breakout level).
    TP:   range midpoint — structural target, not R-based.

    The realized R on a TP exit will vary by trade depending on how far the wick
    went and where price closed.  Log R:R explicitly so it can be reviewed.
    """
    d = signal.direction
    half_spread = params.spread_price / 2.0

    gross_entry = float(entry_candle["open"])

    # ── Entry validation: gap-through rejection ───────────────────────────────
    if d == -1 and gross_entry > signal.rejection_level:
        logger.info(
            "[FADE EXEC] SKIP short fade: entry open %.2f > rejection level %.2f "
            "(gap invalidates reversal).",
            gross_entry, signal.rejection_level,
        )
        return None
    if d == 1 and gross_entry < signal.rejection_level:
        logger.info(
            "[FADE EXEC] SKIP long fade: entry open %.2f < rejection level %.2f "
            "(gap invalidates reversal).",
            gross_entry, signal.rejection_level,
        )
        return None

    # ── Effective entry ───────────────────────────────────────────────────────
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    # ── Stop: beyond wick extreme ─────────────────────────────────────────────
    stop_buffer = max(
        1.5 * params.spread_price,
        params.stop_buffer_floor_pct * abs(effective_entry),
    )
    # Short fade: stop above the wick high (+buffer); long fade: below wick low (-buffer)
    sl_gross = signal.breakout_extreme + d * (-1) * stop_buffer
    # i.e. short (d=-1): sl_gross = extreme + (+1)*buffer  = extreme + buffer
    #      long  (d=+1): sl_gross = extreme + (-1)*buffer  = extreme - buffer
    effective_sl = sl_gross - d * half_spread

    stop_distance = abs(effective_entry - effective_sl)
    if stop_distance <= 0.0:
        logger.error(
            "[FADE EXEC] Zero stop distance: entry=%.2f sl=%.2f. Skipping.",
            effective_entry, effective_sl,
        )
        return None

    # ── TP: range midpoint (structural) ──────────────────────────────────────
    midpoint = (asian_range.high + asian_range.low) / 2.0
    effective_tp = midpoint
    # tp_gross: the candle level that must be reached to net effective_tp after exit spread
    tp_gross = effective_tp + d * half_spread
    # Short (d=-1): tp_gross = midpoint - half_spread (price must fall to this level)
    # Long  (d=+1): tp_gross = midpoint + half_spread (price must rise to this level)

    # ── Position sizing ───────────────────────────────────────────────────────
    risk_amount = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    # ── Log R:R explicitly (variable for fade, unlike fixed TP in breakout) ───
    tp_distance = abs(effective_entry - effective_tp)
    rr = tp_distance / stop_distance if stop_distance > 0.0 else 0.0

    logger.info(
        "[FADE EXEC] Setup | Dir=%s | GrossEntry=%.2f EffEntry=%.2f | "
        "WickExtreme=%.2f SL=%.2f TP=%.2f(midpoint) | "
        "StopDist=%.3f TPDist=%.3f R:R=%.2f | "
        "Size=%.4foz Risk=$%.2f",
        "SHORT" if d == -1 else "LONG",
        gross_entry, effective_entry,
        signal.breakout_extreme, sl_gross, tp_gross,
        stop_distance, tp_distance, rr,
        position_size, risk_amount,
    )

    return TradeSetup(
        direction=d,
        entry_price=effective_entry,
        sl_gross=sl_gross,
        tp_gross=tp_gross,
        effective_sl=effective_sl,
        effective_tp=effective_tp,
        stop_distance=stop_distance,
        position_size=position_size,
        risk_amount=risk_amount,
        entry_timestamp=entry_timestamp,
        signal_timestamp=signal.signal_candle_ts,
        asian_range=asian_range,
    )
