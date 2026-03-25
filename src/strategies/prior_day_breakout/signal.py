"""
Signal detection for the Prior Day High/Low Breakout strategy.

Hypothesis:
  XAUUSD's prior day high and low act as structural resistance and support.
  A 15-minute candle that closes beyond the prior day's high or low — with
  the open still inside the range (close-confirmed, no gap) — signals that
  the level has been consumed and the move is likely to extend.

Prior day range (PDR):
  PD_HIGH = max(high) over all candles in [00:00, 24:00) UTC of the prior day
  PD_LOW  = min(low)  over all candles in [00:00, 24:00) UTC of the prior day

Signal:
  LONG : candle.close > PD_HIGH  AND  candle.open <= PD_HIGH
  SHORT: candle.close < PD_LOW   AND  candle.open >= PD_LOW

  The open condition prevents entries on gap-through moves where the level was
  never tested — those represent a different (and unreliable) market behaviour.

Only one signal per day.  The first qualifying candle within the signal window
is used.

Stop (failed-breakout invalidation):
  Long : PD_HIGH - stop_buffer  — if price returns below the prior high,
                                   the breakout failed.
  Short: PD_LOW  + stop_buffer  — if price returns above the prior low,
                                   the breakdown failed.

  This produces a tight stop relative to the typical prior-day range.
  The trade is risking only the "false alarm" move back inside the range,
  not the full prior-day range.

Take-profit: entry ± stop_distance * tp_r_multiplier.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..asian_range_breakout.execution import TradeResult, TradeSetup, simulate_trade
from ..asian_range_breakout.signal import AsianRange
from .params import PDBreakoutParams

logger = logging.getLogger(__name__)


@dataclass
class PriorDayRange:
    """
    The prior UTC calendar day's high and low.

    valid=False if the prior day is unavailable, has too few candles,
    or the range fails the min/max filter.
    """
    high: float
    low: float
    range_price: float
    range_pct: float
    candle_count: int
    valid: bool
    block_reason: str = ""


@dataclass
class PDBreakoutSignal:
    """
    A close-confirmed breakout of the prior day's high or low.

    direction:          +1 = long (close above PD_HIGH)
                        -1 = short (close below PD_LOW)
    breakout_level:     PD_HIGH (long) or PD_LOW (short).
    signal_candle_ts:   Timestamp of the breakout candle.
    signal_candle_close: Close price of the breakout candle.
    """
    direction: int
    breakout_level: float
    signal_candle_ts: pd.Timestamp
    signal_candle_close: float


def compute_prior_day_range(
    df: pd.DataFrame,
    prior_date: "datetime.date",
    params: PDBreakoutParams,
) -> PriorDayRange:
    """
    Compute PD_HIGH and PD_LOW from all candles in the prior UTC calendar day.

    Returns PriorDayRange with valid=False if:
      - No candles found for the prior day (market closed / data gap).
      - Range is below min_pdr_pct (holiday / abnormally quiet day).
      - Range is above max_pdr_pct (spike day — stop would be too large).
    """
    import datetime
    day_start = pd.Timestamp(prior_date, tz="UTC")
    day_end   = day_start + pd.Timedelta(days=1)

    prior_candles = df[(df.index >= day_start) & (df.index < day_end)]

    if prior_candles.empty:
        logger.info("[PD] No candles for prior day %s. Skipping.", prior_date)
        return PriorDayRange(0, 0, 0, 0, 0, False, "no_prior_day_data")

    pd_high = float(prior_candles["high"].max())
    pd_low  = float(prior_candles["low"].min())
    range_price = pd_high - pd_low
    range_pct   = range_price / pd_high

    if range_pct < params.min_pdr_pct:
        logger.info(
            "[PD] %s range too tight: %.4f%% < %.4f%% min. Skipping.",
            prior_date, range_pct * 100, params.min_pdr_pct * 100,
        )
        return PriorDayRange(
            pd_high, pd_low, range_price, range_pct,
            len(prior_candles), False, "range_too_tight",
        )

    if range_pct > params.max_pdr_pct:
        logger.info(
            "[PD] %s range too wide: %.4f%% > %.4f%% max. Skipping.",
            prior_date, range_pct * 100, params.max_pdr_pct * 100,
        )
        return PriorDayRange(
            pd_high, pd_low, range_price, range_pct,
            len(prior_candles), False, "range_too_wide",
        )

    logger.info(
        "[PD] Prior day %s | HIGH=%.2f LOW=%.2f Range=%.2f (%.4f%%)",
        prior_date, pd_high, pd_low, range_price, range_pct * 100,
    )
    return PriorDayRange(
        high=pd_high,
        low=pd_low,
        range_price=range_price,
        range_pct=range_pct,
        candle_count=len(prior_candles),
        valid=True,
    )


def detect_pd_breakout(
    window_candles: pd.DataFrame,
    pdr: PriorDayRange,
) -> Optional[PDBreakoutSignal]:
    """
    Scan signal-window candles for the first close-confirmed PD breakout.

    LONG : close > PD_HIGH  AND  open <= PD_HIGH
    SHORT: close < PD_LOW   AND  open >= PD_LOW

    Returns the first qualifying candle or None.
    """
    for ts, candle in window_candles.iterrows():
        o = float(candle["open"])
        c = float(candle["close"])

        if c > pdr.high and o <= pdr.high:
            logger.info(
                "[PD SIGNAL] LONG | %s | Close=%.2f > PD_HIGH=%.2f | Open=%.2f",
                ts, c, pdr.high, o,
            )
            return PDBreakoutSignal(
                direction=1,
                breakout_level=pdr.high,
                signal_candle_ts=ts,
                signal_candle_close=c,
            )

        if c < pdr.low and o >= pdr.low:
            logger.info(
                "[PD SIGNAL] SHORT | %s | Close=%.2f < PD_LOW=%.2f | Open=%.2f",
                ts, c, pdr.low, o,
            )
            return PDBreakoutSignal(
                direction=-1,
                breakout_level=pdr.low,
                signal_candle_ts=ts,
                signal_candle_close=c,
            )

    logger.info("[PD SIGNAL] No PD breakout in signal window.")
    return None


def compute_pd_trade_setup(
    signal: PDBreakoutSignal,
    entry_candle: pd.Series,
    entry_timestamp: pd.Timestamp,
    pdr: PriorDayRange,
    equity: float,
    params: PDBreakoutParams,
) -> Optional[TradeSetup]:
    """
    Build a TradeSetup for the PD breakout entry.

    Entry: open of the candle after the signal candle (next-candle-open fill).

    Entry validation:
      If entry gaps back through the breakout level (reversal before we even enter),
      skip the trade.
        Long : entry_open < pdr.high  (price already retreated below PD_HIGH)
        Short: entry_open > pdr.low   (price already retreated above PD_LOW)

    Stop (failed breakout invalidation):
      Long : pdr.high - stop_buffer  — just below the level that was broken
      Short: pdr.low  + stop_buffer  — just above the level that was broken

    TP: fixed R from entry (entry ± stop_distance * tp_r_multiplier).
    """
    d = signal.direction
    half_spread = params.spread_price / 2.0
    gross_entry = float(entry_candle["open"])

    # ── Entry validation: gap-back through breakout level ─────────────────────
    if d == 1 and gross_entry < signal.breakout_level:
        logger.info(
            "[PD EXEC] SKIP long: entry open %.2f < PD_HIGH %.2f (gap-back before entry).",
            gross_entry, signal.breakout_level,
        )
        return None
    if d == -1 and gross_entry > signal.breakout_level:
        logger.info(
            "[PD EXEC] SKIP short: entry open %.2f > PD_LOW %.2f (gap-back before entry).",
            gross_entry, signal.breakout_level,
        )
        return None

    # ── Effective entry ────────────────────────────────────────────────────────
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    # ── Stop: just behind the breakout level ──────────────────────────────────
    stop_buffer = max(
        1.5 * params.spread_price,
        params.stop_buffer_floor_pct * abs(effective_entry),
    )
    # Long : sl_gross = pdr.high - stop_buffer
    # Short: sl_gross = pdr.low  + stop_buffer
    sl_gross = signal.breakout_level + d * (-1) * stop_buffer
    # d=+1 long : sl = breakout_level + (-1)*buf = level - buf   (below PD_HIGH)
    # d=-1 short: sl = breakout_level + (+1)*buf = level + buf   (above PD_LOW)
    effective_sl = sl_gross - d * half_spread

    stop_distance = abs(effective_entry - effective_sl)
    if stop_distance <= 0.0:
        logger.error(
            "[PD EXEC] Zero stop distance: entry=%.2f sl=%.2f. Skipping.",
            effective_entry, effective_sl,
        )
        return None

    # ── TP: fixed R ────────────────────────────────────────────────────────────
    tp_distance = stop_distance * params.tp_r_multiplier
    effective_tp = effective_entry + d * tp_distance
    tp_gross = effective_tp + d * half_spread

    # ── Position sizing ────────────────────────────────────────────────────────
    risk_amount   = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    logger.info(
        "[PD EXEC] Setup | Dir=%s | GrossEntry=%.2f EffEntry=%.2f | "
        "PD=[%.2f, %.2f] SL=%.2f TP=%.2f | "
        "StopDist=%.3f TP_R=%.1f | Size=%.4foz Risk=$%.2f",
        "LONG" if d == 1 else "SHORT",
        gross_entry, effective_entry,
        pdr.low, pdr.high, sl_gross, tp_gross,
        stop_distance, params.tp_r_multiplier,
        position_size, risk_amount,
    )

    # Wrap PriorDayRange into AsianRange to reuse simulate_trade unchanged
    pdr_as_range = AsianRange(
        high=pdr.high,
        low=pdr.low,
        range_price=pdr.range_price,
        range_pct=pdr.range_pct,
        reference_price=pdr.high,
        candle_count=pdr.candle_count,
        valid=pdr.valid,
        block_reason=pdr.block_reason,
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
        asian_range=pdr_as_range,
    )
