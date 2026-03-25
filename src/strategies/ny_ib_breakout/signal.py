"""
Signal detection for the NY Morning Initial Balance Breakout strategy.

Hypothesis:
  The first 30 minutes of NY trading (09:30-10:00 ET) establishes an initial
  balance (IB).  A close beyond the IB high or IB low during the next 2.5 hours
  signals a directional move driven by institutional positioning and US macro
  data reactions.  We enter in the direction of the breakout.

Initial balance definition:
  IB_HIGH = max(high) of all candles with timestamp in [ny_open, ib_close)
  IB_LOW  = min(low)  of all candles with timestamp in [ny_open, ib_close)

Signal (close-confirmed breakout):
  LONG : candle.close > IB_HIGH  AND  candle.open <= IB_HIGH
  SHORT: candle.close < IB_LOW   AND  candle.open >= IB_LOW

  Open condition ensures we do not enter a candle that gapped through the level
  (gap already gone — no edge).

Entry: open of the candle FOLLOWING the signal candle.
Stop: opposite IB boundary ± stop_buffer (structural level, not arbitrary).
  Long  stop: IB_LOW  - stop_buffer
  Short stop: IB_HIGH + stop_buffer
Take-profit: entry ± stop_distance * tp_r_multiplier (fixed R, parametric).

Key difference from the Asian range breakout:
  Stop is at the IB opposite boundary, not an arbitrary buffer beyond a far level.
  This produces tighter, more consistent stop distances relative to the trade size.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..asian_range_breakout.execution import (
    EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP, TradeResult, TradeSetup,
    _build_result, simulate_trade,
)
from .params import NYIBParams
from .session import NYSessionBoundaries

logger = logging.getLogger(__name__)


@dataclass
class InitialBalance:
    """
    The NY session initial balance range for one trading day.

    high, low       : IB high and low prices.
    range_price     : high - low in USD.
    range_pct       : range_price / reference_price.
    reference_price : IB high (used as denominator for pct calculations).
    candle_count    : Number of candles included in the IB.
    valid           : False if the IB is blocked by a filter.
    block_reason    : Human-readable reason for an invalid IB.
    """
    high: float
    low: float
    range_price: float
    range_pct: float
    reference_price: float
    candle_count: int
    valid: bool
    block_reason: str = ""


@dataclass
class IBBreakoutSignal:
    """
    A close-confirmed breakout of the NY initial balance.

    direction:          +1 = long (close above IB_HIGH)
                        -1 = short (close below IB_LOW)
    breakout_level:     IB_HIGH (long) or IB_LOW (short).
    signal_candle_ts:   Timestamp of the breakout candle.
    signal_candle_close: Close price of the breakout candle.
    """
    direction: int
    breakout_level: float
    signal_candle_ts: pd.Timestamp
    signal_candle_close: float


def compute_ib_range(
    day_candles: pd.DataFrame,
    session: NYSessionBoundaries,
    params: NYIBParams,
) -> InitialBalance:
    """
    Compute the NY session initial balance from candles in [ny_open, ib_close).

    Returns an InitialBalance with valid=False if:
      - No candles found in the IB window.
      - Fewer than expected candles (partial IB — holiday or data gap).
      - IB range is below min_ib_range_pct (too tight — noise).
      - IB range is above max_ib_range_pct (too wide — news spike / unfillable).
    """
    ib_candles = day_candles[
        (day_candles.index >= session.ny_open)
        & (day_candles.index < session.ib_close)
    ]

    expected_candles = params.ib_duration_minutes // params.candle_minutes

    if ib_candles.empty:
        logger.info(
            "[NY IB] No candles in IB window [%s, %s). Skipping.",
            session.ny_open.strftime("%H:%M UTC"),
            session.ib_close.strftime("%H:%M UTC"),
        )
        return InitialBalance(0, 0, 0, 0, 1, 0, False, "no_candles_in_ib")

    if len(ib_candles) < expected_candles:
        logger.info(
            "[NY IB] Partial IB: %d of %d expected candles. Skipping.",
            len(ib_candles), expected_candles,
        )
        return InitialBalance(0, 0, 0, 0, 1, len(ib_candles), False, "partial_ib")

    ib_high = float(ib_candles["high"].max())
    ib_low  = float(ib_candles["low"].min())
    range_price = ib_high - ib_low
    range_pct   = range_price / ib_high

    if range_pct < params.min_ib_range_pct:
        logger.info(
            "[NY IB] Range too tight: %.4f%% < %.4f%% min. Skipping.",
            range_pct * 100, params.min_ib_range_pct * 100,
        )
        return InitialBalance(
            ib_high, ib_low, range_price, range_pct, ib_high,
            len(ib_candles), False, "range_too_tight",
        )

    if range_pct > params.max_ib_range_pct:
        logger.info(
            "[NY IB] Range too wide: %.4f%% > %.4f%% max. Skipping.",
            range_pct * 100, params.max_ib_range_pct * 100,
        )
        return InitialBalance(
            ib_high, ib_low, range_price, range_pct, ib_high,
            len(ib_candles), False, "range_too_wide",
        )

    logger.info(
        "[NY IB] IB valid | High=%.2f Low=%.2f Range=%.2f (%.4f%%) | Candles=%d",
        ib_high, ib_low, range_price, range_pct * 100, len(ib_candles),
    )
    return InitialBalance(
        high=ib_high,
        low=ib_low,
        range_price=range_price,
        range_pct=range_pct,
        reference_price=ib_high,
        candle_count=len(ib_candles),
        valid=True,
    )


def detect_ib_breakout(
    window_candles: pd.DataFrame,
    ib: InitialBalance,
) -> Optional[IBBreakoutSignal]:
    """
    Scan signal window candles for the first close-confirmed IB breakout.

    A candle qualifies as:
      LONG : close > IB_HIGH  AND  open <= IB_HIGH  (close-confirmed, no gap)
      SHORT: close < IB_LOW   AND  open >= IB_LOW   (close-confirmed, no gap)

    Returns the first qualifying candle, or None if no breakout occurs.
    """
    for ts, candle in window_candles.iterrows():
        o = float(candle["open"])
        c = float(candle["close"])

        # Long breakout
        if c > ib.high and o <= ib.high:
            logger.info(
                "[NY IB SIGNAL] LONG | Candle=%s | Close=%.2f > IB_HIGH=%.2f | Open=%.2f",
                ts, c, ib.high, o,
            )
            return IBBreakoutSignal(
                direction=1,
                breakout_level=ib.high,
                signal_candle_ts=ts,
                signal_candle_close=c,
            )

        # Short breakout
        if c < ib.low and o >= ib.low:
            logger.info(
                "[NY IB SIGNAL] SHORT | Candle=%s | Close=%.2f < IB_LOW=%.2f | Open=%.2f",
                ts, c, ib.low, o,
            )
            return IBBreakoutSignal(
                direction=-1,
                breakout_level=ib.low,
                signal_candle_ts=ts,
                signal_candle_close=c,
            )

    logger.info("[NY IB SIGNAL] No IB breakout in signal window.")
    return None


def compute_ib_trade_setup(
    signal: IBBreakoutSignal,
    entry_candle: pd.Series,
    entry_timestamp: pd.Timestamp,
    ib: InitialBalance,
    equity: float,
    params: NYIBParams,
) -> Optional[TradeSetup]:
    """
    Build a TradeSetup for the IB breakout entry.

    Entry: open of the candle after the signal candle.

    Entry validation:
      If the entry candle gaps beyond the breakout level in the signal direction,
      the move is already exhausted — skip.
        Long : entry_open > ib.high + 0.5 * ib.range_price  (gapped > halfway across IB)
        Short: entry_open < ib.low  - 0.5 * ib.range_price

    Stop: opposite IB boundary ± stop_buffer.
      Long : ib.low  - stop_buffer
      Short: ib.high + stop_buffer
      This is a structural stop at a known level, not arbitrary distance.

    TP: entry + direction * stop_distance * tp_r_multiplier
    """
    d = signal.direction
    half_spread = params.spread_price / 2.0

    gross_entry = float(entry_candle["open"])

    # ── Entry validation: excessive gap past breakout level ───────────────────
    # If price gapped more than half the IB range beyond the signal level,
    # the move is already well extended; execution risk is too high.
    gap_limit = 0.5 * ib.range_price
    if d == 1 and gross_entry > ib.high + gap_limit:
        logger.info(
            "[NY IB EXEC] SKIP long: entry open %.2f gapped more than %.2f above IB_HIGH %.2f.",
            gross_entry, gap_limit, ib.high,
        )
        return None
    if d == -1 and gross_entry < ib.low - gap_limit:
        logger.info(
            "[NY IB EXEC] SKIP short: entry open %.2f gapped more than %.2f below IB_LOW %.2f.",
            gross_entry, gap_limit, ib.low,
        )
        return None

    # ── Effective entry (spread + slippage) ───────────────────────────────────
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    # ── Stop: opposite IB boundary + buffer ───────────────────────────────────
    stop_buffer = max(
        1.5 * params.spread_price,
        params.stop_buffer_floor_pct * abs(effective_entry),
    )
    # Long : sl_gross = IB_LOW  - stop_buffer
    # Short: sl_gross = IB_HIGH + stop_buffer
    sl_gross = (ib.low if d == 1 else ib.high) + d * (-1) * stop_buffer
    # i.e. long  (d=+1): sl = ib.low  + (-1)*buffer = ib.low  - buffer
    #      short (d=-1): sl = ib.high + (+1)*buffer = ib.high + buffer
    effective_sl = sl_gross - d * half_spread

    stop_distance = abs(effective_entry - effective_sl)
    if stop_distance <= 0.0:
        logger.error(
            "[NY IB EXEC] Zero stop distance: entry=%.2f sl=%.2f. Skipping.",
            effective_entry, effective_sl,
        )
        return None

    # ── TP: fixed R from entry ─────────────────────────────────────────────────
    tp_distance = stop_distance * params.tp_r_multiplier
    effective_tp = effective_entry + d * tp_distance
    tp_gross = effective_tp + d * half_spread
    # tp_gross: candle level that must be reached to net effective_tp after spread

    # ── Position sizing ────────────────────────────────────────────────────────
    risk_amount = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    logger.info(
        "[NY IB EXEC] Setup | Dir=%s | GrossEntry=%.2f EffEntry=%.2f | "
        "IB=[%.2f, %.2f] SL=%.2f TP=%.2f | "
        "StopDist=%.3f TP_R=%.1f | Size=%.4foz Risk=$%.2f",
        "LONG" if d == 1 else "SHORT",
        gross_entry, effective_entry,
        ib.low, ib.high, sl_gross, tp_gross,
        stop_distance, params.tp_r_multiplier,
        position_size, risk_amount,
    )

    # Build a TradeSetup using a thin wrapper around AsianRange to reuse simulate_trade
    # We pass ib data through the asian_range field (same duck-typed structure)
    from ..asian_range_breakout.signal import AsianRange
    ib_as_range = AsianRange(
        high=ib.high,
        low=ib.low,
        range_price=ib.range_price,
        range_pct=ib.range_pct,
        reference_price=ib.reference_price,
        candle_count=ib.candle_count,
        valid=ib.valid,
        block_reason=ib.block_reason,
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
        asian_range=ib_as_range,
    )
