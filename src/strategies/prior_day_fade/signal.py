"""
Signal detection for the Prior Day High/Low Sweep-and-Rejection Fade.

Prior day range computation:
  PDH = max(high) over candles in [session_start - 24h, session_start)
  PDL = min(low)  over the same window

  "session_start" is the 17:00 UTC boundary that starts the current session.
  The prior window is exactly 24 hours of completed price action.

Sweep-and-rejection signal (single candle):
  SHORT fade (PDH sweep fails):
    candle.high  > PDH                         wick pierces above PDH
    candle.close < PDH                         closes back below PDH (rejection)
    (candle.high - PDH) / PDH >= min_overshoot  wick is meaningful

  LONG fade (PDL sweep fails):
    candle.low   < PDL                         wick pierces below PDL
    candle.close > PDL                         closes back above PDL (rejection)
    (PDL - candle.low) / PDL >= min_overshoot

  No open condition: gap opens that immediately reject also qualify.
  A gap open ABOVE PDH that closes back below PDH is a valid short fade signal —
  it indicates the breakout attempt failed even more decisively.

Entry: open of the candle FOLLOWING the signal candle.

Stop: beyond the wick extreme.
  SHORT: sl_gross = candle.high + stop_buffer  (above the rejected wick)
  LONG : sl_gross = candle.low  - stop_buffer  (below the rejected wick)

Take-profit: opposite prior day boundary.
  SHORT: tp_gross = PDL  (structural target — full prior range reversal)
  LONG : tp_gross = PDH

  Effective TP = tp_gross adjusted for exit spread (net of spread paid to close).
  The candle level that triggers TP:
    SHORT: price must fall to PDL + half_spread (net price to us = PDL)
    LONG : price must rise to PDH - half_spread

Realized R:R varies per trade (stop = overshoot distance; TP = full range).
Logged explicitly for every trade.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..asian_range_breakout.execution import TradeSetup
from ..asian_range_breakout.signal import AsianRange
from .params import PDFadeParams

logger = logging.getLogger(__name__)


@dataclass
class PriorDayRange:
    """
    The prior 24-hour session's high and low (17:00 UTC convention).

    high, low       : PDH and PDL prices.
    range_price     : PDH - PDL in USD.
    range_pct       : range_price / PDH.
    candle_count    : Number of candles in the prior session.
    valid           : False if blocked by a filter.
    block_reason    : Human-readable reason if invalid.
    """
    high: float
    low: float
    range_price: float
    range_pct: float
    candle_count: int
    valid: bool
    block_reason: str = ""


@dataclass
class PDFadeSignal:
    """
    A wick sweep-and-rejection of a prior day boundary.

    direction:          +1 = long fade (wick below PDL, closed back above)
                        -1 = short fade (wick above PDH, closed back below)
    wick_extreme:       The wick tip — candle.high (short) or candle.low (long).
    rejection_level:    The prior day boundary that was swept: PDH or PDL.
    tp_level:           The opposite boundary — structural TP target.
    overshoot_pct:      How far the wick went beyond the boundary as a fraction.
    signal_candle_ts:   Timestamp of the rejection candle.
    signal_candle_close: Close of the rejection candle.
    """
    direction: int
    wick_extreme: float
    rejection_level: float
    tp_level: float
    overshoot_pct: float
    signal_candle_ts: pd.Timestamp
    signal_candle_close: float


def compute_prior_day_range(
    df: pd.DataFrame,
    session_start: pd.Timestamp,
    params: PDFadeParams,
) -> PriorDayRange:
    """
    Compute PDH/PDL from the 24-hour window immediately before session_start.

    Prior window: [session_start - 24h, session_start)

    Returns PriorDayRange with valid=False if:
      - No candles in the prior window (weekend / data gap).
      - Range is below min_pdr_pct (holiday / abnormally quiet session).
      - Range is above max_pdr_pct (spike / news day — stop would be too large).
    """
    prior_start = session_start - pd.Timedelta(hours=24)
    prior_end   = session_start

    prior_candles = df[(df.index >= prior_start) & (df.index < prior_end)]

    if prior_candles.empty:
        logger.info(
            "[PDF] No prior-session candles in [%s, %s). Skipping.",
            prior_start.strftime("%Y-%m-%d %H:%M"),
            prior_end.strftime("%Y-%m-%d %H:%M"),
        )
        return PriorDayRange(0, 0, 0, 0, 0, False, "no_prior_session_data")

    pdh = float(prior_candles["high"].max())
    pdl = float(prior_candles["low"].min())
    range_price = pdh - pdl
    range_pct   = range_price / pdh

    if range_pct < params.min_pdr_pct:
        logger.info(
            "[PDF] Prior session range too tight: %.4f%% < %.4f%% min. "
            "[%s → %s]",
            range_pct * 100, params.min_pdr_pct * 100,
            prior_start.strftime("%Y-%m-%d %H:%M"),
            prior_end.strftime("%Y-%m-%d %H:%M"),
        )
        return PriorDayRange(pdh, pdl, range_price, range_pct,
                             len(prior_candles), False, "range_too_tight")

    if range_pct > params.max_pdr_pct:
        logger.info(
            "[PDF] Prior session range too wide: %.4f%% > %.4f%% max. "
            "[%s → %s]",
            range_pct * 100, params.max_pdr_pct * 100,
            prior_start.strftime("%Y-%m-%d %H:%M"),
            prior_end.strftime("%Y-%m-%d %H:%M"),
        )
        return PriorDayRange(pdh, pdl, range_price, range_pct,
                             len(prior_candles), False, "range_too_wide")

    logger.info(
        "[PDF] Prior session %s | PDH=%.2f PDL=%.2f Range=%.2f (%.4f%%) | n=%d",
        prior_start.strftime("%Y-%m-%d"),
        pdh, pdl, range_price, range_pct * 100, len(prior_candles),
    )
    return PriorDayRange(
        high=pdh,
        low=pdl,
        range_price=range_price,
        range_pct=range_pct,
        candle_count=len(prior_candles),
        valid=True,
    )


def detect_pd_fade_signal(
    window_candles: pd.DataFrame,
    pdr: PriorDayRange,
    min_overshoot_pct: float,
) -> Optional[PDFadeSignal]:
    """
    Scan candles in the signal window for the first wick sweep-and-rejection.

    Returns the first qualifying candle or None.

    SHORT fade: high > PDH  AND  close < PDH  AND  overshoot >= min
    LONG  fade: low  < PDL  AND  close > PDL  AND  overshoot >= min
    """
    for ts, candle in window_candles.iterrows():
        h  = float(candle["high"])
        lo = float(candle["low"])
        c  = float(candle["close"])

        # ── SHORT fade: wick above PDH, closes back below ─────────────────────
        if h > pdr.high and c < pdr.high:
            overshoot = (h - pdr.high) / pdr.high
            if overshoot >= min_overshoot_pct:
                logger.info(
                    "[PDF SIGNAL] SHORT | %s | High=%.2f > PDH=%.2f | "
                    "Close=%.2f (inside) | Overshoot=%.4f%% | TP_target=PDL=%.2f",
                    ts, h, pdr.high, c, overshoot * 100, pdr.low,
                )
                return PDFadeSignal(
                    direction=-1,
                    wick_extreme=h,
                    rejection_level=pdr.high,
                    tp_level=pdr.low,
                    overshoot_pct=overshoot,
                    signal_candle_ts=ts,
                    signal_candle_close=c,
                )

        # ── LONG fade: wick below PDL, closes back above ──────────────────────
        if lo < pdr.low and c > pdr.low:
            overshoot = (pdr.low - lo) / pdr.low
            if overshoot >= min_overshoot_pct:
                logger.info(
                    "[PDF SIGNAL] LONG | %s | Low=%.2f < PDL=%.2f | "
                    "Close=%.2f (inside) | Overshoot=%.4f%% | TP_target=PDH=%.2f",
                    ts, lo, pdr.low, c, overshoot * 100, pdr.high,
                )
                return PDFadeSignal(
                    direction=1,
                    wick_extreme=lo,
                    rejection_level=pdr.low,
                    tp_level=pdr.high,
                    overshoot_pct=overshoot,
                    signal_candle_ts=ts,
                    signal_candle_close=c,
                )

    logger.info("[PDF SIGNAL] No sweep-rejection in signal window.")
    return None


def compute_pd_fade_setup(
    signal: PDFadeSignal,
    entry_candle: pd.Series,
    entry_timestamp: pd.Timestamp,
    pdr: PriorDayRange,
    equity: float,
    params: PDFadeParams,
) -> Optional[TradeSetup]:
    """
    Build a TradeSetup for the prior-day fade entry.

    Entry: open of the candle after the signal candle.

    Entry validation: if the entry candle gaps back through the rejection level
    before we can enter, the move has already reversed too fast — skip.
      SHORT: entry_open > rejection_level (gapped above PDH — reversal reversed)
      LONG : entry_open < rejection_level (gapped below PDL)

    Stop: beyond the wick extreme.
      SHORT: sl_gross = wick_extreme + stop_buffer
      LONG : sl_gross = wick_extreme - stop_buffer

    TP: opposite prior day boundary (structural, non-parametric).
      SHORT: PDL — effective_tp is the net price we receive after closing spread
      LONG : PDH

    Logs R:R explicitly per trade because it is variable (stop = overshoot distance).
    """
    d = signal.direction
    half_spread = params.spread_price / 2.0

    gross_entry = float(entry_candle["open"])

    # ── Entry validation: gap back through rejection level ────────────────────
    # If price gaps back above PDH on entry (for short), the rejection premise
    # already failed before we entered.
    if d == -1 and gross_entry > signal.rejection_level:
        logger.info(
            "[PDF EXEC] SKIP short: entry open %.2f > PDH %.2f (gap above rejection).",
            gross_entry, signal.rejection_level,
        )
        return None
    if d == 1 and gross_entry < signal.rejection_level:
        logger.info(
            "[PDF EXEC] SKIP long: entry open %.2f < PDL %.2f (gap below rejection).",
            gross_entry, signal.rejection_level,
        )
        return None

    # ── Effective entry ────────────────────────────────────────────────────────
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    # ── Stop: beyond wick extreme ──────────────────────────────────────────────
    stop_buffer = max(
        1.5 * params.spread_price,
        params.stop_buffer_floor_pct * abs(effective_entry),
    )
    # SHORT (d=-1): sl_gross = wick_high + stop_buffer  (above the wick)
    # LONG  (d=+1): sl_gross = wick_low  - stop_buffer  (below the wick)
    sl_gross = signal.wick_extreme + d * (-1) * stop_buffer
    effective_sl = sl_gross - d * half_spread

    stop_distance = abs(effective_entry - effective_sl)
    if stop_distance <= 0.0:
        logger.error(
            "[PDF EXEC] Zero stop distance: entry=%.2f sl=%.2f. Skipping.",
            effective_entry, effective_sl,
        )
        return None

    # ── TP: opposite prior day boundary ───────────────────────────────────────
    # The TP is structural — not a fixed R multiple.
    # SHORT: effective_tp = PDL (net price received after closing spread)
    # LONG : effective_tp = PDH
    effective_tp = signal.tp_level
    # Candle level that must be reached to trigger TP (before spread):
    #   SHORT: tp_gross = PDL + half_spread  (price needs to fall to PDL+spread/2)
    #   LONG : tp_gross = PDH - half_spread
    tp_gross = effective_tp + d * half_spread

    tp_distance = abs(effective_entry - effective_tp)
    rr = tp_distance / stop_distance if stop_distance > 0.0 else 0.0

    logger.info(
        "[PDF EXEC] Setup | Dir=%s | Entry=%.2f(gross=%.2f) | "
        "WickExtreme=%.2f SL=%.2f | TP=%.2f(opp_boundary) | "
        "StopDist=%.3f TPDist=%.3f R:R=%.2f | Overshoot=%.4f%% | "
        "Size=%.4foz Risk=$%.2f",
        "SHORT" if d == -1 else "LONG",
        effective_entry, gross_entry,
        signal.wick_extreme, sl_gross, tp_gross,
        stop_distance, tp_distance, rr,
        signal.overshoot_pct * 100,
        equity * params.risk_pct / stop_distance,
        equity * params.risk_pct,
    )

    risk_amount   = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    # Wrap PriorDayRange as AsianRange to reuse simulate_trade unchanged
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
