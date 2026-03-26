"""
London Opening Range Breakout backtest engine.

Protocol per session:
  1. Compute LOR from [07:00, 08:00) UTC candles. Validate size.
  2. Scan signal window [08:00, 13:00) UTC for first close outside LOR.
  3. Enter on open of next candle after signal.
  4. Stop: opposite LOR boundary + buffer.
  5. TP:   breakout_level ± LOR_size * tp_lor_multiplier (measured move).
  6. Time exit: 14:00 UTC.

Cost model (identical to simulate_trade in asian_range_breakout.execution):
  LONG:  entry_eff = open + half_spread + slippage
         sl_gross  = LOR_L - stop_buffer  (raw level triggering SL check)
         effective_sl = sl_gross - half_spread
         tp_gross  = LOR_H + LOR_size * multiplier  (raw level triggering TP check)
         effective_tp = tp_gross - half_spread
  SHORT: signs flipped.

Exit priority (same as all prior strategies):
  1. Time exit (candle ts >= time_exit_ts)
  2. Gap-through stop (open beyond sl_gross)
  3. SL hit (candle extreme breaches sl_gross)
  4. TP hit (candle extreme breaches tp_gross)
  5. Both in same candle → SL (conservative)

One trade per session maximum.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP,
)
from ..prior_day_fade.engine import _extract_session_starts
from .params import LBParams
from .signal import LBRange, LBSignal, compute_lor, detect_lb_signal

logger = logging.getLogger(__name__)


@dataclass
class LBTradeSetup:
    direction:       int
    entry_price:     float       # effective (post-cost)
    sl_gross:        float
    tp_gross:        float
    effective_sl:    float
    effective_tp:    float
    stop_distance:   float
    position_size:   float
    risk_amount:     float
    entry_timestamp: pd.Timestamp
    signal_timestamp: pd.Timestamp
    lor:             LBRange


@dataclass
class LBTradeResult:
    setup:           LBTradeSetup
    exit_price:      float
    exit_reason:     str
    exit_timestamp:  pd.Timestamp
    net_pnl:         float
    realized_r:      float
    equity_after:    float


def run_lb_backtest(
    df:     pd.DataFrame,
    params: LBParams,
) -> List[LBTradeResult]:
    """
    Run the London Opening Range Breakout backtest.
    Returns a list of LBTradeResult (both LONG and SHORT).
    """
    params.validate()

    equity  = params.initial_equity
    results: List[LBTradeResult] = []

    session_starts = _extract_session_starts(df)

    logger.info(
        "[LB ENGINE] Start | Sessions=%d | Bars=%d | %s → %s | Equity=$%.2f",
        len(session_starts), len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
        equity,
    )

    skip_counts = {"no_lor": 0, "no_signal": 0, "no_entry_candle": 0, "past_time_exit": 0, "bad_setup": 0}

    for session_start in session_starts:
        session_end_ts  = session_start + pd.Timedelta(hours=24)
        time_exit_ts    = session_start + pd.Timedelta(hours=params.time_exit_hours)
        signal_start_ts = session_start + pd.Timedelta(hours=params.signal_start_hours)
        signal_end_ts   = session_start + pd.Timedelta(hours=params.signal_end_hours)

        # ── 1. Compute and validate LOR ───────────────────────────────────────
        lor = compute_lor(df, session_start, params)
        if not lor.valid:
            skip_counts["no_lor"] += 1
            continue

        # ── 2. Detect breakout signal ─────────────────────────────────────────
        session_candles = df[
            (df.index >= session_start) & (df.index < session_end_ts)
        ]
        window_candles = session_candles[
            (session_candles.index >= signal_start_ts)
            & (session_candles.index < signal_end_ts)
        ]
        if window_candles.empty:
            skip_counts["no_signal"] += 1
            continue

        signal = detect_lb_signal(window_candles, lor, params)
        if signal is None:
            skip_counts["no_signal"] += 1
            continue

        # ── 3. Entry candle ───────────────────────────────────────────────────
        candles_after_signal = session_candles[
            session_candles.index > signal.signal_candle_ts
        ]
        if candles_after_signal.empty:
            skip_counts["no_entry_candle"] += 1
            continue

        entry_candle    = candles_after_signal.iloc[0]
        entry_timestamp = candles_after_signal.index[0]

        if entry_timestamp >= time_exit_ts:
            skip_counts["past_time_exit"] += 1
            continue

        # ── 4. Build trade setup ──────────────────────────────────────────────
        setup = _build_setup(signal, entry_candle, entry_timestamp, lor, equity, params)
        if setup is None:
            skip_counts["bad_setup"] += 1
            continue

        # ── 5. Simulate ───────────────────────────────────────────────────────
        candles_after_entry = session_candles[session_candles.index > entry_timestamp]
        if candles_after_entry.empty:
            skip_counts["no_entry_candle"] += 1
            continue

        result = _simulate(setup, candles_after_entry, time_exit_ts, equity, params)
        results.append(result)
        equity = result.equity_after

        logger.info(
            "[LB ENGINE] Trade | %s | Dir=%s | LOR=[%.2f,%.2f] (%.4f%%) | "
            "%s | R=%.3f | Equity=$%.2f",
            session_start.strftime("%Y-%m-%d %H:%M UTC"),
            "LONG" if signal.direction == 1 else "SHORT",
            lor.low, lor.high, lor.size_pct * 100,
            result.exit_reason,
            result.realized_r,
            equity,
        )

    logger.info(
        "[LB ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%% | "
        "Skips: no_lor=%d no_signal=%d no_entry=%d past_exit=%d bad_setup=%d",
        len(results), equity,
        (equity / params.initial_equity - 1.0) * 100,
        skip_counts["no_lor"], skip_counts["no_signal"],
        skip_counts["no_entry_candle"], skip_counts["past_time_exit"],
        skip_counts["bad_setup"],
    )
    return results


def _build_setup(
    signal:          LBSignal,
    entry_candle:    pd.Series,
    entry_timestamp: pd.Timestamp,
    lor:             LBRange,
    equity:          float,
    params:          LBParams,
) -> Optional[LBTradeSetup]:
    d          = signal.direction
    half_spread = params.spread_price / 2.0

    gross_entry    = float(entry_candle["open"])
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    # Stop: opposite LOR boundary + buffer
    stop_buffer = max(1.5 * params.spread_price, params.stop_buffer_floor_pct * effective_entry)
    sl_gross    = (lor.low if d == 1 else lor.high) - d * stop_buffer
    effective_sl = sl_gross - d * half_spread

    stop_distance = abs(effective_entry - effective_sl)
    if stop_distance <= 0.0:
        logger.error(
            "[LB SETUP] Invalid stop distance %.5f | entry=%.2f sl=%.2f dir=%+d. Skip.",
            stop_distance, effective_entry, effective_sl, d,
        )
        return None

    # TP: measured move from breakout level
    # tp_gross is the raw price level (candle extreme) that triggers TP.
    # effective_tp accounts for exit spread cost.
    tp_gross_raw = signal.tp_gross   # breakout_level ± size * multiplier
    effective_tp  = tp_gross_raw - d * half_spread

    risk_amount   = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    # Implied R:R for logging
    tp_dist = abs(effective_tp - effective_entry)
    rr      = tp_dist / stop_distance if stop_distance > 0 else 0.0

    logger.info(
        "[LB SETUP] Dir=%s | GrossEntry=%.2f  EffEntry=%.2f | "
        "SL_gross=%.2f  TP_gross=%.2f | StopDist=%.3f  TPDist=%.3f  R:R=%.2f | "
        "LOR_size=%.2f | Size=%.4foz Risk=$%.2f",
        "LONG" if d == 1 else "SHORT",
        gross_entry, effective_entry,
        sl_gross, tp_gross_raw,
        stop_distance, tp_dist, rr,
        lor.size,
        position_size, risk_amount,
    )

    return LBTradeSetup(
        direction=d,
        entry_price=effective_entry,
        sl_gross=sl_gross,
        tp_gross=tp_gross_raw,
        effective_sl=effective_sl,
        effective_tp=effective_tp,
        stop_distance=stop_distance,
        position_size=position_size,
        risk_amount=risk_amount,
        entry_timestamp=entry_timestamp,
        signal_timestamp=signal.signal_candle_ts,
        lor=lor,
    )


def _simulate(
    setup:               LBTradeSetup,
    candles_after_entry: pd.DataFrame,
    time_exit_ts:        pd.Timestamp,
    equity_before:       float,
    params:              LBParams,
) -> LBTradeResult:
    """
    Simulate candle-by-candle with identical exit priority to simulate_trade.
    """
    d           = setup.direction
    half_spread = params.spread_price / 2.0

    for ts, candle in candles_after_entry.iterrows():
        o  = float(candle["open"])
        h  = float(candle["high"])
        lo = float(candle["low"])

        # 1. Time exit
        if ts >= time_exit_ts:
            exit_eff = o - d * half_spread
            logger.info("[LB EXIT] TIME | %s | Open=%.2f  EffExit=%.2f", ts, o, exit_eff)
            return _build_result(setup, exit_eff, EXIT_TIME, ts, equity_before)

        # 2. Gap-through stop
        gap_through = (d == 1 and o < setup.sl_gross) or (d == -1 and o > setup.sl_gross)
        if gap_through:
            exit_eff = o - d * half_spread
            logger.info(
                "[LB EXIT] SL_GAP | %s | Open=%.2f  SL_gross=%.2f  EffExit=%.2f",
                ts, o, setup.sl_gross, exit_eff,
            )
            return _build_result(setup, exit_eff, EXIT_SL_GAP, ts, equity_before)

        # 3 & 4. SL and TP
        sl_hit = (d == 1 and lo <= setup.sl_gross) or (d == -1 and h >= setup.sl_gross)
        tp_hit = (d == 1 and h >= setup.tp_gross)  or (d == -1 and lo <= setup.tp_gross)

        if sl_hit and tp_hit:
            exit_eff = setup.effective_sl
            logger.info(
                "[LB EXIT] SL (both hit — conservative) | %s | EffExit=%.2f", ts, exit_eff,
            )
            return _build_result(setup, exit_eff, EXIT_SL, ts, equity_before)

        if sl_hit:
            logger.info(
                "[LB EXIT] SL | %s | Low=%.2f <= SL_gross=%.2f | EffExit=%.2f",
                ts, lo, setup.sl_gross, setup.effective_sl,
            )
            return _build_result(setup, setup.effective_sl, EXIT_SL, ts, equity_before)

        if tp_hit:
            logger.info(
                "[LB EXIT] TP | %s | High=%.2f >= TP_gross=%.2f | EffExit=%.2f",
                ts, h, setup.tp_gross, setup.effective_tp,
            )
            return _build_result(setup, setup.effective_tp, EXIT_TP, ts, equity_before)

    # End-of-data
    last_ts    = candles_after_entry.index[-1]
    last_close = float(candles_after_entry.iloc[-1]["close"])
    exit_eff   = last_close - d * half_spread
    logger.warning(
        "[LB EXIT] END_OF_DATA | %s | Close=%.2f  EffExit=%.2f", last_ts, last_close, exit_eff,
    )
    return _build_result(setup, exit_eff, EXIT_END_OF_DATA, last_ts, equity_before)


def _build_result(
    setup:           LBTradeSetup,
    exit_price:      float,
    exit_reason:     str,
    exit_timestamp:  pd.Timestamp,
    equity_before:   float,
) -> LBTradeResult:
    d         = setup.direction
    net_pnl   = (exit_price - setup.entry_price) * d * setup.position_size
    realized_r = net_pnl / setup.risk_amount if setup.risk_amount > 0.0 else 0.0
    equity_after = equity_before + net_pnl

    logger.info(
        "[LB RESULT] %s | Entry=%.2f  Exit=%.2f | PnL=$%.2f | R=%.3f | EquityAfter=$%.2f",
        exit_reason, setup.entry_price, exit_price, net_pnl, realized_r, equity_after,
    )
    return LBTradeResult(
        setup=setup,
        exit_price=exit_price,
        exit_reason=exit_reason,
        exit_timestamp=exit_timestamp,
        net_pnl=net_pnl,
        realized_r=realized_r,
        equity_after=equity_after,
    )
