"""
Round Number Intraday Rejection backtest engine.

Protocol per session:
  1. Extract signal-window candles [07:00, 16:00) UTC.
  2. Scan candles in time order for the first qualifying round-number rejection.
  3. Enter at the open of the next candle after the signal.
  4. Stop: beyond wick extreme + stop_buffer_price.
  5. TP: entry + d * stop_distance * tp_r_multiplier (default 1R).
  6. Time exit: 16:30 UTC (session + 23.5h), whichever comes first.

One trade per session maximum.

Execution cost model (consistent with prior strategies):
  LONG entry  : candle_open + half_spread + slippage
  SHORT entry : candle_open - half_spread - slippage
  SL exit     : sl_gross ± half_spread
  TP exit     : tp_gross ± half_spread
  TIME exit   : candle_open ± half_spread
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP,
)
from ..prior_day_fade.engine import _extract_session_starts
from .params import RNParams
from .signal import RNSignal, detect_rn_signal

logger = logging.getLogger(__name__)


@dataclass
class RNTradeSetup:
    direction: int
    entry_price: float
    sl_gross: float
    tp_gross: float
    effective_sl: float
    effective_tp: float
    stop_distance: float
    position_size: float
    risk_amount: float
    entry_timestamp: pd.Timestamp
    time_exit_ts: pd.Timestamp
    signal: RNSignal


@dataclass
class RNTradeResult:
    setup: RNTradeSetup
    exit_price: float
    exit_reason: str
    exit_timestamp: pd.Timestamp
    net_pnl: float
    realized_r: float
    equity_after: float


def run_rn_rejection_backtest(
    df: pd.DataFrame,
    params: RNParams,
) -> List[RNTradeResult]:
    params.validate()

    equity  = params.initial_equity
    results: List[RNTradeResult] = []
    session_starts = _extract_session_starts(df)

    logger.info(
        "[RN ENGINE] Start | Sessions=%d | Bars=%d | %s → %s | Equity=$%.2f",
        len(session_starts), len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
        equity,
    )

    for session_start in session_starts:
        result = _process_session(df, session_start, equity, params)
        if result is None:
            continue
        results.append(result)
        equity = result.equity_after
        logger.info(
            "[RN ENGINE] Trade | %s | Level=%.0f | Dir=%s | %s | R=%.3f | Equity=$%.2f",
            session_start.strftime("%Y-%m-%d"),
            result.setup.signal.round_level,
            "LONG" if result.setup.direction == 1 else "SHORT",
            result.exit_reason,
            result.realized_r,
            equity,
        )

    logger.info(
        "[RN ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%%",
        len(results), equity, (equity / params.initial_equity - 1.0) * 100,
    )
    return results


def _process_session(
    df: pd.DataFrame,
    session_start: pd.Timestamp,
    equity: float,
    params: RNParams,
) -> Optional[RNTradeResult]:

    signal_start_ts = session_start + pd.Timedelta(hours=params.signal_start_hours)
    signal_end_ts   = session_start + pd.Timedelta(hours=params.signal_end_hours)
    time_exit_ts    = session_start + pd.Timedelta(hours=params.time_exit_hours)
    session_end_ts  = session_start + pd.Timedelta(hours=24)

    # All candles for this session day (for simulation after entry)
    session_candles = df[
        (df.index >= session_start) & (df.index < session_end_ts)
    ]

    # Active signal window
    window_candles = session_candles[
        (session_candles.index >= signal_start_ts)
        & (session_candles.index < signal_end_ts)
    ]

    if window_candles.empty:
        return None

    # Scan for first qualifying signal
    signal: Optional[RNSignal] = None
    for ts, candle in window_candles.iterrows():
        sig = detect_rn_signal(candle, ts, params)
        if sig is not None:
            signal = sig
            break

    if signal is None:
        return None

    # Entry: next candle after signal
    candles_after_signal = session_candles[session_candles.index > signal.signal_candle_ts]
    if candles_after_signal.empty:
        return None

    entry_candle    = candles_after_signal.iloc[0]
    entry_timestamp = candles_after_signal.index[0]

    if entry_timestamp >= time_exit_ts:
        return None

    setup = _build_setup(signal, entry_candle, entry_timestamp, time_exit_ts, equity, params)
    if setup is None:
        return None

    candles_after_entry = session_candles[session_candles.index > entry_timestamp]
    if candles_after_entry.empty:
        return None

    return _simulate(setup, candles_after_entry, equity, params)


def _build_setup(
    signal: RNSignal,
    entry_candle: pd.Series,
    entry_timestamp: pd.Timestamp,
    time_exit_ts: pd.Timestamp,
    equity: float,
    params: RNParams,
) -> Optional[RNTradeSetup]:

    d = signal.direction
    half_spread = params.spread_price / 2.0

    gross_entry     = float(entry_candle["open"])
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    # Stop: beyond wick extreme + buffer
    sl_gross     = signal.wick_extreme + d * params.stop_buffer_price  # d=-1 → above wick; d=+1 → below wick
    # For LONG (d=1): wick_extreme is the LOW → sl_gross = low - buffer  (d*buffer is negative for d=-1)
    # Correction: stop is AWAY from entry, so:
    #   SHORT (d=-1): sl_gross = wick_extreme + buffer  (above the high)
    #   LONG  (d=+1): sl_gross = wick_extreme - buffer  (below the low)
    sl_gross     = signal.wick_extreme - d * params.stop_buffer_price
    effective_sl = sl_gross - d * half_spread

    stop_distance = abs(effective_entry - effective_sl)
    if stop_distance <= 0.0:
        logger.warning("[RN ENGINE] Zero stop distance at %s. Skip.", entry_timestamp)
        return None

    # TP: 1R (or tp_r_multiplier)
    effective_tp = effective_entry + d * stop_distance * params.tp_r_multiplier
    tp_gross     = effective_tp + d * half_spread

    risk_amount   = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    logger.info(
        "[RN ENGINE] Setup | %s | Level=%.0f | Dir=%s | Entry=%.2f | "
        "SL=%.2f  TP=%.2f | StopDist=%.3f | Size=%.4foz | Risk=$%.2f",
        entry_timestamp.strftime("%Y-%m-%d %H:%M"),
        signal.round_level,
        "LONG" if d == 1 else "SHORT",
        effective_entry, effective_sl, effective_tp,
        stop_distance, position_size, risk_amount,
    )

    return RNTradeSetup(
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
        time_exit_ts=time_exit_ts,
        signal=signal,
    )


def _simulate(
    setup: RNTradeSetup,
    candles_after_entry: pd.DataFrame,
    equity_before: float,
    params: RNParams,
) -> RNTradeResult:

    d = setup.direction
    half_spread = params.spread_price / 2.0

    for ts, candle in candles_after_entry.iterrows():
        o  = float(candle["open"])
        h  = float(candle["high"])
        lo = float(candle["low"])

        if ts >= setup.time_exit_ts:
            exit_eff = o - d * half_spread
            return _build_result(setup, exit_eff, EXIT_TIME, ts, equity_before)

        gap = (d == 1 and o < setup.sl_gross) or (d == -1 and o > setup.sl_gross)
        if gap:
            exit_eff = o - d * half_spread
            return _build_result(setup, exit_eff, EXIT_SL_GAP, ts, equity_before)

        sl_hit = (d == 1 and lo <= setup.sl_gross) or (d == -1 and h >= setup.sl_gross)
        tp_hit = (d == 1 and h >= setup.tp_gross)  or (d == -1 and lo <= setup.tp_gross)

        if sl_hit and tp_hit:
            return _build_result(setup, setup.effective_sl, EXIT_SL, ts, equity_before)
        if sl_hit:
            return _build_result(setup, setup.effective_sl, EXIT_SL, ts, equity_before)
        if tp_hit:
            return _build_result(setup, setup.effective_tp, EXIT_TP, ts, equity_before)

    last_ts    = candles_after_entry.index[-1]
    last_close = float(candles_after_entry.iloc[-1]["close"])
    exit_eff   = last_close - d * half_spread
    logger.warning("[RN ENGINE] END_OF_DATA exit at %s", last_ts)
    return _build_result(setup, exit_eff, EXIT_END_OF_DATA, last_ts, equity_before)


def _build_result(
    setup: RNTradeSetup,
    exit_price: float,
    exit_reason: str,
    exit_timestamp: pd.Timestamp,
    equity_before: float,
) -> RNTradeResult:
    d       = setup.direction
    net_pnl = (exit_price - setup.entry_price) * d * setup.position_size
    r       = net_pnl / setup.risk_amount if setup.risk_amount > 0 else 0.0
    return RNTradeResult(
        setup=setup,
        exit_price=exit_price,
        exit_reason=exit_reason,
        exit_timestamp=exit_timestamp,
        net_pnl=net_pnl,
        realized_r=r,
        equity_after=equity_before + net_pnl,
    )
