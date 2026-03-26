"""
Multi-Session Trend Exhaustion backtest engine.

Protocol per session:
  1. After the session at idx closes, check for N-consecutive signal.
  2. If signal fires, enter at the open of the first candle of session idx+1.
  3. Stop: S[-1].high + buffer (SHORT) or S[-1].low - buffer (LONG).
  4. TP: S[-2].close (retrace to two sessions ago).
  5. Time exit: 17:00 UTC of session idx+1 (open of session idx+2).

Execution cost model (consistent across all strategies):
  LONG entry  : candle_open + half_spread + slippage
  SHORT entry : candle_open - half_spread - slippage
  SL exit     : sl_gross ± half_spread
  TP exit     : tp_gross ± half_spread
  TIME exit   : candle_open ± half_spread

No trade is taken if:
  - TP is on the wrong side of effective entry (gap through TP on open)
  - Stop distance ≤ 0
  - No candles available in the entry session
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP,
)
from .params import MSEParams
from .signal import MSESignal, SessionInfo, build_session_index, detect_mse_signal

logger = logging.getLogger(__name__)


@dataclass
class MSETradeSetup:
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
    signal: MSESignal


@dataclass
class MSETradeResult:
    setup: MSETradeSetup
    exit_price: float
    exit_reason: str
    exit_timestamp: pd.Timestamp
    net_pnl: float
    realized_r: float
    equity_after: float


def run_mse_backtest(
    df: pd.DataFrame,
    params: MSEParams,
) -> List[MSETradeResult]:
    params.validate()

    equity  = params.initial_equity
    results: List[MSETradeResult] = []

    sessions = build_session_index(df)
    logger.info(
        "[MSE ENGINE] Start | Sessions=%d | Bars=%d | %s → %s | Equity=$%.2f",
        len(sessions), len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
        equity,
    )

    for idx in range(len(sessions) - 1):
        # Check for signal after session idx closes
        signal = detect_mse_signal(sessions, idx, params)
        if signal is None:
            continue

        # Entry session = sessions[idx + 1]
        entry_session = sessions[idx + 1]
        entry_session_start = entry_session.session_start
        time_exit_ts = entry_session_start + pd.Timedelta(hours=24)

        # Candles from entry session start through the time exit candle.
        # Extend one extra hour past time_exit_ts so the 17:00 UTC candle
        # of the next session is included — this triggers the time exit
        # check correctly instead of falling through to END_OF_DATA.
        session_candles = df[
            (df.index >= entry_session_start)
            & (df.index < time_exit_ts + pd.Timedelta(hours=2))
        ]
        if session_candles.empty:
            logger.debug("[MSE ENGINE] No candles for entry session %s", entry_session_start.date())
            continue

        entry_candle    = session_candles.iloc[0]
        entry_timestamp = session_candles.index[0]

        setup = _build_setup(signal, entry_candle, entry_timestamp, time_exit_ts, equity, params)
        if setup is None:
            continue

        candles_after_entry = session_candles[session_candles.index > entry_timestamp]
        if candles_after_entry.empty:
            continue

        result = _simulate(setup, candles_after_entry, equity, params)
        results.append(result)
        equity = result.equity_after

        logger.info(
            "[MSE ENGINE] Trade | %s | Dir=%s | Entry=%.2f | "
            "SL=%.2f TP=%.2f | %s | R=%.3f | Equity=$%.2f",
            entry_timestamp.strftime("%Y-%m-%d"),
            "LONG" if setup.direction == 1 else "SHORT",
            setup.entry_price,
            setup.effective_sl, setup.effective_tp,
            result.exit_reason, result.realized_r, equity,
        )

    logger.info(
        "[MSE ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%%",
        len(results), equity, (equity / params.initial_equity - 1.0) * 100,
    )
    return results


def _build_setup(
    signal: MSESignal,
    entry_candle: pd.Series,
    entry_timestamp: pd.Timestamp,
    time_exit_ts: pd.Timestamp,
    equity: float,
    params: MSEParams,
) -> Optional[MSETradeSetup]:

    d = signal.direction
    half_spread = params.spread_price / 2.0

    gross_entry     = float(entry_candle["open"])
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    stop_buffer = params.stop_buffer_pct * effective_entry
    if d == -1:   # SHORT: stop above S[-1].high
        sl_gross = signal.s_minus1.high + stop_buffer
    else:          # LONG: stop below S[-1].low
        sl_gross = signal.s_minus1.low - stop_buffer
    effective_sl = sl_gross - d * half_spread

    stop_distance = abs(effective_entry - effective_sl)
    if stop_distance <= 0.0:
        logger.warning("[MSE ENGINE] Zero stop distance at %s. Skip.", entry_timestamp)
        return None

    # TP: S[-2].close (pre-spread gross level; adjusted for exit cost)
    tp_gross    = signal.tp_level
    effective_tp = tp_gross + d * half_spread

    # Validate TP is on the correct side of entry
    if d == 1 and effective_tp <= effective_entry:
        logger.debug(
            "[MSE ENGINE] LONG: TP=%.2f <= entry=%.2f on %s. Entry gaped through TP. Skip.",
            effective_tp, effective_entry, entry_timestamp.date(),
        )
        return None
    if d == -1 and effective_tp >= effective_entry:
        logger.debug(
            "[MSE ENGINE] SHORT: TP=%.2f >= entry=%.2f on %s. Entry gaped through TP. Skip.",
            effective_tp, effective_entry, entry_timestamp.date(),
        )
        return None

    tp_distance   = abs(effective_entry - effective_tp)
    rr            = tp_distance / stop_distance
    risk_amount   = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    logger.info(
        "[MSE ENGINE] Setup | %s | Dir=%s | Entry=%.2f | SL=%.2f TP=%.2f | "
        "StopDist=%.3f | R:R=%.2f | Size=%.4foz | Risk=$%.2f",
        entry_timestamp.strftime("%Y-%m-%d %H:%M"),
        "LONG" if d == 1 else "SHORT",
        effective_entry, effective_sl, effective_tp,
        stop_distance, rr, position_size, risk_amount,
    )

    return MSETradeSetup(
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
    setup: MSETradeSetup,
    candles_after_entry: pd.DataFrame,
    equity_before: float,
    params: MSEParams,
) -> MSETradeResult:

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
    logger.warning("[MSE ENGINE] END_OF_DATA at %s", last_ts)
    return _build_result(setup, exit_eff, EXIT_END_OF_DATA, last_ts, equity_before)


def _build_result(
    setup: MSETradeSetup,
    exit_price: float,
    exit_reason: str,
    exit_timestamp: pd.Timestamp,
    equity_before: float,
) -> MSETradeResult:
    d       = setup.direction
    net_pnl = (exit_price - setup.entry_price) * d * setup.position_size
    r       = net_pnl / setup.risk_amount if setup.risk_amount > 0 else 0.0
    return MSETradeResult(
        setup=setup,
        exit_price=exit_price,
        exit_reason=exit_reason,
        exit_timestamp=exit_timestamp,
        net_pnl=net_pnl,
        realized_r=r,
        equity_after=equity_before + net_pnl,
    )
