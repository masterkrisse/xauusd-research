"""
NY IB Breakout backtest engine.

Pipeline per trading day:
  1. DST-correct NY session boundaries
  2. Day candles (UTC calendar date)
  3. Compute initial balance (IB) from first ib_duration_minutes of NY
  4. Scan signal window for close-confirmed IB breakout
  5. Entry candle = next candle after signal
  6. Build trade setup (entry, SL at opposite IB boundary, TP at fixed R)
  7. Simulate trade through remaining candles
"""

import logging
from datetime import date
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.engine import _extract_trading_dates, _validate_dataframe
from ..asian_range_breakout.execution import TradeResult, simulate_trade
from .params import NYIBParams
from .session import NYSessionBoundaries, get_ny_session_boundaries
from .signal import (
    IBBreakoutSignal, InitialBalance,
    compute_ib_range, compute_ib_trade_setup, detect_ib_breakout,
)

logger = logging.getLogger(__name__)


def run_ny_ib_backtest(
    df: pd.DataFrame,
    params: NYIBParams,
) -> List[TradeResult]:
    """
    Run the NY Morning IB Breakout backtest over the full dataset.

    Args:
        df:     15-minute OHLC DataFrame.  Index must be UTC-aware DatetimeIndex.
        params: Validated NYIBParams.

    Returns:
        List of TradeResult objects in chronological order.
    """
    params.validate()
    _validate_dataframe(df, params)

    equity = params.initial_equity
    trade_results: List[TradeResult] = []
    trading_dates = _extract_trading_dates(df)

    logger.info(
        "[NY IB ENGINE] Backtest start | TradingDays=%d | Bars=%d | "
        "Start=%s | End=%s | InitialEquity=$%.2f",
        len(trading_dates), len(df),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
        equity,
    )

    for trading_date in trading_dates:
        result = _process_day(df, trading_date, equity, params)
        if result is None:
            continue

        trade_results.append(result)
        equity = result.equity_after

        logger.info(
            "[NY IB ENGINE] Trade | Date=%s | %s | R=%.3f | Equity=$%.2f",
            trading_date, result.exit_reason, result.realized_r, equity,
        )

    logger.info(
        "[NY IB ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%%",
        len(trade_results),
        equity,
        (equity / params.initial_equity - 1.0) * 100,
    )

    return trade_results


def _process_day(
    df: pd.DataFrame,
    trading_date: date,
    equity: float,
    params: NYIBParams,
) -> Optional[TradeResult]:
    date_tag = str(trading_date)

    # ── 1. Session boundaries ─────────────────────────────────────────────────
    session: NYSessionBoundaries = get_ny_session_boundaries(trading_date, params)

    # ── 2. Day candles ────────────────────────────────────────────────────────
    day_start = pd.Timestamp(trading_date, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    day_candles = df[(df.index >= day_start) & (df.index < day_end)].copy()

    if len(day_candles) < params.min_day_candles:
        logger.warning(
            "[NY IB ENGINE] %s | SKIP | Only %d candles (min=%d).",
            date_tag, len(day_candles), params.min_day_candles,
        )
        return None

    # ── 3. Initial balance ────────────────────────────────────────────────────
    ib: InitialBalance = compute_ib_range(day_candles, session, params)
    if not ib.valid:
        return None

    # ── 4. Signal: close-confirmed breakout of IB ─────────────────────────────
    window_candles = day_candles[
        (day_candles.index >= session.ib_close)
        & (day_candles.index < session.signal_window_close)
    ]

    if window_candles.empty:
        logger.warning(
            "[NY IB ENGINE] %s | SKIP | No candles in signal window [%s, %s).",
            date_tag,
            session.ib_close.strftime("%H:%M UTC"),
            session.signal_window_close.strftime("%H:%M UTC"),
        )
        return None

    signal: Optional[IBBreakoutSignal] = detect_ib_breakout(window_candles, ib)
    if signal is None:
        return None

    # ── 5. Entry candle ───────────────────────────────────────────────────────
    candles_after_signal = day_candles[day_candles.index > signal.signal_candle_ts]

    if candles_after_signal.empty:
        logger.warning(
            "[NY IB ENGINE] %s | SKIP | Signal at %s but no candles follow.",
            date_tag, signal.signal_candle_ts,
        )
        return None

    entry_candle = candles_after_signal.iloc[0]
    entry_timestamp = candles_after_signal.index[0]

    if entry_timestamp >= session.time_exit:
        logger.info(
            "[NY IB ENGINE] %s | SKIP | Entry candle %s >= TimeExit %s UTC.",
            date_tag,
            entry_timestamp.strftime("%H:%M"),
            session.time_exit.strftime("%H:%M"),
        )
        return None

    # ── 6. Trade setup ────────────────────────────────────────────────────────
    setup = compute_ib_trade_setup(
        signal=signal,
        entry_candle=entry_candle,
        entry_timestamp=entry_timestamp,
        ib=ib,
        equity=equity,
        params=params,
    )
    if setup is None:
        return None

    # ── 7. Trade simulation (reused from breakout) ────────────────────────────
    candles_after_entry = day_candles[day_candles.index > entry_timestamp]

    if candles_after_entry.empty:
        logger.warning(
            "[NY IB ENGINE] %s | SKIP | No candles after entry %s.",
            date_tag, entry_timestamp,
        )
        return None

    return simulate_trade(
        setup=setup,
        candles_after_entry=candles_after_entry,
        time_exit_ts=session.time_exit,
        equity_before=equity,
        params=params,
    )
