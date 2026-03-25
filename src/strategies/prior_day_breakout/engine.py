"""
Prior Day H/L Breakout backtest engine.

Pipeline per trading day:
  1. Identify the prior UTC calendar day.
  2. Compute PD_HIGH and PD_LOW (range filter applied).
  3. Extract signal window: today's candles in [signal_start_utc, signal_end_utc).
  4. Scan for first close-confirmed PD breakout.
  5. Entry candle = next candle after signal.
  6. Trade setup: stop just behind breakout level, TP at fixed R.
  7. Simulate trade through remaining day candles to time_exit_utc_hour.

Key distinction from the session strategies:
  The prior day's range is computed from a completed calendar day — there is
  no session window or DST handling required for range computation.
  The signal window uses fixed UTC hours (wide enough to cover London and NY
  regardless of DST transitions).
"""

import logging
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.engine import _extract_trading_dates, _validate_dataframe
from ..asian_range_breakout.execution import TradeResult, simulate_trade
from .params import PDBreakoutParams
from .signal import (
    PDBreakoutSignal, PriorDayRange,
    compute_pd_trade_setup, compute_prior_day_range, detect_pd_breakout,
)

logger = logging.getLogger(__name__)


def run_pd_breakout_backtest(
    df: pd.DataFrame,
    params: PDBreakoutParams,
) -> List[TradeResult]:
    """
    Run the Prior Day H/L Breakout backtest over the full dataset.

    The first calendar day in the dataset is always skipped (no prior day available).
    """
    params.validate()
    _validate_dataframe(df, params)

    equity = params.initial_equity
    trade_results: List[TradeResult] = []
    trading_dates = _extract_trading_dates(df)

    logger.info(
        "[PD ENGINE] Backtest start | TradingDays=%d | Bars=%d | "
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
            "[PD ENGINE] Trade | Date=%s | %s | R=%.3f | Equity=$%.2f",
            trading_date, result.exit_reason, result.realized_r, equity,
        )

    logger.info(
        "[PD ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%%",
        len(trade_results),
        equity,
        (equity / params.initial_equity - 1.0) * 100,
    )

    return trade_results


def _process_day(
    df: pd.DataFrame,
    trading_date: date,
    equity: float,
    params: PDBreakoutParams,
) -> Optional[TradeResult]:
    date_tag = str(trading_date)

    # ── 1. Prior day ──────────────────────────────────────────────────────────
    prior_date = trading_date - timedelta(days=1)

    # ── 2. Prior day range ────────────────────────────────────────────────────
    pdr: PriorDayRange = compute_prior_day_range(df, prior_date, params)
    if not pdr.valid:
        return None

    # ── 3. Today's day candles ────────────────────────────────────────────────
    day_start = pd.Timestamp(trading_date, tz="UTC")
    day_end   = day_start + pd.Timedelta(days=1)
    day_candles = df[(df.index >= day_start) & (df.index < day_end)].copy()

    if len(day_candles) < params.min_day_candles:
        logger.warning(
            "[PD ENGINE] %s | SKIP | Only %d candles (min=%d).",
            date_tag, len(day_candles), params.min_day_candles,
        )
        return None

    # ── 4. Signal window ──────────────────────────────────────────────────────
    window_start = day_start + pd.Timedelta(hours=params.signal_window_start_utc)
    window_end   = day_start + pd.Timedelta(hours=params.signal_window_end_utc)
    time_exit_ts = day_start + pd.Timedelta(hours=params.time_exit_utc_hour)

    window_candles = day_candles[
        (day_candles.index >= window_start)
        & (day_candles.index < window_end)
    ]

    if window_candles.empty:
        logger.warning(
            "[PD ENGINE] %s | SKIP | No candles in signal window [%02d:00, %02d:00) UTC.",
            date_tag,
            params.signal_window_start_utc,
            params.signal_window_end_utc,
        )
        return None

    # ── 5. Signal ─────────────────────────────────────────────────────────────
    signal: Optional[PDBreakoutSignal] = detect_pd_breakout(window_candles, pdr)
    if signal is None:
        return None

    # ── 6. Entry candle ───────────────────────────────────────────────────────
    candles_after_signal = day_candles[day_candles.index > signal.signal_candle_ts]

    if candles_after_signal.empty:
        logger.warning(
            "[PD ENGINE] %s | SKIP | Signal at %s but no candles follow.",
            date_tag, signal.signal_candle_ts,
        )
        return None

    entry_candle    = candles_after_signal.iloc[0]
    entry_timestamp = candles_after_signal.index[0]

    if entry_timestamp >= time_exit_ts:
        logger.info(
            "[PD ENGINE] %s | SKIP | Entry candle %s >= TimeExit %02d:00 UTC.",
            date_tag, entry_timestamp.strftime("%H:%M"), params.time_exit_utc_hour,
        )
        return None

    # ── 7. Trade setup ────────────────────────────────────────────────────────
    setup = compute_pd_trade_setup(
        signal=signal,
        entry_candle=entry_candle,
        entry_timestamp=entry_timestamp,
        pdr=pdr,
        equity=equity,
        params=params,
    )
    if setup is None:
        return None

    # ── 8. Simulate trade ─────────────────────────────────────────────────────
    candles_after_entry = day_candles[day_candles.index > entry_timestamp]

    if candles_after_entry.empty:
        logger.warning(
            "[PD ENGINE] %s | SKIP | No candles after entry %s.",
            date_tag, entry_timestamp,
        )
        return None

    return simulate_trade(
        setup=setup,
        candles_after_entry=candles_after_entry,
        time_exit_ts=time_exit_ts,
        equity_before=equity,
        params=params,
    )
