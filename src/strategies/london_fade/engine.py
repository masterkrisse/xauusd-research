"""
Fade backtest engine: wires together session, Asian range, fade signal,
trade setup, and simulation — one trade maximum per calendar day.

Architecture mirrors the breakout engine exactly; only the signal and setup
steps differ.  All session, range, and simulation logic is reused unchanged.
"""

import logging
from datetime import date
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.engine import _extract_trading_dates, _validate_dataframe
from ..asian_range_breakout.execution import TradeResult, simulate_trade
from ..asian_range_breakout.session import SessionBoundaries, get_session_boundaries
from ..asian_range_breakout.signal import AsianRange, compute_asian_range
from .params import FadeParams
from .signal import FadeSignal, compute_fade_trade_setup, detect_fade_signal

logger = logging.getLogger(__name__)


def run_fade_backtest(
    df: pd.DataFrame,
    params: FadeParams,
) -> List[TradeResult]:
    """
    Run the London False Breakout Fade backtest over the full dataset.

    Args:
        df:     15-minute OHLC DataFrame.  Index must be UTC-aware DatetimeIndex.
                Required columns: open, high, low, close.
        params: Validated FadeParams.

    Returns:
        List of TradeResult objects in chronological order.
    """
    params.validate()
    _validate_dataframe(df, params)

    equity = params.initial_equity
    trade_results: List[TradeResult] = []
    trading_dates = _extract_trading_dates(df)

    logger.info(
        "[FADE ENGINE] Backtest start | TradingDays=%d | Bars=%d | "
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
            "[FADE ENGINE] Trade done | Date=%s | %s | R=%.3f | Equity=$%.2f",
            trading_date,
            result.exit_reason,
            result.realized_r,
            equity,
        )

    logger.info(
        "[FADE ENGINE] Backtest complete | Trades=%d | FinalEquity=$%.2f | "
        "Return=%.2f%%",
        len(trade_results),
        equity,
        (equity / params.initial_equity - 1.0) * 100,
    )

    return trade_results


def _process_day(
    df: pd.DataFrame,
    trading_date: date,
    equity: float,
    params: FadeParams,
) -> Optional[TradeResult]:
    """
    Process one trading day for the fade strategy.

    Stages:
      1. Compute DST-correct session boundaries.
      2. Extract day candles; skip if too few.
      3. Compute and validate Asian range.
      4. Scan London window for wick rejection (fade signal).
      5. Determine entry candle (next candle after signal).
      6. Build fade trade setup (entry, SL at wick extreme, TP at midpoint).
      7. Simulate trade through remaining day candles.
    """
    date_tag = str(trading_date)

    # ── 1. Session boundaries ─────────────────────────────────────────────────
    session: SessionBoundaries = get_session_boundaries(
        trading_date=trading_date,
        london_window_duration_hours=params.london_window_duration_hours,
        time_exit_hours_after_london_open=params.time_exit_hours_after_london_open,
        candle_minutes=params.candle_minutes,
    )

    # ── 2. Day candles ────────────────────────────────────────────────────────
    day_start = pd.Timestamp(trading_date, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    day_candles = df[(df.index >= day_start) & (df.index < day_end)].copy()

    if len(day_candles) < params.min_day_candles:
        logger.warning(
            "[FADE ENGINE] %s | SKIP | Only %d candles (min=%d).",
            date_tag, len(day_candles), params.min_day_candles,
        )
        return None

    # ── 3. Asian range ────────────────────────────────────────────────────────
    asian_range: AsianRange = compute_asian_range(day_candles, session, params)
    if not asian_range.valid:
        return None

    # ── 4. Fade signal (wick rejection) ───────────────────────────────────────
    window_candles = day_candles[
        (day_candles.index >= session.london_open)
        & (day_candles.index < session.london_window_close)
    ]

    if window_candles.empty:
        logger.warning(
            "[FADE ENGINE] %s | SKIP | No candles in London window [%s, %s).",
            date_tag,
            session.london_open.strftime("%H:%M"),
            session.london_window_close.strftime("%H:%M"),
        )
        return None

    signal: Optional[FadeSignal] = detect_fade_signal(
        window_candles=window_candles,
        asian_range=asian_range,
        min_overshoot_pct=params.min_overshoot_pct,
    )
    if signal is None:
        return None

    # ── 5. Entry candle ───────────────────────────────────────────────────────
    candles_after_signal = day_candles[day_candles.index > signal.signal_candle_ts]

    if candles_after_signal.empty:
        logger.warning(
            "[FADE ENGINE] %s | SKIP | Signal at %s but no candles follow.",
            date_tag, signal.signal_candle_ts,
        )
        return None

    entry_candle = candles_after_signal.iloc[0]
    entry_timestamp = candles_after_signal.index[0]

    if entry_timestamp >= session.time_exit:
        logger.info(
            "[FADE ENGINE] %s | SKIP | Entry candle %s >= TimeExit %s UTC.",
            date_tag,
            entry_timestamp.strftime("%H:%M"),
            session.time_exit.strftime("%H:%M"),
        )
        return None

    # ── 6. Trade setup ────────────────────────────────────────────────────────
    setup = compute_fade_trade_setup(
        signal=signal,
        entry_candle=entry_candle,
        entry_timestamp=entry_timestamp,
        asian_range=asian_range,
        equity=equity,
        params=params,
    )
    if setup is None:
        return None

    # ── 7. Trade simulation (reused unchanged from breakout) ──────────────────
    candles_after_entry = day_candles[day_candles.index > entry_timestamp]

    if candles_after_entry.empty:
        logger.warning(
            "[FADE ENGINE] %s | SKIP | No candles after entry timestamp %s.",
            date_tag, entry_timestamp,
        )
        return None

    result = simulate_trade(
        setup=setup,
        candles_after_entry=candles_after_entry,
        time_exit_ts=session.time_exit,
        equity_before=equity,
        params=params,
    )

    return result
