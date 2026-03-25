"""
Backtest engine: orchestrates signal detection, execution, and trade simulation
day by day.

One trade maximum per calendar day (UTC).
Days with insufficient candles are skipped and logged.
Signal detection and trade execution are fully separated — the engine is the
only component that connects them.

Known limitation (Logic Risk #7):
  Equity compounds across trades.  Running IS and OOS as separate calls will
  reset equity to initial_equity for OOS.  Compare per-trade R metrics across
  windows, not dollar PnL.

Known limitation (Logic Risk #10):
  No holiday or partial-session detection beyond the min_day_candles threshold.
  Days with reduced-hours trading (e.g. US holidays) may produce narrow ranges
  that are filtered by min_range_pct, or they may pass the filter with a
  misleadingly tight range.
"""

import logging
from datetime import date
from typing import List, Optional

import pandas as pd

from .execution import TradeResult, compute_trade_setup, simulate_trade
from .params import StrategyParams
from .session import SessionBoundaries, get_session_boundaries
from .signal import AsianRange, BreakoutSignal, compute_asian_range, detect_breakout

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams,
) -> List[TradeResult]:
    """
    Run the Asian Range → London Breakout backtest over the full dataset.

    Args:
        df:     15-minute OHLC DataFrame.  Index must be UTC-aware DatetimeIndex.
                Required columns: open, high, low, close.
        params: Validated StrategyParams.

    Returns:
        List of TradeResult objects in chronological order.
        One result per executed trade.  Blocked/no-signal days produce no entry.
    """
    params.validate()
    _validate_dataframe(df, params)

    equity = params.initial_equity
    trade_results: List[TradeResult] = []
    trading_dates = _extract_trading_dates(df)

    logger.info(
        "[ENGINE] Backtest start | TradingDays=%d | Bars=%d | "
        "Start=%s | End=%s | InitialEquity=$%.2f",
        len(trading_dates), len(df),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
        equity,
    )

    blocked_range = 0
    no_signal = 0
    skipped_data = 0

    for trading_date in trading_dates:
        result = _process_day(df, trading_date, equity, params)

        if result is None:
            # _process_day logs the specific reason internally
            continue

        trade_results.append(result)
        equity = result.equity_after

        logger.info(
            "[ENGINE] Trade done | Date=%s | %s | R=%.3f | Equity=$%.2f",
            trading_date,
            result.exit_reason,
            result.realized_r,
            equity,
        )

    logger.info(
        "[ENGINE] Backtest complete | Trades=%d | FinalEquity=$%.2f | "
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
    params: StrategyParams,
) -> Optional[TradeResult]:
    """
    Process one trading day.  Returns a TradeResult if a trade was executed,
    None otherwise.

    Stages:
      1. Compute DST-correct session boundaries.
      2. Extract day candles; skip if too few.
      3. Compute and validate Asian range.
      4. Scan London window for breakout signal.
      5. Determine entry candle (next candle after signal).
      6. Build trade setup (entry, SL, TP, size).
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
            "[ENGINE] %s | SKIP | Only %d candles (min=%d). "
            "Likely incomplete or holiday session.",
            date_tag, len(day_candles), params.min_day_candles,
        )
        return None

    # ── 3. Asian range ────────────────────────────────────────────────────────
    asian_range: AsianRange = compute_asian_range(day_candles, session, params)
    if not asian_range.valid:
        return None   # reason already logged by compute_asian_range

    # ── 4. Breakout signal ────────────────────────────────────────────────────
    window_candles = day_candles[
        (day_candles.index >= session.london_open)
        & (day_candles.index < session.london_window_close)
    ]

    if window_candles.empty:
        logger.warning(
            "[ENGINE] %s | SKIP | No candles in London window "
            "[%s, %s).",
            date_tag,
            session.london_open.strftime("%H:%M"),
            session.london_window_close.strftime("%H:%M"),
        )
        return None

    signal: Optional[BreakoutSignal] = detect_breakout(window_candles, asian_range)
    if signal is None:
        return None   # reason already logged by detect_breakout

    # ── 5. Entry candle ───────────────────────────────────────────────────────
    # The entry candle is the first candle whose open is AFTER the signal candle.
    candles_after_signal = day_candles[day_candles.index > signal.signal_candle_ts]

    if candles_after_signal.empty:
        logger.warning(
            "[ENGINE] %s | SKIP | Signal at %s but no candles follow — "
            "cannot fill entry (end-of-data for this day).",
            date_tag, signal.signal_candle_ts,
        )
        return None

    entry_candle = candles_after_signal.iloc[0]
    entry_timestamp = candles_after_signal.index[0]

    # Reject entries at or after the time exit
    if entry_timestamp >= session.time_exit:
        logger.info(
            "[ENGINE] %s | SKIP | Entry candle %s >= TimeExit %s UTC.",
            date_tag,
            entry_timestamp.strftime("%H:%M"),
            session.time_exit.strftime("%H:%M"),
        )
        return None

    # ── 6. Trade setup ────────────────────────────────────────────────────────
    setup = compute_trade_setup(
        signal=signal,
        entry_candle=entry_candle,
        entry_timestamp=entry_timestamp,
        asian_range=asian_range,
        equity=equity,
        params=params,
    )
    if setup is None:
        return None   # reason already logged by compute_trade_setup

    # ── 7. Trade simulation ───────────────────────────────────────────────────
    candles_after_entry = day_candles[day_candles.index > entry_timestamp]

    if candles_after_entry.empty:
        logger.warning(
            "[ENGINE] %s | SKIP | No candles after entry timestamp %s.",
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


def _extract_trading_dates(df: pd.DataFrame) -> List[date]:
    """Return a sorted list of unique UTC calendar dates present in the DataFrame."""
    return sorted(df.index.normalize().unique().map(lambda ts: ts.date()).tolist())


def _validate_dataframe(df: pd.DataFrame, params: StrategyParams) -> None:
    """
    Validate the input DataFrame before running the backtest.
    Raises ValueError on any structural problem.
    Logs warnings for data quality issues that do not halt execution.
    """
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame index must be a pandas DatetimeIndex. "
            f"Got: {type(df.index).__name__}"
        )

    if df.index.tz is None:
        raise ValueError(
            "DataFrame index must be timezone-aware UTC. "
            "Fix: df.index = df.index.tz_localize('UTC')"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index must be sorted in ascending chronological order."
        )

    # Check candle frequency
    if len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        expected = pd.Timedelta(minutes=params.candle_minutes)
        non_standard = (diffs != expected).sum()
        non_standard_pct = non_standard / len(diffs)
        if non_standard_pct > 0.15:
            raise ValueError(
                f"{non_standard_pct:.1%} of candle intervals are not "
                f"{params.candle_minutes} minutes. "
                "Verify the data timeframe matches candle_minutes."
            )
        if non_standard_pct > 0.05:
            logger.warning(
                "[ENGINE] %s of candle intervals are non-standard. "
                "This may indicate missing bars around weekends or holidays.",
                f"{non_standard_pct:.1%}",
            )

    # Data integrity checks
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        raise ValueError(
            f"Data integrity error: {bad_hl} candles have high < low."
        )

    bad_prices = ((df["open"] <= 0) | (df["close"] <= 0)).sum()
    if bad_prices > 0:
        raise ValueError(
            f"Data integrity error: {bad_prices} candles have non-positive open or close."
        )

    logger.info(
        "[ENGINE] DataFrame validated | Rows=%d | %s → %s",
        len(df),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )
