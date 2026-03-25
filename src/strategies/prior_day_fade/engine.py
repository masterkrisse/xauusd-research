"""
Prior Day High/Low Sweep-and-Rejection Fade — backtest engine.

Session loop (17:00 UTC convention):
  Rather than iterating over UTC calendar dates, the engine iterates over
  "session starts" — every 17:00 UTC timestamp in the dataset.  This correctly
  aligns the prior-day window (the 24h before each session start) with the
  signal window (the 23h after each session start).

  One trade maximum per session.

Session skipped if:
  - Prior 24h has no candles or a range outside [min_pdr_pct, max_pdr_pct].
  - Signal window has no candles.
  - No sweep-rejection signal is found.
  - Entry candle gaps back through the rejection level.
  - Entry candle timestamp is at or after time_exit.

Weekend handling:
  XAUUSD is closed Friday ~21:00 UTC to Sunday ~21:00 UTC.
  Sessions starting at 17:00 UTC on Saturday and Sunday will have empty
  or near-empty candle windows and will be skipped automatically.

No-lookahead guarantee:
  PDH/PDL are computed exclusively from candles with timestamps BEFORE
  session_start.  The signal can only fire on candles AFTER session_start.
  There is no overlap between the prior-day computation window and the
  signal window.
"""

import logging
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.engine import _validate_dataframe
from ..asian_range_breakout.execution import TradeResult, simulate_trade
from .params import PDFadeParams
from .signal import (
    PDFadeSignal, PriorDayRange,
    compute_pd_fade_setup, compute_prior_day_range, detect_pd_fade_signal,
)

logger = logging.getLogger(__name__)

_SESSION_HOUR_UTC = 17  # 17:00 UTC session boundary


def run_pd_fade_backtest(
    df: pd.DataFrame,
    params: PDFadeParams,
) -> List[TradeResult]:
    """
    Run the Prior Day Sweep-and-Rejection Fade backtest over the full dataset.
    """
    params.validate()
    _validate_dataframe(df, params)

    equity = params.initial_equity
    trade_results: List[TradeResult] = []

    session_starts = _extract_session_starts(df)
    logger.info(
        "[PDF ENGINE] Backtest start | Sessions=%d | Bars=%d | "
        "DataStart=%s | DataEnd=%s | InitialEquity=$%.2f",
        len(session_starts), len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
        equity,
    )

    for session_start in session_starts:
        result = _process_session(df, session_start, equity, params)
        if result is None:
            continue

        trade_results.append(result)
        equity = result.equity_after
        logger.info(
            "[PDF ENGINE] Trade | Session=%s | %s | R=%.3f | Equity=$%.2f",
            session_start.strftime("%Y-%m-%d %H:%M UTC"),
            result.exit_reason,
            result.realized_r,
            equity,
        )

    logger.info(
        "[PDF ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%%",
        len(trade_results),
        equity,
        (equity / params.initial_equity - 1.0) * 100,
    )
    return trade_results


def _process_session(
    df: pd.DataFrame,
    session_start: pd.Timestamp,
    equity: float,
    params: PDFadeParams,
) -> Optional[TradeResult]:
    """
    Process one 17:00 UTC session.

    1. Compute PDH/PDL from the prior 24h.
    2. Extract the signal window within the current session.
    3. Detect first sweep-rejection signal.
    4. Build trade setup (stop beyond wick, TP at opposite boundary).
    5. Simulate trade.
    """
    tag = session_start.strftime("%Y-%m-%d %H:%M UTC")

    # ── 1. Prior day range ────────────────────────────────────────────────────
    pdr: PriorDayRange = compute_prior_day_range(df, session_start, params)
    if not pdr.valid:
        return None

    # ── 2. Session windows ────────────────────────────────────────────────────
    signal_start_ts = session_start + pd.Timedelta(hours=params.signal_offset_hours)
    signal_end_ts   = session_start + pd.Timedelta(hours=params.signal_window_end_hours)
    time_exit_ts    = session_start + pd.Timedelta(hours=params.time_exit_hours)

    # All candles in the current session (for entry + simulation)
    session_end_ts  = session_start + pd.Timedelta(hours=24)
    session_candles = df[
        (df.index >= session_start) & (df.index < session_end_ts)
    ].copy()

    if len(session_candles) < params.min_session_candles:
        logger.info(
            "[PDF ENGINE] %s | SKIP | Only %d session candles (min=%d).",
            tag, len(session_candles), params.min_session_candles,
        )
        return None

    # ── 3. Signal window ──────────────────────────────────────────────────────
    window_candles = session_candles[
        (session_candles.index >= signal_start_ts)
        & (session_candles.index < signal_end_ts)
    ]

    if window_candles.empty:
        logger.info(
            "[PDF ENGINE] %s | SKIP | No candles in signal window [%s, %s).",
            tag,
            signal_start_ts.strftime("%H:%M"),
            signal_end_ts.strftime("%H:%M"),
        )
        return None

    # ── 4. Sweep-rejection signal ─────────────────────────────────────────────
    signal: Optional[PDFadeSignal] = detect_pd_fade_signal(
        window_candles=window_candles,
        pdr=pdr,
        min_overshoot_pct=params.min_overshoot_pct,
    )
    if signal is None:
        return None

    # ── 5. Entry candle ───────────────────────────────────────────────────────
    candles_after_signal = session_candles[
        session_candles.index > signal.signal_candle_ts
    ]

    if candles_after_signal.empty:
        logger.warning(
            "[PDF ENGINE] %s | SKIP | Signal at %s but no candles follow.",
            tag, signal.signal_candle_ts,
        )
        return None

    entry_candle    = candles_after_signal.iloc[0]
    entry_timestamp = candles_after_signal.index[0]

    if entry_timestamp >= time_exit_ts:
        logger.info(
            "[PDF ENGINE] %s | SKIP | Entry %s >= TimeExit %s.",
            tag,
            entry_timestamp.strftime("%Y-%m-%d %H:%M"),
            time_exit_ts.strftime("%Y-%m-%d %H:%M"),
        )
        return None

    # ── 6. Trade setup ────────────────────────────────────────────────────────
    setup = compute_pd_fade_setup(
        signal=signal,
        entry_candle=entry_candle,
        entry_timestamp=entry_timestamp,
        pdr=pdr,
        equity=equity,
        params=params,
    )
    if setup is None:
        return None

    # ── 7. Simulate trade ─────────────────────────────────────────────────────
    candles_after_entry = session_candles[
        session_candles.index > entry_timestamp
    ]

    if candles_after_entry.empty:
        logger.warning(
            "[PDF ENGINE] %s | SKIP | No candles after entry %s.",
            tag, entry_timestamp,
        )
        return None

    return simulate_trade(
        setup=setup,
        candles_after_entry=candles_after_entry,
        time_exit_ts=time_exit_ts,
        equity_before=equity,
        params=params,
    )


def _extract_session_starts(df: pd.DataFrame) -> List[pd.Timestamp]:
    """
    Return all 17:00 UTC timestamps that fall within the data range.

    Generates one candidate per UTC calendar date in the index.
    Candidates before df.index[0] + 25h are excluded (need >= 24h of prior data).
    """
    dates = sorted(df.index.normalize().unique())
    min_ts = df.index[0] + pd.Timedelta(hours=25)   # need at least 24h prior data
    max_ts = df.index[-1]

    session_starts = []
    for d in dates:
        ts = d + pd.Timedelta(hours=_SESSION_HOUR_UTC)
        if min_ts <= ts <= max_ts:
            session_starts.append(ts)

    return session_starts
