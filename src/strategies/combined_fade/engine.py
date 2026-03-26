"""
Combined Prior-Day Fade backtest engine with regime filter.

Extends the prior_day_fade engine with one additional step:
  Before any signal processing, compute the 10-session trend.
  LONG signals are only acted on in uptrend (trend_10 > 0).
  SHORT signals are only acted on in downtrend (trend_10 < 0).

All signal detection, trade setup, and simulation logic is reused from
prior_day_fade unchanged.  The regime check is the only addition.

This means:
  - The LONG strategy is the original PDL sweep-rejection, regime-gated.
  - The SHORT strategy is a separate behavioural edge: PDH sweep-rejection
    only in environments where gold is weakening over the prior 2 weeks.
    These are not symmetric — they represent two independent observations.

One trade per session maximum (same as prior_day_fade).
If both long and short signals fire in the same session (impossible by
construction — a candle can't be both above PDH and below PDL), the first
in time would be taken.  In practice this never occurs.
"""

import logging
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.engine import _validate_dataframe
from ..asian_range_breakout.execution import TradeResult, simulate_trade
from ..prior_day_fade.engine import _extract_session_starts
from ..prior_day_fade.params import PDFadeParams
from ..prior_day_fade.signal import (
    PDFadeSignal, PriorDayRange,
    compute_pd_fade_setup, compute_prior_day_range, detect_pd_fade_signal,
)
from .regime import (
    REGIME_DOWNTREND, REGIME_UNDEFINED, REGIME_UPTREND,
    build_session_close_index, get_regime,
)

logger = logging.getLogger(__name__)

_TREND_LOOKBACK = 10   # sessions — structural, not a parameter to optimise


def run_combined_fade_backtest(
    df: pd.DataFrame,
    params: PDFadeParams,
) -> List[TradeResult]:
    """
    Run the Combined Fade backtest.

    Long trades: PDL sweep-rejection, only in 10-session uptrend.
    Short trades: PDH sweep-rejection, only in 10-session downtrend.
    """
    params.validate()
    _validate_dataframe(df, params)

    equity = params.initial_equity
    trade_results: List[TradeResult] = []

    session_starts       = _extract_session_starts(df)
    session_close_index  = build_session_close_index(df)

    logger.info(
        "[COMBINED ENGINE] Backtest start | Sessions=%d | Bars=%d | "
        "Start=%s | End=%s | TrendLookback=%d sessions | Equity=$%.2f",
        len(session_starts), len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
        _TREND_LOOKBACK,
        equity,
    )

    regime_counts = {REGIME_UPTREND: 0, REGIME_DOWNTREND: 0, REGIME_UNDEFINED: 0}

    for session_start in session_starts:
        regime = get_regime(session_start, session_close_index, _TREND_LOOKBACK)
        regime_counts[regime] += 1

        result = _process_session(df, session_start, regime, equity, params)
        if result is None:
            continue

        trade_results.append(result)
        equity = result.equity_after
        logger.info(
            "[COMBINED ENGINE] Trade | Session=%s | Regime=%-10s | Dir=%s | "
            "%s | R=%.3f | Equity=$%.2f",
            session_start.strftime("%Y-%m-%d %H:%M UTC"),
            regime,
            "LONG" if result.setup.direction == 1 else "SHORT",
            result.exit_reason,
            result.realized_r,
            equity,
        )

    logger.info(
        "[COMBINED ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%% | "
        "Regime: up=%d down=%d undef=%d",
        len(trade_results),
        equity,
        (equity / params.initial_equity - 1.0) * 100,
        regime_counts[REGIME_UPTREND],
        regime_counts[REGIME_DOWNTREND],
        regime_counts[REGIME_UNDEFINED],
    )

    return trade_results


def _process_session(
    df: pd.DataFrame,
    session_start: pd.Timestamp,
    regime: str,
    equity: float,
    params: PDFadeParams,
) -> Optional[TradeResult]:
    """Process one session with regime gating."""
    tag = session_start.strftime("%Y-%m-%d %H:%M UTC")

    # ── Regime gate ───────────────────────────────────────────────────────────
    if regime == REGIME_UNDEFINED:
        return None   # insufficient history or flat trend — no trade

    # ── Prior day range ───────────────────────────────────────────────────────
    pdr: PriorDayRange = compute_prior_day_range(df, session_start, params)
    if not pdr.valid:
        return None

    # ── Session candles ───────────────────────────────────────────────────────
    signal_start_ts = session_start + pd.Timedelta(hours=params.signal_offset_hours)
    signal_end_ts   = session_start + pd.Timedelta(hours=params.signal_window_end_hours)
    time_exit_ts    = session_start + pd.Timedelta(hours=params.time_exit_hours)
    session_end_ts  = session_start + pd.Timedelta(hours=24)

    session_candles = df[
        (df.index >= session_start) & (df.index < session_end_ts)
    ].copy()

    if len(session_candles) < params.min_session_candles:
        return None

    window_candles = session_candles[
        (session_candles.index >= signal_start_ts)
        & (session_candles.index < signal_end_ts)
    ]

    if window_candles.empty:
        return None

    # ── Detect signal, gated by regime ────────────────────────────────────────
    # We scan the full window for any qualifying signal, then discard if it
    # doesn't match the current regime.  This avoids running two separate scans.
    signal: Optional[PDFadeSignal] = detect_pd_fade_signal(
        window_candles=window_candles,
        pdr=pdr,
        min_overshoot_pct=params.min_overshoot_pct,
    )
    if signal is None:
        return None

    # ── Regime-direction gate ─────────────────────────────────────────────────
    if regime == REGIME_UPTREND and signal.direction != 1:
        logger.info(
            "[COMBINED ENGINE] %s | SKIP short: regime=uptrend, only longs allowed.",
            tag,
        )
        return None
    if regime == REGIME_DOWNTREND and signal.direction != -1:
        logger.info(
            "[COMBINED ENGINE] %s | SKIP long: regime=downtrend, only shorts allowed.",
            tag,
        )
        return None

    # ── Entry candle ──────────────────────────────────────────────────────────
    candles_after_signal = session_candles[
        session_candles.index > signal.signal_candle_ts
    ]
    if candles_after_signal.empty:
        return None

    entry_candle    = candles_after_signal.iloc[0]
    entry_timestamp = candles_after_signal.index[0]

    if entry_timestamp >= time_exit_ts:
        return None

    # ── Trade setup ───────────────────────────────────────────────────────────
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

    # ── Simulate ──────────────────────────────────────────────────────────────
    candles_after_entry = session_candles[session_candles.index > entry_timestamp]
    if candles_after_entry.empty:
        return None

    return simulate_trade(
        setup=setup,
        candles_after_entry=candles_after_entry,
        time_exit_ts=time_exit_ts,
        equity_before=equity,
        params=params,
    )
