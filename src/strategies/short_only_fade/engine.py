"""
Short-Only Macro-Filtered Fade backtest engine.

Protocol per session:
  1. Check 20-session MA slope.  If not bearish → skip.
  2. Compute prior day range (PDR).  If invalid → skip.
  3. Scan signal window [+14h, +23h) for PDH sweep-rejection signal.
  4. Enter SHORT on next candle open after signal.
  5. Stop: wick high + buffer.
  6. TP: PDL (full prior day range reversal).
  7. Time exit: +23.5h (16:30 UTC).

Signal and execution logic is reused verbatim from prior_day_fade.
The only additions are:
  - macro filter gate before any processing
  - direction check: only direction == -1 (SHORT) is accepted

One trade per session maximum.
"""

import logging
from typing import List, Optional

import pandas as pd

from ..asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP, TradeResult,
)
from ..asian_range_breakout.execution import simulate_trade
from ..prior_day_fade.engine import _extract_session_starts
from ..prior_day_fade.signal import (
    compute_pd_fade_setup, compute_prior_day_range,
    detect_pd_fade_signal,
)
from .filter import FILTER_BEARISH, build_filter_index, get_filter_state
from .params import SOFParams

logger = logging.getLogger(__name__)


def run_sof_backtest(
    df: pd.DataFrame,
    params: SOFParams,
) -> List[TradeResult]:
    """
    Run the Short-Only Macro-Filtered Fade backtest.
    Returns a list of TradeResult (all trades are SHORT).
    """
    params.validate()

    equity  = params.initial_equity
    results: List[TradeResult] = []

    filter_index   = build_filter_index(df, params)
    session_starts = _extract_session_starts(df)

    filter_counts = {FILTER_BEARISH: 0, "neutral": 0, "undefined": 0}

    logger.info(
        "[SOF ENGINE] Start | Sessions=%d | Bars=%d | %s → %s | Equity=$%.2f",
        len(session_starts), len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
        equity,
    )

    for session_start in session_starts:
        # ── 1. Macro filter ───────────────────────────────────────────────────
        state = get_filter_state(session_start, filter_index)
        filter_counts[state] = filter_counts.get(state, 0) + 1

        if state != FILTER_BEARISH:
            continue

        # ── 2. Prior day range ────────────────────────────────────────────────
        # Reuse prior_day_fade params adapter via duck-typed SOFParams
        pdr = compute_prior_day_range(df, session_start, _ParamsAdapter(params))
        if not pdr.valid:
            continue

        # ── 3. Signal window candles ──────────────────────────────────────────
        signal_start_ts = session_start + pd.Timedelta(hours=params.signal_offset_hours)
        signal_end_ts   = session_start + pd.Timedelta(hours=params.signal_window_end_hours)
        time_exit_ts    = session_start + pd.Timedelta(hours=params.time_exit_hours)
        session_end_ts  = session_start + pd.Timedelta(hours=24)

        session_candles = df[
            (df.index >= session_start) & (df.index < session_end_ts)
        ]
        if len(session_candles) < params.min_session_candles:
            continue

        window_candles = session_candles[
            (session_candles.index >= signal_start_ts)
            & (session_candles.index < signal_end_ts)
        ]
        if window_candles.empty:
            continue

        # ── 4. Signal detection — SHORT only ──────────────────────────────────
        signal = detect_pd_fade_signal(
            window_candles=window_candles,
            pdr=pdr,
            min_overshoot_pct=params.min_overshoot_pct,
        )
        if signal is None or signal.direction != -1:
            if signal is not None and signal.direction == 1:
                logger.debug(
                    "[SOF ENGINE] %s | SKIP LONG signal: short-only strategy.",
                    session_start.strftime("%Y-%m-%d"),
                )
            continue

        # ── 5. Entry candle ───────────────────────────────────────────────────
        candles_after_signal = session_candles[
            session_candles.index > signal.signal_candle_ts
        ]
        if candles_after_signal.empty:
            continue

        entry_candle    = candles_after_signal.iloc[0]
        entry_timestamp = candles_after_signal.index[0]

        if entry_timestamp >= time_exit_ts:
            continue

        # ── 6. Trade setup ────────────────────────────────────────────────────
        setup = compute_pd_fade_setup(
            signal=signal,
            entry_candle=entry_candle,
            entry_timestamp=entry_timestamp,
            pdr=pdr,
            equity=equity,
            params=_ParamsAdapter(params),
        )
        if setup is None:
            continue

        # ── 7. Simulate ───────────────────────────────────────────────────────
        candles_after_entry = session_candles[session_candles.index > entry_timestamp]
        if candles_after_entry.empty:
            continue

        result = simulate_trade(
            setup=setup,
            candles_after_entry=candles_after_entry,
            time_exit_ts=time_exit_ts,
            equity_before=equity,
            params=_ParamsAdapter(params),
        )
        results.append(result)
        equity = result.equity_after

        logger.info(
            "[SOF ENGINE] Trade | %s | Macro=bearish | PDH=%.2f → SHORT | "
            "%s | R=%.3f | Equity=$%.2f",
            session_start.strftime("%Y-%m-%d %H:%M UTC"),
            pdr.high,
            result.exit_reason,
            result.realized_r,
            equity,
        )

    logger.info(
        "[SOF ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%% | "
        "Filter: bearish=%d neutral=%d undef=%d",
        len(results), equity,
        (equity / params.initial_equity - 1.0) * 100,
        filter_counts.get(FILTER_BEARISH, 0),
        filter_counts.get("neutral", 0),
        filter_counts.get("undefined", 0),
    )
    return results


class _ParamsAdapter:
    """
    Thin adapter so SOFParams can be passed to prior_day_fade functions
    that expect a PDFadeParams-shaped object.
    All required attributes are forwarded.
    """
    def __init__(self, p: SOFParams):
        self.candle_minutes          = p.candle_minutes
        self.min_pdr_pct             = p.min_pdr_pct
        self.max_pdr_pct             = p.max_pdr_pct
        self.signal_offset_hours     = p.signal_offset_hours
        self.signal_window_end_hours = p.signal_window_end_hours
        self.time_exit_hours         = p.time_exit_hours
        self.min_overshoot_pct       = p.min_overshoot_pct
        self.spread_price            = p.spread_price
        self.slippage_price          = p.slippage_price
        self.stop_buffer_floor_pct   = p.stop_buffer_floor_pct
        self.risk_pct                = p.risk_pct
        self.initial_equity          = p.initial_equity
        self.min_session_candles     = p.min_session_candles
        # tp_r_multiplier not used by compute_pd_fade_setup (TP is structural: PDL)
