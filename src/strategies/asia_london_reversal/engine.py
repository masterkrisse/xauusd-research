"""
Asia–London Session Reversal backtest engine.

Protocol per session:
  1. Extract Asian session candles [17:00, 07:00) UTC.
  2. Compute net move (close-to-open %).
  3. If |net_pct| >= min_asian_move_pct, generate a reversal signal.
  4. Enter at the open of the 07:00 UTC candle (London open).
  5. Stop: beyond the Asian session extreme.
  6. TP: 50% retrace of the Asian net move = (asian_open + asian_close) / 2.
  7. Exit on TP, SL, or time (11:00 UTC), whichever comes first.

One trade per session maximum.

Execution cost model (consistent with prior strategies):
  LONG entry  : candle_open + half_spread + slippage
  SHORT entry : candle_open - half_spread - slippage
  SL exit     : sl_gross level ± half_spread
  TP exit     : tp_gross level ± half_spread
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
from .params import ALParams
from .signal import ALSignal, AsianSessionInfo, compute_asian_session, detect_al_signal

logger = logging.getLogger(__name__)


# ── Trade dataclasses ──────────────────────────────────────────────────────────

@dataclass
class ALTradeSetup:
    direction: int
    entry_price: float        # effective (post-cost)
    sl_gross: float           # candle level that triggers SL check
    tp_gross: float           # candle level that triggers TP check
    effective_sl: float       # net exit price at SL
    effective_tp: float       # net exit price at TP
    stop_distance: float      # |entry_price - effective_sl|
    position_size: float      # oz
    risk_amount: float        # USD risked
    entry_timestamp: pd.Timestamp
    time_exit_ts: pd.Timestamp
    signal: ALSignal


@dataclass
class ALTradeResult:
    setup: ALTradeSetup
    exit_price: float
    exit_reason: str
    exit_timestamp: pd.Timestamp
    net_pnl: float
    realized_r: float
    equity_after: float


# ── Engine ────────────────────────────────────────────────────────────────────

def run_al_reversal_backtest(
    df: pd.DataFrame,
    params: ALParams,
) -> List[ALTradeResult]:
    """
    Run the Asia–London Reversal backtest over the full dataset.

    Returns a list of ALTradeResult, one per executed trade.
    """
    params.validate()

    equity = params.initial_equity
    results: List[ALTradeResult] = []

    session_starts = _extract_session_starts(df)

    logger.info(
        "[AL ENGINE] Start | Sessions=%d | Bars=%d | %s → %s | Equity=$%.2f",
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
            "[AL ENGINE] Trade | %s | Dir=%s | %s | R=%.3f | Equity=$%.2f",
            session_start.strftime("%Y-%m-%d"),
            "LONG" if result.setup.direction == 1 else "SHORT",
            result.exit_reason,
            result.realized_r,
            equity,
        )

    logger.info(
        "[AL ENGINE] Complete | Trades=%d | FinalEquity=$%.2f | Return=%.2f%%",
        len(results), equity, (equity / params.initial_equity - 1.0) * 100,
    )
    return results


# ── Session processing ────────────────────────────────────────────────────────

def _process_session(
    df: pd.DataFrame,
    session_start: pd.Timestamp,
    equity: float,
    params: ALParams,
) -> Optional[ALTradeResult]:

    # ── 1. Asian session ──────────────────────────────────────────────────────
    asian = compute_asian_session(df, session_start, params)
    if not asian.valid:
        return None

    # ── 2. Signal ─────────────────────────────────────────────────────────────
    signal = detect_al_signal(asian, params)
    if signal is None:
        return None

    # ── 3. Entry candle ───────────────────────────────────────────────────────
    london_open_ts = signal.london_open_ts
    time_exit_ts   = session_start + pd.Timedelta(hours=params.london_exit_hours)

    # Collect the full London window for simulation
    session_end_ts = session_start + pd.Timedelta(hours=24)
    london_candles = df[
        (df.index >= london_open_ts) & (df.index < session_end_ts)
    ]

    if london_candles.empty:
        logger.debug("[AL ENGINE] No London candles for %s", session_start.date())
        return None

    entry_candle    = london_candles.iloc[0]
    entry_timestamp = london_candles.index[0]

    if entry_timestamp >= time_exit_ts:
        logger.debug("[AL ENGINE] Entry candle at/after time exit for %s", session_start.date())
        return None

    # ── 4. Trade setup ────────────────────────────────────────────────────────
    setup = _build_setup(signal, entry_candle, entry_timestamp, time_exit_ts, equity, params)
    if setup is None:
        return None

    # ── 5. Simulate ───────────────────────────────────────────────────────────
    candles_after_entry = london_candles[london_candles.index > entry_timestamp]
    if candles_after_entry.empty:
        return None

    return _simulate(setup, candles_after_entry, equity, params)


def _build_setup(
    signal: ALSignal,
    entry_candle: pd.Series,
    entry_timestamp: pd.Timestamp,
    time_exit_ts: pd.Timestamp,
    equity: float,
    params: ALParams,
) -> Optional[ALTradeSetup]:

    d = signal.direction
    half_spread = params.spread_price / 2.0

    # Effective entry: open of London candle + spread + slippage
    gross_entry     = float(entry_candle["open"])
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    # Stop: beyond Asian extreme, with half-spread baked into effective SL
    sl_gross    = signal.stop_level                     # raw price level
    effective_sl = sl_gross - d * half_spread           # net fill at stop

    stop_distance = abs(effective_entry - effective_sl)
    if stop_distance <= 0.0:
        logger.warning(
            "[AL ENGINE] Zero stop distance on %s. Entry=%.2f SL=%.2f. Skip.",
            entry_timestamp, effective_entry, effective_sl,
        )
        return None

    # TP: 50% retrace (midpoint of Asian open and close)
    tp_gross    = signal.tp_level
    effective_tp = tp_gross + d * half_spread           # net fill at TP

    # Validate TP is on the correct side of entry
    if d == 1 and effective_tp <= effective_entry:
        logger.debug(
            "[AL ENGINE] LONG TP=%.2f <= entry=%.2f on %s. Skip (entry gaped through TP).",
            effective_tp, effective_entry, entry_timestamp.date(),
        )
        return None
    if d == -1 and effective_tp >= effective_entry:
        logger.debug(
            "[AL ENGINE] SHORT TP=%.2f >= entry=%.2f on %s. Skip (entry gaped through TP).",
            effective_tp, effective_entry, entry_timestamp.date(),
        )
        return None

    risk_amount   = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    rr = abs(effective_tp - effective_entry) / stop_distance

    logger.info(
        "[AL ENGINE] Setup | %s | Dir=%s | Entry=%.2f | SL=%.2f TP=%.2f | "
        "StopDist=%.3f | R:R=%.2f | Size=%.4foz | Risk=$%.2f",
        entry_timestamp.strftime("%Y-%m-%d %H:%M"),
        "LONG" if d == 1 else "SHORT",
        effective_entry, effective_sl, effective_tp,
        stop_distance, rr, position_size, risk_amount,
    )

    return ALTradeSetup(
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
    setup: ALTradeSetup,
    candles_after_entry: pd.DataFrame,
    equity_before: float,
    params: ALParams,
) -> ALTradeResult:

    d = setup.direction
    half_spread = params.spread_price / 2.0

    for ts, candle in candles_after_entry.iterrows():
        o  = float(candle["open"])
        h  = float(candle["high"])
        lo = float(candle["low"])

        # 1. Time exit
        if ts >= setup.time_exit_ts:
            exit_eff = o - d * half_spread
            return _build_result(setup, exit_eff, EXIT_TIME, ts, equity_before)

        # 2. Gap-through SL
        gap = (d == 1 and o < setup.sl_gross) or (d == -1 and o > setup.sl_gross)
        if gap:
            exit_eff = o - d * half_spread
            return _build_result(setup, exit_eff, EXIT_SL_GAP, ts, equity_before)

        # 3 & 4. SL and TP
        sl_hit = (d == 1 and lo <= setup.sl_gross) or (d == -1 and h >= setup.sl_gross)
        tp_hit = (d == 1 and h >= setup.tp_gross)  or (d == -1 and lo <= setup.tp_gross)

        if sl_hit and tp_hit:
            return _build_result(setup, setup.effective_sl, EXIT_SL, ts, equity_before)
        if sl_hit:
            return _build_result(setup, setup.effective_sl, EXIT_SL, ts, equity_before)
        if tp_hit:
            return _build_result(setup, setup.effective_tp, EXIT_TP, ts, equity_before)

    # End of data
    last_ts    = candles_after_entry.index[-1]
    last_close = float(candles_after_entry.iloc[-1]["close"])
    exit_eff   = last_close - d * half_spread
    logger.warning("[AL ENGINE] END_OF_DATA exit at %s", last_ts)
    return _build_result(setup, exit_eff, EXIT_END_OF_DATA, last_ts, equity_before)


def _build_result(
    setup: ALTradeSetup,
    exit_price: float,
    exit_reason: str,
    exit_timestamp: pd.Timestamp,
    equity_before: float,
) -> ALTradeResult:
    d       = setup.direction
    net_pnl = (exit_price - setup.entry_price) * d * setup.position_size
    r       = net_pnl / setup.risk_amount if setup.risk_amount > 0 else 0.0
    return ALTradeResult(
        setup=setup,
        exit_price=exit_price,
        exit_reason=exit_reason,
        exit_timestamp=exit_timestamp,
        net_pnl=net_pnl,
        realized_r=r,
        equity_after=equity_before + net_pnl,
    )
