"""
Execution and risk layer.

Responsibilities:
  1. Compute effective entry price (next-candle open + spread/slippage).
  2. Place stop loss and take profit at principled levels.
  3. Size the position based on fixed fractional risk.
  4. Simulate the trade candle-by-candle and return the result.

Spread/cost model:
  All costs are embedded directly into entry and exit prices.
  This avoids a separate cost ledger and makes R calculations self-consistent.

  For a LONG trade:
    effective_entry = next_candle_open + half_spread + slippage
    effective_sl    = sl_gross_level - half_spread
    effective_tp    = tp_gross_level - half_spread
    effective_time  = time_exit_candle_open - half_spread

  For a SHORT trade:
    effective_entry = next_candle_open - half_spread - slippage
    effective_sl    = sl_gross_level + half_spread
    effective_tp    = tp_gross_level + half_spread
    effective_time  = time_exit_candle_open + half_spread

  direction * half_spread always works out because:
    Long  (+1): buy high (add spread), sell low (subtract spread)
    Short (-1): sell low (subtract spread), buy high (add spread)

  Formulas in code use `+ direction * value` or `- direction * value` consistently.

Stop buffer:
  stop_buffer = max(1.5 * spread_price, stop_buffer_floor_pct * effective_entry)
  This ensures the buffer is never swallowed by spread noise.

Position sizing:
  risk_amount  = current_equity * risk_pct
  stop_distance = |effective_entry - effective_sl|
  position_size = risk_amount / stop_distance   (in ounces of gold)

  A stop hit should produce realized_r ≈ -1.0.
  A TP hit should produce realized_r ≈ +tp_r_multiplier.

Known limitation (Logic Risk #3):
  When both SL and TP levels are breached within the same candle, we cannot
  determine order from OHLC data alone.  We assume SL fired first (conservative).
  This systematically understates performance by an unknown amount.

Known limitation (Logic Risk #2):
  Gap entries (when the entry candle opens well past the breakout level) are
  accepted as-is.  Position size is not adjusted for the reduced structural
  stop distance.  This is flagged in the log.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .params import StrategyParams
from .signal import AsianRange, BreakoutSignal

logger = logging.getLogger(__name__)

# ── Exit reason constants ─────────────────────────────────────────────────────
EXIT_TP = "TP"
EXIT_SL = "SL"
EXIT_SL_GAP = "SL_GAP"
EXIT_TIME = "TIME"
EXIT_END_OF_DATA = "END_OF_DATA"


@dataclass
class TradeSetup:
    """
    Fully specified trade, computed before entry.

    All prices are effective (post-cost).
    sl_gross and tp_gross are the raw price levels checked against candle OHLC.
    They include the half-spread so that when the exit occurs, the effective
    exit price equals the intended R multiple.
    """
    direction: int                  # +1 long, -1 short
    entry_price: float              # Effective fill (after spread + slippage)
    sl_gross: float                 # Candle level at which SL triggers
    tp_gross: float                 # Candle level at which TP triggers
    effective_sl: float             # sl_gross adjusted for exit spread cost
    effective_tp: float             # tp_gross adjusted for exit spread cost
    stop_distance: float            # |entry_price - effective_sl|  (USD)
    position_size: float            # Ounces of gold
    risk_amount: float              # USD risked on this trade
    entry_timestamp: pd.Timestamp
    signal_timestamp: pd.Timestamp
    asian_range: AsianRange


@dataclass
class TradeResult:
    """
    Outcome of a completed trade.  All PnL figures are net (costs embedded).
    """
    setup: TradeSetup
    exit_price: float               # Effective fill at exit (post-cost)
    exit_reason: str                # EXIT_* constant
    exit_timestamp: pd.Timestamp
    net_pnl: float                  # USD net P&L
    realized_r: float               # net_pnl / risk_amount
    equity_after: float             # Running equity after this trade


def compute_trade_setup(
    signal: BreakoutSignal,
    entry_candle: pd.Series,
    entry_timestamp: pd.Timestamp,
    asian_range: AsianRange,
    equity: float,
    params: StrategyParams,
) -> Optional[TradeSetup]:
    """
    Build a TradeSetup from a signal and the entry candle (next candle after signal).

    Entry is at the OPEN of the entry candle (not the close of the signal candle).
    This models the next-candle-open fill assumption.

    Returns None if the setup is invalid (zero or negative stop distance).
    """
    d = signal.direction
    half_spread = params.spread_price / 2.0

    # ── Effective entry ───────────────────────────────────────────────────────
    gross_entry = float(entry_candle["open"])
    effective_entry = gross_entry + d * (half_spread + params.slippage_price)

    # ── Log gap entry warning ─────────────────────────────────────────────────
    # A gap entry means the market opened well beyond the breakout level.
    gap_pct = abs(gross_entry - signal.breakout_level) / signal.breakout_level
    if gap_pct > 0.002:   # > 0.2% gap from breakout level
        logger.warning(
            "[EXECUTION] Gap entry detected | "
            "BreakoutLevel=%.2f  EntryOpen=%.2f  Gap=%.4f%%  Dir=%+d | "
            "Stop distance may be reduced. Position size not adjusted.",
            signal.breakout_level, gross_entry, gap_pct * 100, d,
        )

    # ── Stop loss ─────────────────────────────────────────────────────────────
    # Placed at the other side of the Asian range + buffer.
    # Buffer ensures stop is not placed inside the spread.
    stop_buffer = max(
        1.5 * params.spread_price,
        params.stop_buffer_floor_pct * effective_entry,
    )
    sl_gross = (asian_range.low if d == 1 else asian_range.high) - d * stop_buffer
    effective_sl = sl_gross - d * half_spread   # cost of exiting at SL

    stop_distance = abs(effective_entry - effective_sl)

    if stop_distance <= 0.0:
        logger.error(
            "[EXECUTION] Invalid stop distance (%.5f). "
            "entry=%.2f  sl=%.2f  direction=%+d. Skipping trade.",
            stop_distance, effective_entry, effective_sl, d,
        )
        return None

    # ── Take profit ───────────────────────────────────────────────────────────
    # tp_gross is the raw price level that triggers the TP check.
    # It is offset by half_spread so that effective_tp = entry + stop_distance * R.
    effective_tp = effective_entry + d * stop_distance * params.tp_r_multiplier
    tp_gross = effective_tp + d * half_spread   # must reach this level to net the R target

    # ── Position size ─────────────────────────────────────────────────────────
    risk_amount = equity * params.risk_pct
    position_size = risk_amount / stop_distance

    logger.info(
        "[EXECUTION] Setup | Dir=%s | GrossEntry=%.2f  EffEntry=%.2f | "
        "SL_gross=%.2f  TP_gross=%.2f | StopDist=%.3f | "
        "Size=%.4foz | Risk=$%.2f | Equity=$%.2f",
        "LONG" if d == 1 else "SHORT",
        gross_entry, effective_entry,
        sl_gross, tp_gross,
        stop_distance,
        position_size, risk_amount, equity,
    )

    return TradeSetup(
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
        signal_timestamp=signal.signal_candle_ts,
        asian_range=asian_range,
    )


def simulate_trade(
    setup: TradeSetup,
    candles_after_entry: pd.DataFrame,
    time_exit_ts: pd.Timestamp,
    equity_before: float,
    params: StrategyParams,
) -> TradeResult:
    """
    Simulate a trade candle-by-candle until an exit condition is met.

    Exit priority (checked in order for each candle):
      1. Time exit:      candle open timestamp >= time_exit_ts → exit at candle open
      2. Gap-through SL: candle open is beyond the SL level   → fill at candle open
      3. SL hit:         candle extreme breaches sl_gross
      4. TP hit:         candle extreme breaches tp_gross
      5. Both in candle: SL assumed first (conservative — Logic Risk #3)

    If candles are exhausted without an exit (end of data), exit at the last
    candle close and log END_OF_DATA.

    All exit prices are effective (post-spread).

    Args:
        setup:               TradeSetup as returned by compute_trade_setup.
        candles_after_entry: Candles starting AFTER the entry candle.
        time_exit_ts:        Hard exit timestamp (UTC-aware pd.Timestamp).
        equity_before:       Account equity before this trade.
        params:              StrategyParams (spread used for exit cost).

    Returns:
        TradeResult with all outcome fields populated.
    """
    d = setup.direction
    half_spread = params.spread_price / 2.0

    for ts, candle in candles_after_entry.iterrows():
        o = float(candle["open"])
        h = float(candle["high"])
        lo = float(candle["low"])

        # ── 1. Time exit ──────────────────────────────────────────────────────
        if ts >= time_exit_ts:
            exit_eff = o - d * half_spread
            logger.info(
                "[EXIT] TIME | Candle=%s | Open=%.2f  EffExit=%.2f",
                ts, o, exit_eff,
            )
            return _build_result(setup, exit_eff, EXIT_TIME, ts, equity_before)

        # ── 2. Gap-through stop ───────────────────────────────────────────────
        gap_through = (d == 1 and o < setup.sl_gross) or (d == -1 and o > setup.sl_gross)
        if gap_through:
            exit_eff = o - d * half_spread
            logger.info(
                "[EXIT] SL_GAP | Candle=%s | Open=%.2f < SL_gross=%.2f | EffExit=%.2f",
                ts, o, setup.sl_gross, exit_eff,
            )
            return _build_result(setup, exit_eff, EXIT_SL_GAP, ts, equity_before)

        # ── 3 & 4. SL and TP checks ───────────────────────────────────────────
        sl_hit = (d == 1 and lo <= setup.sl_gross) or (d == -1 and h >= setup.sl_gross)
        tp_hit = (d == 1 and h >= setup.tp_gross) or (d == -1 and lo <= setup.tp_gross)

        if sl_hit and tp_hit:
            # Cannot determine order from OHLC. Conservative: SL fires first.
            exit_eff = setup.effective_sl
            logger.info(
                "[EXIT] SL (both hit same candle — conservative) | Candle=%s | "
                "SL_gross=%.2f  TP_gross=%.2f  EffExit=%.2f",
                ts, setup.sl_gross, setup.tp_gross, exit_eff,
            )
            return _build_result(setup, exit_eff, EXIT_SL, ts, equity_before)

        if sl_hit:
            exit_eff = setup.effective_sl
            logger.info(
                "[EXIT] SL | Candle=%s | Low=%.2f <= SL_gross=%.2f | EffExit=%.2f",
                ts, lo, setup.sl_gross, exit_eff,
            )
            return _build_result(setup, exit_eff, EXIT_SL, ts, equity_before)

        if tp_hit:
            exit_eff = setup.effective_tp
            logger.info(
                "[EXIT] TP | Candle=%s | High=%.2f >= TP_gross=%.2f | EffExit=%.2f",
                ts, h, setup.tp_gross, exit_eff,
            )
            return _build_result(setup, exit_eff, EXIT_TP, ts, equity_before)

    # ── End-of-data fallback ─────────────────────────────────────────────────
    if candles_after_entry.empty:
        raise RuntimeError(
            "simulate_trade called with empty candles_after_entry. "
            "Caller must guard against this before calling."
        )

    last_ts = candles_after_entry.index[-1]
    last_close = float(candles_after_entry.iloc[-1]["close"])
    exit_eff = last_close - d * half_spread
    logger.warning(
        "[EXIT] END_OF_DATA | LastCandle=%s | Close=%.2f  EffExit=%.2f | "
        "Trade was still open at end of data. This result may be unreliable.",
        last_ts, last_close, exit_eff,
    )
    return _build_result(setup, exit_eff, EXIT_END_OF_DATA, last_ts, equity_before)


def _build_result(
    setup: TradeSetup,
    exit_price: float,
    exit_reason: str,
    exit_timestamp: pd.Timestamp,
    equity_before: float,
) -> TradeResult:
    """
    Compute PnL and R-multiple from the effective exit price.

    All costs are embedded in setup.entry_price and exit_price.
    net_pnl = (exit_price - entry_price) * direction * position_size
    realized_r = net_pnl / risk_amount
    """
    d = setup.direction
    net_pnl = (exit_price - setup.entry_price) * d * setup.position_size
    realized_r = net_pnl / setup.risk_amount if setup.risk_amount > 0.0 else 0.0
    equity_after = equity_before + net_pnl

    logger.info(
        "[RESULT] %s | Entry=%.2f  Exit=%.2f | PnL=$%.2f | R=%.3f | "
        "EquityAfter=$%.2f",
        exit_reason,
        setup.entry_price, exit_price,
        net_pnl, realized_r, equity_after,
    )

    return TradeResult(
        setup=setup,
        exit_price=exit_price,
        exit_reason=exit_reason,
        exit_timestamp=exit_timestamp,
        net_pnl=net_pnl,
        realized_r=realized_r,
        equity_after=equity_after,
    )
