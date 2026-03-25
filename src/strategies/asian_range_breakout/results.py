"""
Results aggregation and reporting.

Produces a BacktestSummary dataclass that is:
  - Fully machine-readable (all fields are plain Python types).
  - JSON-serialisable via to_json().
  - Split by direction (long-only, short-only) to expose asymmetry.
  - Annotated with pass/fail flags from the rejection criteria.

Sharpe ratio note (Logic Risk #9):
  The Sharpe here is computed on per-trade R values, annualised by assuming
  ~252 trading days per year.  This is an approximation — it will overstate
  the true Sharpe when fewer than 252 trades occur per year.  It is provided
  as a directional indicator only.  A proper daily-equity Sharpe requires a
  separate daily equity snapshot, which this baseline does not track.

Equity comparison note (Logic Risk #7):
  Dollar PnL and equity figures are only comparable across windows if both
  windows use the same starting equity.  Use expectancy_r and win_rate for
  IS vs OOS comparisons.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .execution import EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP, TradeResult
from .params import StrategyParams

logger = logging.getLogger(__name__)


@dataclass
class DirectionSummary:
    """Metrics broken out for one direction (long or short)."""
    trade_count: int
    win_count: int
    win_rate: float
    expectancy_r: float
    avg_win_r: float
    avg_loss_r: float


@dataclass
class BacktestSummary:
    """
    Machine-readable backtest result.

    Rejection flags:
      passed_min_trade_count:   True if total_trades >= 200.
      passed_oos_expectancy:    Set externally when comparing IS vs OOS windows.
                                None until set.  Pass if OOS >= 0.60 * IS expectancy_r.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    strategy_name: str
    version: str
    start_date: str
    end_date: str

    # ── Trade counts ──────────────────────────────────────────────────────────
    total_trades: int
    tp_count: int
    sl_count: int
    sl_gap_count: int
    time_exit_count: int
    end_of_data_count: int

    # ── Combined metrics ──────────────────────────────────────────────────────
    win_count: int          # Trades with realized_r > 0
    loss_count: int         # Trades with realized_r <= 0
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float     # Mean realized_r per trade (positive = edge)
    profit_factor: float    # sum(positive R) / abs(sum(negative R))
    median_r: float
    best_trade_r: float
    worst_trade_r: float

    # ── Direction breakdown ───────────────────────────────────────────────────
    long_summary: DirectionSummary
    short_summary: DirectionSummary

    # ── Dollar / equity metrics ───────────────────────────────────────────────
    net_pnl_usd: float
    initial_equity: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float     # Max peak-to-trough drawdown as fraction of peak equity

    # ── Approximate Sharpe ───────────────────────────────────────────────────
    sharpe_approx: float        # Per-trade R annualised; see module docstring

    # ── Parameter snapshot ───────────────────────────────────────────────────
    params_snapshot: Dict[str, Any]

    # ── Rejection flags ───────────────────────────────────────────────────────
    passed_min_trade_count: bool        # >= 200 required
    passed_oos_expectancy: Optional[float]   # Set externally; None until compared


def compute_results(
    trade_results: List[TradeResult],
    params: StrategyParams,
    strategy_name: str = "XAUUSD_AsianRange_LondonBreakout",
    version: str = "v1.0_baseline",
) -> BacktestSummary:
    """
    Compute all summary metrics from a list of TradeResults.

    Returns a BacktestSummary with passed_oos_expectancy=None.
    The caller is responsible for setting that field after IS/OOS comparison.
    """
    if not trade_results:
        logger.warning("[RESULTS] No trades to summarise. Returning empty summary.")
        return _empty_summary(strategy_name, version, params)

    n = len(trade_results)
    r_values = [t.realized_r for t in trade_results]

    # ── Exit reason counts ────────────────────────────────────────────────────
    tp_count = sum(1 for t in trade_results if t.exit_reason == EXIT_TP)
    sl_count = sum(1 for t in trade_results if t.exit_reason == EXIT_SL)
    sl_gap_count = sum(1 for t in trade_results if t.exit_reason == EXIT_SL_GAP)
    time_exit_count = sum(1 for t in trade_results if t.exit_reason == EXIT_TIME)
    eod_count = sum(1 for t in trade_results if t.exit_reason == EXIT_END_OF_DATA)

    # ── Win / loss ────────────────────────────────────────────────────────────
    wins = [r for r in r_values if r > 0.0]
    losses = [r for r in r_values if r <= 0.0]
    win_rate = len(wins) / n

    avg_win_r = _mean(wins)
    avg_loss_r = _mean(losses)
    expectancy_r = _mean(r_values)
    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    profit_factor = (sum_wins / sum_losses) if sum_losses > 0.0 else float("inf")

    sorted_r = sorted(r_values)
    median_r = sorted_r[n // 2]

    # ── Direction split ───────────────────────────────────────────────────────
    long_summary = _direction_summary(
        [t for t in trade_results if t.setup.direction == 1]
    )
    short_summary = _direction_summary(
        [t for t in trade_results if t.setup.direction == -1]
    )

    # ── Dollar / equity ───────────────────────────────────────────────────────
    net_pnl = sum(t.net_pnl for t in trade_results)
    final_equity = trade_results[-1].equity_after
    total_return_pct = (final_equity / params.initial_equity - 1.0) * 100.0

    equity_curve = [params.initial_equity] + [t.equity_after for t in trade_results]
    max_dd = _max_drawdown_pct(equity_curve)

    sharpe = _approx_sharpe(r_values)

    summary = BacktestSummary(
        strategy_name=strategy_name,
        version=version,
        start_date=str(trade_results[0].setup.entry_timestamp.date()),
        end_date=str(trade_results[-1].exit_timestamp.date()),
        total_trades=n,
        tp_count=tp_count,
        sl_count=sl_count,
        sl_gap_count=sl_gap_count,
        time_exit_count=time_exit_count,
        end_of_data_count=eod_count,
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=round(win_rate, 4),
        avg_win_r=round(avg_win_r, 4),
        avg_loss_r=round(avg_loss_r, 4),
        expectancy_r=round(expectancy_r, 4),
        profit_factor=round(profit_factor, 4) if not math.isinf(profit_factor) else "inf",
        median_r=round(median_r, 4),
        best_trade_r=round(max(r_values), 4),
        worst_trade_r=round(min(r_values), 4),
        long_summary=long_summary,
        short_summary=short_summary,
        net_pnl_usd=round(net_pnl, 2),
        initial_equity=params.initial_equity,
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 4),
        max_drawdown_pct=round(max_dd, 4),
        sharpe_approx=round(sharpe, 4),
        params_snapshot=_params_snapshot(params),
        passed_min_trade_count=(n >= 200),
        passed_oos_expectancy=None,
    )

    _log_summary(summary)
    return summary


def to_json(summary: BacktestSummary, indent: int = 2) -> str:
    """
    Serialise BacktestSummary to a JSON string.
    Handles float('inf') and float('nan') which are not valid JSON.
    """
    raw = asdict(summary)

    def _clean(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isinf(obj):
                return "Infinity"
            if math.isnan(obj):
                return "NaN"
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    return json.dumps(_clean(raw), indent=indent)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _direction_summary(trades: List[TradeResult]) -> DirectionSummary:
    if not trades:
        return DirectionSummary(0, 0, 0.0, 0.0, 0.0, 0.0)
    r_vals = [t.realized_r for t in trades]
    wins = [r for r in r_vals if r > 0.0]
    losses = [r for r in r_vals if r <= 0.0]
    return DirectionSummary(
        trade_count=len(trades),
        win_count=len(wins),
        win_rate=round(len(wins) / len(trades), 4),
        expectancy_r=round(_mean(r_vals), 4),
        avg_win_r=round(_mean(wins), 4),
        avg_loss_r=round(_mean(losses), 4),
    )


def _max_drawdown_pct(equity_curve: List[float]) -> float:
    """Maximum peak-to-trough drawdown as a fraction of peak equity."""
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0.0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _approx_sharpe(r_values: List[float]) -> float:
    """
    Approximate annualised Sharpe from per-trade R values.
    Assumption: ~252 trades per year.  Likely overstated for low-frequency strategies.
    """
    n = len(r_values)
    if n < 2:
        return 0.0
    mean = sum(r_values) / n
    variance = sum((r - mean) ** 2 for r in r_values) / (n - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(252)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _params_snapshot(params: StrategyParams) -> Dict[str, Any]:
    return {
        "candle_minutes": params.candle_minutes,
        "london_window_duration_hours": params.london_window_duration_hours,
        "time_exit_hours_after_london_open": params.time_exit_hours_after_london_open,
        "min_range_pct": params.min_range_pct,
        "max_range_pct": params.max_range_pct,
        "spread_price": params.spread_price,
        "slippage_price": params.slippage_price,
        "stop_buffer_floor_pct": params.stop_buffer_floor_pct,
        "tp_r_multiplier": params.tp_r_multiplier,
        "risk_pct": params.risk_pct,
        "initial_equity": params.initial_equity,
    }


def _log_summary(s: BacktestSummary) -> None:
    sep = "=" * 65
    logger.info(sep)
    logger.info("[RESULTS] %s  %s", s.strategy_name, s.version)
    logger.info("[RESULTS] Period  : %s → %s", s.start_date, s.end_date)
    logger.info(
        "[RESULTS] Trades  : %d  (TP=%d  SL=%d  SL_GAP=%d  TIME=%d  EOD=%d)",
        s.total_trades, s.tp_count, s.sl_count,
        s.sl_gap_count, s.time_exit_count, s.end_of_data_count,
    )
    logger.info(
        "[RESULTS] WinRate : %.1f%%  (L=%.1f%%  S=%.1f%%)",
        s.win_rate * 100,
        s.long_summary.win_rate * 100,
        s.short_summary.win_rate * 100,
    )
    logger.info(
        "[RESULTS] Expect  : %.4fR  AvgWin=%.4fR  AvgLoss=%.4fR  PF=%.2f",
        s.expectancy_r, s.avg_win_r, s.avg_loss_r, s.profit_factor,
    )
    logger.info(
        "[RESULTS] Long    : trades=%d  exp=%.4fR",
        s.long_summary.trade_count, s.long_summary.expectancy_r,
    )
    logger.info(
        "[RESULTS] Short   : trades=%d  exp=%.4fR",
        s.short_summary.trade_count, s.short_summary.expectancy_r,
    )
    logger.info(
        "[RESULTS] R range : best=%.3f  worst=%.3f  median=%.3f",
        s.best_trade_r, s.worst_trade_r, s.median_r,
    )
    logger.info(
        "[RESULTS] Equity  : $%.2f → $%.2f  (+%.2f%%)  MaxDD=%.2f%%",
        s.initial_equity, s.final_equity,
        s.total_return_pct, s.max_drawdown_pct * 100,
    )
    logger.info(
        "[RESULTS] Sharpe  : %.2f (approx, per-trade R)",
        s.sharpe_approx,
    )
    logger.info(
        "[RESULTS] MinTradeCount(>=200): %s",
        "PASS" if s.passed_min_trade_count else "FAIL",
    )
    logger.info(sep)


def _empty_summary(
    name: str, version: str, params: StrategyParams
) -> BacktestSummary:
    empty_dir = DirectionSummary(0, 0, 0.0, 0.0, 0.0, 0.0)
    return BacktestSummary(
        strategy_name=name, version=version,
        start_date="", end_date="",
        total_trades=0, tp_count=0, sl_count=0,
        sl_gap_count=0, time_exit_count=0, end_of_data_count=0,
        win_count=0, loss_count=0, win_rate=0.0,
        avg_win_r=0.0, avg_loss_r=0.0, expectancy_r=0.0,
        profit_factor=0.0, median_r=0.0,
        best_trade_r=0.0, worst_trade_r=0.0,
        long_summary=empty_dir, short_summary=empty_dir,
        net_pnl_usd=0.0, initial_equity=params.initial_equity,
        final_equity=params.initial_equity,
        total_return_pct=0.0, max_drawdown_pct=0.0,
        sharpe_approx=0.0,
        params_snapshot=_params_snapshot(params),
        passed_min_trade_count=False,
        passed_oos_expectancy=None,
    )
