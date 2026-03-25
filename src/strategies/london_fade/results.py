"""
Results aggregation for the London False Breakout Fade strategy.

Reuses BacktestSummary, DirectionSummary, to_json, and all math helpers
from the breakout results module.  Only the params snapshot differs.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from ..asian_range_breakout.results import (
    BacktestSummary,
    DirectionSummary,
    to_json,
    _approx_sharpe,
    _direction_summary,
    _log_summary,
    _max_drawdown_pct,
    _mean,
)
from ..asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP, TradeResult,
)
from .params import FadeParams

logger = logging.getLogger(__name__)


def compute_fade_results(
    trade_results: List[TradeResult],
    params: FadeParams,
    strategy_name: str = "XAUUSD_LondonFade_FalseBreakout",
    version: str = "v1.0_baseline",
) -> BacktestSummary:
    """
    Compute all summary metrics from a list of fade TradeResults.

    Returns a BacktestSummary with passed_oos_expectancy=None.
    """
    if not trade_results:
        logger.warning("[FADE RESULTS] No trades to summarise.")
        return _empty_fade_summary(strategy_name, version, params)

    n = len(trade_results)
    r_values = [t.realized_r for t in trade_results]

    # ── Exit reason counts ────────────────────────────────────────────────────
    tp_count        = sum(1 for t in trade_results if t.exit_reason == EXIT_TP)
    sl_count        = sum(1 for t in trade_results if t.exit_reason == EXIT_SL)
    sl_gap_count    = sum(1 for t in trade_results if t.exit_reason == EXIT_SL_GAP)
    time_exit_count = sum(1 for t in trade_results if t.exit_reason == EXIT_TIME)
    eod_count       = sum(1 for t in trade_results if t.exit_reason == EXIT_END_OF_DATA)

    # ── Win / loss ────────────────────────────────────────────────────────────
    wins   = [r for r in r_values if r > 0.0]
    losses = [r for r in r_values if r <= 0.0]
    win_rate = len(wins) / n

    avg_win_r    = _mean(wins)
    avg_loss_r   = _mean(losses)
    expectancy_r = _mean(r_values)
    sum_wins     = sum(wins)
    sum_losses   = abs(sum(losses))
    profit_factor = (sum_wins / sum_losses) if sum_losses > 0.0 else float("inf")

    sorted_r = sorted(r_values)
    median_r = sorted_r[n // 2]

    # ── Direction split ───────────────────────────────────────────────────────
    long_summary  = _direction_summary([t for t in trade_results if t.setup.direction ==  1])
    short_summary = _direction_summary([t for t in trade_results if t.setup.direction == -1])

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
        params_snapshot=_fade_params_snapshot(params),
        passed_min_trade_count=(n >= 200),
        passed_oos_expectancy=None,
    )

    _log_summary(summary)
    return summary


def _fade_params_snapshot(params: FadeParams) -> Dict[str, Any]:
    return {
        "candle_minutes": params.candle_minutes,
        "london_window_duration_hours": params.london_window_duration_hours,
        "time_exit_hours_after_london_open": params.time_exit_hours_after_london_open,
        "min_range_pct": params.min_range_pct,
        "max_range_pct": params.max_range_pct,
        "min_overshoot_pct": params.min_overshoot_pct,
        "spread_price": params.spread_price,
        "slippage_price": params.slippage_price,
        "stop_buffer_floor_pct": params.stop_buffer_floor_pct,
        "risk_pct": params.risk_pct,
        "initial_equity": params.initial_equity,
    }


def _empty_fade_summary(
    name: str, version: str, params: FadeParams
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
        params_snapshot=_fade_params_snapshot(params),
        passed_min_trade_count=False,
        passed_oos_expectancy=None,
    )
