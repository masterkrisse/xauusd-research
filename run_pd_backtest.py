"""
Entry point for the Prior Day High/Low Breakout backtest.

Usage:
    uv run python run_pd_backtest.py <data_file.csv> [results.json]

Example:
    uv run python run_pd_backtest.py data/xauusd_15m_2018-01-01_2021-12-31.csv results/pd_is.json
    uv run python run_pd_backtest.py data/xauusd_15m_2022-01-01_2024-12-31.csv results/pd_oos.json
"""

import logging
import math
import sys
from pathlib import Path

from run_backtest import load_ohlcv
from src.strategies.asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP,
)
from src.strategies.asian_range_breakout.results import (
    BacktestSummary, DirectionSummary, to_json,
    _approx_sharpe, _direction_summary, _log_summary, _max_drawdown_pct, _mean,
)
from src.strategies.prior_day_breakout.engine import run_pd_breakout_backtest
from src.strategies.prior_day_breakout.params import PDBreakoutParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pd_backtest.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path   = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "pd_results.json"

    params = PDBreakoutParams(
        candle_minutes=15,
        signal_window_start_utc=7,
        signal_window_end_utc=20,
        time_exit_utc_hour=21,
        min_pdr_pct=0.0030,
        max_pdr_pct=0.0200,
        spread_price=0.30,
        slippage_price=0.20,
        stop_buffer_floor_pct=0.0003,
        tp_r_multiplier=1.5,
        risk_pct=0.01,
        initial_equity=100_000.0,
    )

    df = load_ohlcv(data_path)
    trade_results = run_pd_breakout_backtest(df, params)
    summary = _compute_pd_results(trade_results, params)

    json_str = to_json(summary)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_str)
    logger.info("Results written to %s", output_path)
    print(json_str)


def _compute_pd_results(trade_results, params: PDBreakoutParams) -> BacktestSummary:
    if not trade_results:
        empty_dir = DirectionSummary(0, 0, 0.0, 0.0, 0.0, 0.0)
        return BacktestSummary(
            strategy_name="XAUUSD_PriorDay_Breakout", version="v1.0_baseline",
            start_date="", end_date="",
            total_trades=0, tp_count=0, sl_count=0, sl_gap_count=0,
            time_exit_count=0, end_of_data_count=0,
            win_count=0, loss_count=0, win_rate=0.0,
            avg_win_r=0.0, avg_loss_r=0.0, expectancy_r=0.0,
            profit_factor=0.0, median_r=0.0, best_trade_r=0.0, worst_trade_r=0.0,
            long_summary=empty_dir, short_summary=empty_dir,
            net_pnl_usd=0.0, initial_equity=params.initial_equity,
            final_equity=params.initial_equity,
            total_return_pct=0.0, max_drawdown_pct=0.0, sharpe_approx=0.0,
            params_snapshot=_pd_params_snapshot(params),
            passed_min_trade_count=False, passed_oos_expectancy=None,
        )

    n = len(trade_results)
    r_values = [t.realized_r for t in trade_results]
    wins   = [r for r in r_values if r > 0.0]
    losses = [r for r in r_values if r <= 0.0]
    sum_losses = abs(sum(losses))
    profit_factor = (sum(wins) / sum_losses) if sum_losses > 0.0 else float("inf")
    equity_curve = [params.initial_equity] + [t.equity_after for t in trade_results]

    summary = BacktestSummary(
        strategy_name="XAUUSD_PriorDay_Breakout",
        version="v1.0_baseline",
        start_date=str(trade_results[0].setup.entry_timestamp.date()),
        end_date=str(trade_results[-1].exit_timestamp.date()),
        total_trades=n,
        tp_count=sum(1 for t in trade_results if t.exit_reason == EXIT_TP),
        sl_count=sum(1 for t in trade_results if t.exit_reason == EXIT_SL),
        sl_gap_count=sum(1 for t in trade_results if t.exit_reason == EXIT_SL_GAP),
        time_exit_count=sum(1 for t in trade_results if t.exit_reason == EXIT_TIME),
        end_of_data_count=sum(1 for t in trade_results if t.exit_reason == EXIT_END_OF_DATA),
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=round(len(wins) / n, 4),
        avg_win_r=round(_mean(wins), 4),
        avg_loss_r=round(_mean(losses), 4),
        expectancy_r=round(_mean(r_values), 4),
        profit_factor=round(profit_factor, 4) if not math.isinf(profit_factor) else "inf",
        median_r=round(sorted(r_values)[n // 2], 4),
        best_trade_r=round(max(r_values), 4),
        worst_trade_r=round(min(r_values), 4),
        long_summary=_direction_summary([t for t in trade_results if t.setup.direction == 1]),
        short_summary=_direction_summary([t for t in trade_results if t.setup.direction == -1]),
        net_pnl_usd=round(sum(t.net_pnl for t in trade_results), 2),
        initial_equity=params.initial_equity,
        final_equity=round(trade_results[-1].equity_after, 2),
        total_return_pct=round(
            (trade_results[-1].equity_after / params.initial_equity - 1.0) * 100.0, 4
        ),
        max_drawdown_pct=round(_max_drawdown_pct(equity_curve), 4),
        sharpe_approx=round(_approx_sharpe(r_values), 4),
        params_snapshot=_pd_params_snapshot(params),
        passed_min_trade_count=(n >= 200),
        passed_oos_expectancy=None,
    )
    _log_summary(summary)
    return summary


def _pd_params_snapshot(params: PDBreakoutParams) -> dict:
    return {
        "candle_minutes": params.candle_minutes,
        "signal_window_utc": f"{params.signal_window_start_utc:02d}:00-{params.signal_window_end_utc:02d}:00",
        "time_exit_utc_hour": params.time_exit_utc_hour,
        "min_pdr_pct": params.min_pdr_pct,
        "max_pdr_pct": params.max_pdr_pct,
        "spread_price": params.spread_price,
        "slippage_price": params.slippage_price,
        "stop_buffer_floor_pct": params.stop_buffer_floor_pct,
        "tp_r_multiplier": params.tp_r_multiplier,
        "risk_pct": params.risk_pct,
        "initial_equity": params.initial_equity,
    }


if __name__ == "__main__":
    main()
