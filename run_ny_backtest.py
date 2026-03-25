"""
Entry point for the NY Morning Initial Balance Breakout backtest.

Usage:
    uv run python run_ny_backtest.py <data_file.csv> [results.json]

Example:
    uv run python run_ny_backtest.py data/xauusd_15m_2018-01-01_2021-12-31.csv results/ny_is.json
    uv run python run_ny_backtest.py data/xauusd_15m_2022-01-01_2024-12-31.csv results/ny_oos.json
"""

import logging
import sys
from pathlib import Path

from run_backtest import load_ohlcv
from src.strategies.asian_range_breakout.results import (
    BacktestSummary, DirectionSummary, to_json,
    _approx_sharpe, _direction_summary, _log_summary, _max_drawdown_pct, _mean,
)
from src.strategies.asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP,
)
from src.strategies.ny_ib_breakout.engine import run_ny_ib_backtest
from src.strategies.ny_ib_breakout.params import NYIBParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ny_backtest.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path   = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "ny_results.json"

    # ── Parameters ────────────────────────────────────────────────────────────
    params = NYIBParams(
        candle_minutes=15,
        ib_duration_minutes=30,
        ib_signal_window_hours=2.5,
        time_exit_hours_after_ny_open=4.0,
        min_ib_range_pct=0.0008,
        max_ib_range_pct=0.0060,
        spread_price=0.30,
        slippage_price=0.20,
        stop_buffer_floor_pct=0.0003,
        tp_r_multiplier=1.5,
        risk_pct=0.01,
        initial_equity=100_000.0,
    )

    df = load_ohlcv(data_path)
    trade_results = run_ny_ib_backtest(df, params)
    summary = _compute_ny_results(trade_results, params)
    json_str = to_json(summary)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_str)

    logger.info("Results written to %s", output_path)
    print(json_str)


def _compute_ny_results(trade_results, params: NYIBParams) -> BacktestSummary:
    import math
    if not trade_results:
        empty_dir = DirectionSummary(0, 0, 0.0, 0.0, 0.0, 0.0)
        return BacktestSummary(
            strategy_name="XAUUSD_NY_IB_Breakout", version="v1.0_baseline",
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
            params_snapshot=_ny_params_snapshot(params),
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
        strategy_name="XAUUSD_NY_IB_Breakout",
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
        total_return_pct=round((trade_results[-1].equity_after / params.initial_equity - 1.0) * 100.0, 4),
        max_drawdown_pct=round(_max_drawdown_pct(equity_curve), 4),
        sharpe_approx=round(_approx_sharpe(r_values), 4),
        params_snapshot=_ny_params_snapshot(params),
        passed_min_trade_count=(n >= 200),
        passed_oos_expectancy=None,
    )
    _log_summary(summary)
    return summary


def _ny_params_snapshot(params: NYIBParams) -> dict:
    return {
        "candle_minutes": params.candle_minutes,
        "ib_duration_minutes": params.ib_duration_minutes,
        "ib_signal_window_hours": params.ib_signal_window_hours,
        "time_exit_hours_after_ny_open": params.time_exit_hours_after_ny_open,
        "min_ib_range_pct": params.min_ib_range_pct,
        "max_ib_range_pct": params.max_ib_range_pct,
        "spread_price": params.spread_price,
        "slippage_price": params.slippage_price,
        "stop_buffer_floor_pct": params.stop_buffer_floor_pct,
        "tp_r_multiplier": params.tp_r_multiplier,
        "risk_pct": params.risk_pct,
        "initial_equity": params.initial_equity,
    }


if __name__ == "__main__":
    main()
