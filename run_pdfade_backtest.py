"""
Entry point for the Prior Day High/Low Sweep-and-Rejection Fade backtest.

Usage:
    uv run python run_pdfade_backtest.py <data_file.csv> [results.json]

Example:
    uv run python run_pdfade_backtest.py data/xauusd_15m_2018-01-01_2021-12-31.csv results/pdfade_is.json
    uv run python run_pdfade_backtest.py data/xauusd_15m_2022-01-01_2024-12-31.csv results/pdfade_oos.json

Output includes: full metrics, long/short breakdown, yearly breakdown, R distribution,
TP/SL/TIME split.
"""

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

from run_backtest import load_ohlcv
from src.strategies.asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP, TradeResult,
)
from src.strategies.asian_range_breakout.results import (
    BacktestSummary, DirectionSummary,
    _approx_sharpe, _direction_summary, _log_summary, _max_drawdown_pct, _mean,
)
from src.strategies.prior_day_fade.engine import run_pd_fade_backtest
from src.strategies.prior_day_fade.params import PDFadeParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pdfade_backtest.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path   = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "pdfade_results.json"

    params = PDFadeParams(
        candle_minutes=15,
        signal_offset_hours=14.0,       # 07:00 UTC next day (London open)
        signal_window_end_hours=23.0,   # 16:00 UTC next day (NY afternoon)
        time_exit_hours=23.5,           # 16:30 UTC
        min_pdr_pct=0.0030,
        max_pdr_pct=0.0200,
        min_overshoot_pct=0.0002,
        spread_price=0.30,
        slippage_price=0.20,
        stop_buffer_floor_pct=0.0003,
        risk_pct=0.01,
        initial_equity=100_000.0,
    )

    df = load_ohlcv(data_path)
    trade_results = run_pd_fade_backtest(df, params)

    summary   = _compute_results(trade_results, params)
    yearly    = _yearly_breakdown(trade_results)
    r_dist    = _r_distribution(trade_results)
    full_out  = _assemble_output(summary, yearly, r_dist)

    json_str = json.dumps(full_out, indent=2)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_str)
    logger.info("Results written to %s", output_path)
    print(json_str)


# ── Results computation ────────────────────────────────────────────────────────

def _compute_results(
    trade_results: list[TradeResult],
    params: PDFadeParams,
    name: str = "XAUUSD_PriorDay_SweepRejection_Fade",
    version: str = "v1.0_baseline",
) -> BacktestSummary:
    if not trade_results:
        empty_dir = DirectionSummary(0, 0, 0.0, 0.0, 0.0, 0.0)
        return BacktestSummary(
            strategy_name=name, version=version,
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
            params_snapshot=_params_snapshot(params),
            passed_min_trade_count=False, passed_oos_expectancy=None,
        )

    n = len(trade_results)
    r_values = [t.realized_r for t in trade_results]
    wins   = [r for r in r_values if r > 0.0]
    losses = [r for r in r_values if r <= 0.0]
    sum_losses = abs(sum(losses))
    profit_factor = (sum(wins) / sum_losses) if sum_losses > 0.0 else float("inf")
    equity_curve  = [params.initial_equity] + [t.equity_after for t in trade_results]

    summary = BacktestSummary(
        strategy_name=name, version=version,
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
        params_snapshot=_params_snapshot(params),
        passed_min_trade_count=(n >= 200),
        passed_oos_expectancy=None,
    )
    _log_summary(summary)
    return summary


def _yearly_breakdown(trade_results: list[TradeResult]) -> dict:
    """Group trades by year of entry and compute per-year metrics."""
    by_year: dict[int, list[TradeResult]] = defaultdict(list)
    for t in trade_results:
        yr = t.setup.entry_timestamp.year
        by_year[yr].append(t)

    yearly = {}
    for yr in sorted(by_year):
        trades = by_year[yr]
        r_vals = [t.realized_r for t in trades]
        wins   = [r for r in r_vals if r > 0.0]
        losses = [r for r in r_vals if r <= 0.0]
        tp  = sum(1 for t in trades if t.exit_reason == EXIT_TP)
        sl  = sum(1 for t in trades if t.exit_reason in (EXIT_SL, EXIT_SL_GAP))
        te  = sum(1 for t in trades if t.exit_reason == EXIT_TIME)
        pf  = (sum(wins) / abs(sum(losses))) if losses else float("inf")
        yearly[str(yr)] = {
            "trades":       len(trades),
            "win_rate":     round(len(wins) / len(trades), 4),
            "expectancy_r": round(_mean(r_vals), 4),
            "profit_factor": round(pf, 4) if not math.isinf(pf) else "inf",
            "avg_win_r":    round(_mean(wins), 4),
            "avg_loss_r":   round(_mean(losses), 4),
            "tp_sl_time":   f"{tp}/{sl}/{te}",
            "long_count":   sum(1 for t in trades if t.setup.direction ==  1),
            "short_count":  sum(1 for t in trades if t.setup.direction == -1),
        }
    return yearly


def _r_distribution(trade_results: list[TradeResult]) -> dict:
    """Compute R-value distribution statistics and bucket counts."""
    if not trade_results:
        return {}

    r_vals = sorted(t.realized_r for t in trade_results)
    n = len(r_vals)

    # Percentiles
    def pct(p):
        idx = int(p / 100 * (n - 1))
        return round(r_vals[idx], 4)

    # Bucket counts: <-1, [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1), [1,1.5), >=1.5
    buckets = {
        "lt_neg1":       sum(1 for r in r_vals if r < -1.0),
        "neg1_to_neg05": sum(1 for r in r_vals if -1.0 <= r < -0.5),
        "neg05_to_0":    sum(1 for r in r_vals if -0.5 <= r < 0.0),
        "0_to_05":       sum(1 for r in r_vals if  0.0 <= r < 0.5),
        "05_to_1":       sum(1 for r in r_vals if  0.5 <= r < 1.0),
        "1_to_15":       sum(1 for r in r_vals if  1.0 <= r < 1.5),
        "gte_15":        sum(1 for r in r_vals if r >= 1.5),
    }

    return {
        "percentiles": {
            "p5":  pct(5),  "p10": pct(10), "p25": pct(25),
            "p50": pct(50), "p75": pct(75), "p90": pct(90), "p95": pct(95),
        },
        "mean":   round(sum(r_vals) / n, 4),
        "std":    round(math.sqrt(sum((r - sum(r_vals)/n)**2 for r in r_vals) / (n-1)), 4) if n > 1 else 0.0,
        "min":    round(r_vals[0], 4),
        "max":    round(r_vals[-1], 4),
        "buckets": buckets,
    }


def _assemble_output(summary: BacktestSummary, yearly: dict, r_dist: dict) -> dict:
    """Combine BacktestSummary, yearly breakdown, and R distribution into one dict."""
    from dataclasses import asdict
    import math as _math

    def _clean(obj):
        if isinstance(obj, float):
            if _math.isinf(obj): return "Infinity"
            if _math.isnan(obj): return "NaN"
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        return obj

    out = _clean(asdict(summary))
    out["yearly_breakdown"] = yearly
    out["r_distribution"]   = r_dist
    return out


def _params_snapshot(params: PDFadeParams) -> dict:
    return {
        "candle_minutes": params.candle_minutes,
        "session_boundary": "17:00 UTC",
        "signal_window": f"+{params.signal_offset_hours:.0f}h to +{params.signal_window_end_hours:.0f}h after session start",
        "time_exit": f"+{params.time_exit_hours}h after session start",
        "min_pdr_pct": params.min_pdr_pct,
        "max_pdr_pct": params.max_pdr_pct,
        "min_overshoot_pct": params.min_overshoot_pct,
        "spread_price": params.spread_price,
        "slippage_price": params.slippage_price,
        "stop_buffer_floor_pct": params.stop_buffer_floor_pct,
        "tp": "opposite_prior_day_boundary (structural)",
        "risk_pct": params.risk_pct,
        "initial_equity": params.initial_equity,
    }


if __name__ == "__main__":
    main()
