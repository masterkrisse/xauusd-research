"""
Short-Only Macro-Filtered Fade — IS / OOS backtest entry point.

Usage:
    uv run python run_sof_backtest.py <data_file.csv> [results.json]

Strategy:
  SHORT only. PDH sweep-rejection signal.
  Macro filter: 20-session MA slope < 0 (bearish macro environment).
  Stop: wick high + buffer. TP: PDL. Time exit: 16:30 UTC.
"""

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

from run_backtest import load_ohlcv
from src.strategies.short_only_fade.engine import run_sof_backtest
from src.strategies.short_only_fade.params import SOFParams
from src.strategies.asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP, TradeResult,
)
from src.strategies.asian_range_breakout.results import (
    _approx_sharpe, _max_drawdown_pct, _mean,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sof_backtest.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path   = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "sof_results.json"

    params = SOFParams(
        candle_minutes=15,
        ma_period=20,
        slope_lookback=5,
        min_pdr_pct=0.0030,
        max_pdr_pct=0.0200,
        signal_offset_hours=14.0,
        signal_window_end_hours=23.0,
        time_exit_hours=23.5,
        min_overshoot_pct=0.0002,
        spread_price=0.30,
        slippage_price=0.20,
        stop_buffer_floor_pct=0.0003,
        risk_pct=0.01,
        initial_equity=100_000.0,
    )

    df     = load_ohlcv(data_path)
    trades = run_sof_backtest(df, params)

    summary = _compute_summary(trades, params)
    yearly  = _yearly_breakdown(trades)
    r_dist  = _r_distribution(trades)
    out     = _assemble(summary, yearly, r_dist, params)

    json_str = json.dumps(out, indent=2)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json_str)
    logger.info("Results written to %s", output_path)
    print(json_str)


def _compute_summary(trades: list[TradeResult], params: SOFParams) -> dict:
    if not trades:
        return {"total_trades": 0, "note": "No trades generated"}

    n      = len(trades)
    r_vals = [t.realized_r for t in trades]
    wins   = [r for r in r_vals if r > 0.0]
    losses = [r for r in r_vals if r <= 0.0]
    sum_l  = abs(sum(losses))
    pf     = (sum(wins) / sum_l) if sum_l > 0.0 else float("inf")
    eq_curve = [params.initial_equity] + [t.equity_after for t in trades]

    return {
        "strategy":      "XAUUSD_Short_Only_Macro_Filtered_Fade",
        "version":       "v1.0_baseline",
        "start_date":    str(trades[0].setup.entry_timestamp.date()),
        "end_date":      str(trades[-1].exit_timestamp.date()),
        "total_trades":  n,
        "tp_count":      sum(1 for t in trades if t.exit_reason == EXIT_TP),
        "sl_count":      sum(1 for t in trades if t.exit_reason == EXIT_SL),
        "sl_gap_count":  sum(1 for t in trades if t.exit_reason == EXIT_SL_GAP),
        "time_exit_count": sum(1 for t in trades if t.exit_reason == EXIT_TIME),
        "end_of_data_count": sum(1 for t in trades if t.exit_reason == EXIT_END_OF_DATA),
        "win_count":     len(wins),
        "loss_count":    len(losses),
        "win_rate":      round(len(wins) / n, 4),
        "avg_win_r":     round(_mean(wins), 4),
        "avg_loss_r":    round(_mean(losses), 4),
        "expectancy_r":  round(_mean(r_vals), 4),
        "profit_factor": round(pf, 4) if not math.isinf(pf) else "inf",
        "median_r":      round(sorted(r_vals)[n // 2], 4),
        "best_trade_r":  round(max(r_vals), 4),
        "worst_trade_r": round(min(r_vals), 4),
        "net_pnl_usd":   round(sum(t.net_pnl for t in trades), 2),
        "initial_equity": params.initial_equity,
        "final_equity":  round(trades[-1].equity_after, 2),
        "total_return_pct": round(
            (trades[-1].equity_after / params.initial_equity - 1.0) * 100.0, 4
        ),
        "max_drawdown_pct": round(_max_drawdown_pct(eq_curve), 4),
        "sharpe_approx":    round(_approx_sharpe(r_vals), 4),
        "passed_min_trade_count": (n >= 100),
    }


def _yearly_breakdown(trades: list[TradeResult]) -> dict:
    by_year: dict[int, list] = defaultdict(list)
    for t in trades:
        by_year[t.setup.entry_timestamp.year].append(t)

    yearly = {}
    for yr in sorted(by_year):
        ts   = by_year[yr]
        rs   = [t.realized_r for t in ts]
        wins = [r for r in rs if r > 0]
        lss  = [r for r in rs if r <= 0]
        tp   = sum(1 for t in ts if t.exit_reason == EXIT_TP)
        sl   = sum(1 for t in ts if t.exit_reason in (EXIT_SL, EXIT_SL_GAP))
        te   = sum(1 for t in ts if t.exit_reason == EXIT_TIME)
        pf   = (sum(wins) / abs(sum(lss))) if lss else float("inf")
        yearly[str(yr)] = {
            "trades":        len(ts),
            "win_rate":      round(len(wins) / len(ts), 4),
            "expectancy_r":  round(_mean(rs), 4),
            "profit_factor": round(pf, 4) if not math.isinf(pf) else "inf",
            "avg_win_r":     round(_mean(wins), 4),
            "avg_loss_r":    round(_mean(lss), 4),
            "tp_sl_time":    f"{tp}/{sl}/{te}",
        }
    return yearly


def _r_distribution(trades: list[TradeResult]) -> dict:
    if not trades:
        return {}
    r_vals = sorted(t.realized_r for t in trades)
    n      = len(r_vals)
    def pct(p): return round(r_vals[int(p / 100 * (n - 1))], 4)
    mean_r   = sum(r_vals) / n
    variance = sum((r - mean_r) ** 2 for r in r_vals) / (n - 1) if n > 1 else 0.0
    return {
        "percentiles": {
            "p5": pct(5), "p10": pct(10), "p25": pct(25),
            "p50": pct(50), "p75": pct(75), "p90": pct(90), "p95": pct(95),
        },
        "mean": round(mean_r, 4),
        "std":  round(math.sqrt(variance), 4),
        "min":  round(r_vals[0], 4),
        "max":  round(r_vals[-1], 4),
        "buckets": {
            "lt_neg1":       sum(1 for r in r_vals if r < -1.0),
            "neg1_to_neg05": sum(1 for r in r_vals if -1.0 <= r < -0.5),
            "neg05_to_0":    sum(1 for r in r_vals if -0.5 <= r < 0.0),
            "0_to_05":       sum(1 for r in r_vals if 0.0 <= r < 0.5),
            "05_to_1":       sum(1 for r in r_vals if 0.5 <= r < 1.0),
            "1_to_15":       sum(1 for r in r_vals if 1.0 <= r < 1.5),
            "gte_15":        sum(1 for r in r_vals if r >= 1.5),
        },
    }


def _assemble(summary, yearly, r_dist, params: SOFParams) -> dict:
    def _clean(obj):
        if isinstance(obj, float):
            if math.isinf(obj): return "Infinity"
            if math.isnan(obj): return "NaN"
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_clean(v) for v in obj]
        return obj

    out = _clean(summary)
    out["params"] = {
        "macro_filter":    f"20-session MA slope over 5-session lookback < 0",
        "ma_period":       params.ma_period,
        "slope_lookback":  params.slope_lookback,
        "signal":          "PDH sweep-rejection (wick above PDH, close below)",
        "direction":       "SHORT only",
        "tp_rule":         "PDL (full prior-day range reversal — structural)",
        "stop_rule":       "wick high + stop_buffer_floor_pct * entry",
        "signal_window":   "07:00-16:00 UTC (+14h to +23h)",
        "time_exit":       "16:30 UTC (+23.5h)",
        "min_pdr_pct":     params.min_pdr_pct,
        "max_pdr_pct":     params.max_pdr_pct,
        "min_overshoot_pct": params.min_overshoot_pct,
        "spread_price":    params.spread_price,
        "slippage_price":  params.slippage_price,
        "risk_pct":        params.risk_pct,
    }
    out["yearly_breakdown"] = yearly
    out["r_distribution"]   = r_dist
    return out


if __name__ == "__main__":
    main()
