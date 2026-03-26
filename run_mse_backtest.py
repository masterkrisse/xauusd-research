"""
Multi-Session Trend Exhaustion — IS / OOS backtest entry point.

Usage:
    uv run python run_mse_backtest.py <data_file.csv> [results.json]

Strategy:
  Signal: 3 consecutive session closes in the same direction.
  Entry: 17:00 UTC session open (next session after signal).
  Stop: beyond S[-1] extreme.
  TP: S[-2] close (retrace to 2 sessions ago).
  Time exit: next 17:00 UTC.
"""

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

from run_backtest import load_ohlcv
from src.strategies.multi_session_exhaustion.engine import MSETradeResult, run_mse_backtest
from src.strategies.multi_session_exhaustion.params import MSEParams
from src.strategies.asian_range_breakout.execution import (
    EXIT_END_OF_DATA, EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP,
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
        logging.FileHandler("mse_backtest.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path   = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "mse_results.json"

    params = MSEParams(
        candle_minutes=15,
        n_sessions=3,
        stop_buffer_pct=0.0003,
        spread_price=0.30,
        slippage_price=0.20,
        risk_pct=0.01,
        initial_equity=100_000.0,
    )

    df     = load_ohlcv(data_path)
    trades = run_mse_backtest(df, params)

    summary = _compute_summary(trades, params)
    yearly  = _yearly_breakdown(trades)
    r_dist  = _r_distribution(trades)
    rr_analysis = _rr_analysis(trades)
    out     = _assemble(summary, yearly, r_dist, rr_analysis, params)

    json_str = json.dumps(out, indent=2)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json_str)
    logger.info("Results written to %s", output_path)
    print(json_str)


def _compute_summary(trades: list[MSETradeResult], params: MSEParams) -> dict:
    if not trades:
        return {"total_trades": 0, "note": "No trades generated"}

    n      = len(trades)
    r_vals = [t.realized_r for t in trades]
    wins   = [r for r in r_vals if r > 0.0]
    losses = [r for r in r_vals if r <= 0.0]
    sum_l  = abs(sum(losses))
    pf     = (sum(wins) / sum_l) if sum_l > 0.0 else float("inf")
    eq_curve = [params.initial_equity] + [t.equity_after for t in trades]

    long_t  = [t for t in trades if t.setup.direction == 1]
    short_t = [t for t in trades if t.setup.direction == -1]

    def dir_block(ts):
        if not ts:
            return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0,
                    "avg_win_r": 0.0, "avg_loss_r": 0.0}
        rs = [t.realized_r for t in ts]
        w  = [r for r in rs if r > 0]
        lo = [r for r in rs if r <= 0]
        return {
            "trades":       len(ts),
            "win_rate":     round(len(w) / len(ts), 4),
            "expectancy_r": round(_mean(rs), 4),
            "avg_win_r":    round(_mean(w), 4),
            "avg_loss_r":   round(_mean(lo), 4),
        }

    # Structural R:R per trade
    rr_vals = []
    for t in trades:
        s = t.setup
        tp_dist = abs(s.effective_tp - s.entry_price)
        rr_vals.append(tp_dist / s.stop_distance)

    return {
        "strategy":      "XAUUSD_Multi_Session_Exhaustion",
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
        "avg_rr_structural": round(sum(rr_vals) / len(rr_vals), 3),
        "long":          dir_block(long_t),
        "short":         dir_block(short_t),
        "net_pnl_usd":   round(sum(t.net_pnl for t in trades), 2),
        "initial_equity": params.initial_equity,
        "final_equity":  round(trades[-1].equity_after, 2),
        "total_return_pct": round(
            (trades[-1].equity_after / params.initial_equity - 1.0) * 100.0, 4
        ),
        "max_drawdown_pct": round(_max_drawdown_pct(eq_curve), 4),
        "sharpe_approx":    round(_approx_sharpe(r_vals), 4),
        "passed_min_trade_count": (n >= 100),   # lower bar: signal frequency is limited
    }


def _yearly_breakdown(trades: list[MSETradeResult]) -> dict:
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
        long_r  = _mean([t.realized_r for t in ts if t.setup.direction == 1])
        short_r = _mean([t.realized_r for t in ts if t.setup.direction == -1])
        rr_avg  = _mean([
            abs(t.setup.effective_tp - t.setup.entry_price) / t.setup.stop_distance
            for t in ts
        ])
        yearly[str(yr)] = {
            "trades":        len(ts),
            "win_rate":      round(len(wins) / len(ts), 4),
            "expectancy_r":  round(_mean(rs), 4),
            "profit_factor": round(pf, 4) if not math.isinf(pf) else "inf",
            "avg_win_r":     round(_mean(wins), 4),
            "avg_loss_r":    round(_mean(lss), 4),
            "avg_rr":        round(rr_avg, 3),
            "tp_sl_time":    f"{tp}/{sl}/{te}",
            "long_trades":   sum(1 for t in ts if t.setup.direction == 1),
            "short_trades":  sum(1 for t in ts if t.setup.direction == -1),
            "long_exp_r":    round(long_r, 4),
            "short_exp_r":   round(short_r, 4),
        }
    return yearly


def _rr_analysis(trades: list[MSETradeResult]) -> dict:
    """Distribution of structural R:R ratios — key diagnostic for stop placement."""
    if not trades:
        return {}
    rr_vals = sorted(
        abs(t.setup.effective_tp - t.setup.entry_price) / t.setup.stop_distance
        for t in trades
    )
    n = len(rr_vals)
    def pct(p): return round(rr_vals[int(p / 100 * (n - 1))], 3)
    return {
        "min":  round(rr_vals[0], 3),
        "p25":  pct(25),
        "p50":  pct(50),
        "p75":  pct(75),
        "max":  round(rr_vals[-1], 3),
        "mean": round(sum(rr_vals) / n, 3),
        "lt_0.5":  sum(1 for r in rr_vals if r < 0.5),
        "0.5_1.0": sum(1 for r in rr_vals if 0.5 <= r < 1.0),
        "gte_1.0": sum(1 for r in rr_vals if r >= 1.0),
    }


def _r_distribution(trades: list[MSETradeResult]) -> dict:
    if not trades:
        return {}
    r_vals = sorted(t.realized_r for t in trades)
    n      = len(r_vals)
    def pct(p): return round(r_vals[int(p / 100 * (n - 1))], 4)
    mean_r   = sum(r_vals) / n
    variance = sum((r - mean_r) ** 2 for r in r_vals) / (n - 1) if n > 1 else 0.0
    return {
        "percentiles": {
            "p5":  pct(5),  "p10": pct(10), "p25": pct(25),
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


def _assemble(summary, yearly, r_dist, rr_analysis, params: MSEParams) -> dict:
    def _clean(obj):
        if isinstance(obj, float):
            if math.isinf(obj): return "Infinity"
            if math.isnan(obj): return "NaN"
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_clean(v) for v in obj]
        return obj

    out = _clean(summary)
    out["params"] = {
        "n_sessions":      params.n_sessions,
        "stop_rule":       "S[-1].extreme + stop_buffer_pct * entry",
        "tp_rule":         "S[-2].close (retrace to 2 sessions ago)",
        "time_exit":       "next 17:00 UTC (one full session)",
        "stop_buffer_pct": params.stop_buffer_pct,
        "spread_price":    params.spread_price,
        "slippage_price":  params.slippage_price,
        "risk_pct":        params.risk_pct,
    }
    out["rr_analysis"]       = rr_analysis
    out["yearly_breakdown"]  = yearly
    out["r_distribution"]    = r_dist
    return out


if __name__ == "__main__":
    main()
