"""
Walk-forward validation for Multi-Session Trend Exhaustion.

Design:
  1-year training, 6-month test, 6-month step.
  No learned components — training window provides session-close history to
  seed the N-consecutive detection for the first test sessions.

  Segments (test windows):
    Seg 01: 2019-01 → 2019-06    (train: 2018-01 → 2018-12)
    Seg 02: 2019-07 → 2019-12    (train: 2018-07 → 2019-06)
    Seg 03: 2020-01 → 2020-06    (train: 2019-01 → 2019-12)
    Seg 04: 2020-07 → 2020-12    (train: 2019-07 → 2020-06)
    Seg 05: 2021-01 → 2021-06    (train: 2020-01 → 2020-12)
    Seg 06: 2021-07 → 2021-12    (train: 2020-07 → 2021-06)
    Seg 07: 2022-01 → 2022-06    (train: 2021-01 → 2021-12)
    Seg 08: 2022-07 → 2022-12    (train: 2021-07 → 2022-06)
    Seg 09: 2023-01 → 2023-06    (train: 2022-01 → 2022-12)
    Seg 10: 2023-07 → 2023-12    (train: 2022-07 → 2023-06)
    Seg 11: 2024-01 → 2024-06    (train: 2023-01 → 2023-12)
    Seg 12: 2024-07 → 2024-12    (train: 2023-07 → 2024-06)

Usage:
    uv run python run_mse_walkforward.py [results.json]
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from run_backtest import load_ohlcv
from src.strategies.multi_session_exhaustion.engine import run_mse_backtest
from src.strategies.multi_session_exhaustion.params import MSEParams
from src.strategies.asian_range_breakout.execution import EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP
from src.strategies.asian_range_breakout.results import _approx_sharpe, _max_drawdown_pct, _mean

WF_SEGMENTS = [
    ("2018-01-01", "2018-12-31", "2019-01-01", "2019-06-30"),
    ("2018-07-01", "2019-06-30", "2019-07-01", "2019-12-31"),
    ("2019-01-01", "2019-12-31", "2020-01-01", "2020-06-30"),
    ("2019-07-01", "2020-06-30", "2020-07-01", "2020-12-31"),
    ("2020-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),
    ("2020-07-01", "2021-06-30", "2021-07-01", "2021-12-31"),
    ("2021-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
    ("2021-07-01", "2022-06-30", "2022-07-01", "2022-12-31"),
    ("2022-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2022-07-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2023-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2023-07-01", "2024-06-30", "2024-07-01", "2024-12-31"),
]

PARAMS = MSEParams(
    n_sessions=3, stop_buffer_pct=0.0003,
    spread_price=0.30, slippage_price=0.20,
    risk_pct=0.01, initial_equity=100_000.0,
)


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "results/mse_walkforward.json"

    print("Loading data...", flush=True)
    df_is  = load_ohlcv("data/xauusd_15m_2018-01-01_2021-12-31.csv")
    df_oos = load_ohlcv("data/xauusd_15m_2022-01-01_2024-12-31.csv")
    df_all = pd.concat([df_is, df_oos]).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    print(f"Full dataset: {df_all.index[0].date()} → {df_all.index[-1].date()} "
          f"({len(df_all):,} bars)", flush=True)

    segment_results = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(WF_SEGMENTS, 1):
        test_start  = pd.Timestamp(te_s, tz="UTC")
        test_end    = pd.Timestamp(te_e, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        train_start = pd.Timestamp(tr_s, tz="UTC")

        df_window = df_all[
            (df_all.index >= train_start)
            & (df_all.index <= test_end + pd.Timedelta(hours=26))
        ]

        all_trades = run_mse_backtest(df_window, PARAMS)
        test_trades = [
            t for t in all_trades
            if test_start <= t.setup.entry_timestamp <= test_end
        ]

        seg = _segment_stats(i, tr_s, tr_e, te_s, te_e, test_trades)
        segment_results.append(seg)

        sign = "+" if seg["positive"] else "-"
        print(
            f"  Seg {i:02d} [{te_s} → {te_e}]  {sign}  "
            f"n={seg['trades']:3d}  "
            f"exp={seg['expectancy_r']:+.4f}R  "
            f"PF={seg['profit_factor']}  "
            f"ret={seg['return_pct']:+.1f}%  "
            f"DD={seg['max_drawdown_pct']:.1f}%  "
            f"Sharpe={seg['sharpe']:+.2f}  "
            f"avgRR={seg['avg_rr']:.2f}  |  "
            f"L:{seg['long']['trades']}@{seg['long']['expectancy_r']:+.3f}R  "
            f"S:{seg['short']['trades']}@{seg['short']['expectancy_r']:+.3f}R  |  "
            f"TP={seg['tp']} SL={seg['sl']}(+{seg['sl_gap']}g) TIME={seg['time_exit']}",
            flush=True,
        )

    # Aggregate
    all_r = [seg["expectancy_r"] for seg in segment_results if seg["trades"] > 0]
    pos   = sum(1 for s in segment_results if s["positive"])
    neg   = len(segment_results) - pos
    total = sum(s["trades"] for s in segment_results)

    all_r_flat = []
    for seg in segment_results:
        all_r_flat.extend(seg.get("_r_values", []))

    combined_exp = _mean(all_r_flat) if all_r_flat else 0.0
    all_wins  = [r for r in all_r_flat if r > 0]
    all_losses= [r for r in all_r_flat if r <= 0]
    pf_all = (sum(all_wins) / abs(sum(all_losses))) if all_losses else float("inf")

    yearly: dict[str, list] = defaultdict(list)
    for seg in segment_results:
        for yr, rs in seg.get("_by_year", {}).items():
            yearly[yr].extend(rs)

    yearly_out = {}
    for yr in sorted(yearly):
        rs   = yearly[yr]
        wins = [r for r in rs if r > 0]
        yearly_out[yr] = {
            "trades":       len(rs),
            "win_rate":     round(len(wins) / len(rs), 4) if rs else 0.0,
            "expectancy_r": round(_mean(rs), 4),
        }

    print()
    print("=" * 80)
    print(f"  WALK-FORWARD AGGREGATE  (12 segments, 2019-01 → 2024-12)")
    print("=" * 80)
    print(f"  Segments:         {len(segment_results)}  ({pos} positive / {neg} negative = {pos/len(segment_results):.0%} hit rate)")
    print(f"  Total trades:     {total}")
    print(f"  Combined exp:     {combined_exp:+.4f}R")
    print(f"  Combined WR:      {len(all_wins)/len(all_r_flat):.1%}" if all_r_flat else "")
    print(f"  Profit factor:    {round(pf_all,4) if not math.isinf(pf_all) else 'inf'}")
    print(f"  Avg win/loss:     {round(_mean(all_wins),4):+}R / {round(_mean(all_losses),4):+}R")
    print(f"  Sharpe (approx):  {round(_approx_sharpe(all_r_flat),3):+}")
    print()
    print(f"  Segment expectancy:  mean={round(_mean(all_r),4):+}R  min={min(all_r):+.4f}R  max={max(all_r):+.4f}R")
    print(f"  Values: {[round(r,4) for r in all_r]}")
    print()
    print(f"  Yearly summary:")
    print(f"  {'Year':<6} {'Trades':>6}  {'WinRate':>7}  {'Expectancy':>10}")
    for yr, v in yearly_out.items():
        print(f"  {yr:<6} {v['trades']:>6}    {v['win_rate']:.1%}     {v['expectancy_r']:+.4f}R")

    # Save
    def _clean(obj):
        if isinstance(obj, float):
            if math.isinf(obj): return "Infinity"
            if math.isnan(obj): return "NaN"
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_clean(v) for v in obj]
        return obj

    clean_segs = [{k: v for k, v in s.items() if not k.startswith("_")} for s in segment_results]
    output = _clean({
        "strategy": "XAUUSD_Multi_Session_Exhaustion",
        "version": "v1.0_walkforward",
        "wf_design": {"train_window": "1 year", "test_window": "6 months", "step": "6 months", "segments": 12},
        "segments": clean_segs,
        "aggregate": {
            "positive_segments": pos, "negative_segments": neg,
            "hit_rate": round(pos / len(segment_results), 4),
            "total_trades": total,
            "combined_expectancy_r": round(combined_exp, 4),
            "profit_factor": round(pf_all, 4) if not math.isinf(pf_all) else "Infinity",
            "segment_exp_values": [round(r, 4) for r in all_r],
            "yearly": yearly_out,
        },
    })
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {output_path}")


def _segment_stats(i, tr_s, tr_e, te_s, te_e, trades) -> dict:
    if not trades:
        return {
            "segment": i, "train": f"{tr_s} → {tr_e}", "test": f"{te_s} → {te_e}",
            "trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0,
            "return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "avg_rr": 0.0,
            "tp": 0, "sl": 0, "sl_gap": 0, "time_exit": 0,
            "long": {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0},
            "short": {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0},
            "positive": False, "_r_values": [], "_by_year": {},
        }

    r_vals = [t.realized_r for t in trades]
    wins   = [r for r in r_vals if r > 0]
    losses = [r for r in r_vals if r <= 0]
    pf     = (sum(wins) / abs(sum(losses))) if losses else float("inf")

    eq, eq_curve = PARAMS.initial_equity, [PARAMS.initial_equity]
    for t in trades:
        eq += t.net_pnl; eq_curve.append(eq)

    rr_vals = [abs(t.setup.effective_tp - t.setup.entry_price) / t.setup.stop_distance for t in trades]

    def dir_s(ts):
        if not ts: return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0}
        rs = [t.realized_r for t in ts]; w = [r for r in rs if r > 0]
        return {"trades": len(ts), "win_rate": round(len(w)/len(ts),4), "expectancy_r": round(_mean(rs),4)}

    by_year: dict[str, list] = defaultdict(list)
    for t in trades:
        by_year[str(t.setup.entry_timestamp.year)].append(t.realized_r)

    return {
        "segment":          i,
        "train":            f"{tr_s} → {tr_e}",
        "test":             f"{te_s} → {te_e}",
        "trades":           len(trades),
        "win_rate":         round(len(wins) / len(trades), 4),
        "expectancy_r":     round(_mean(r_vals), 4),
        "profit_factor":    round(pf, 4) if not math.isinf(pf) else "inf",
        "return_pct":       round((eq_curve[-1] / PARAMS.initial_equity - 1) * 100, 2),
        "max_drawdown_pct": round(_max_drawdown_pct(eq_curve) * 100, 2),
        "sharpe":           round(_approx_sharpe(r_vals), 2),
        "avg_rr":           round(_mean(rr_vals), 3),
        "tp":               sum(1 for t in trades if t.exit_reason == EXIT_TP),
        "sl":               sum(1 for t in trades if t.exit_reason == EXIT_SL),
        "sl_gap":           sum(1 for t in trades if t.exit_reason == EXIT_SL_GAP),
        "time_exit":        sum(1 for t in trades if t.exit_reason == EXIT_TIME),
        "long":             dir_s([t for t in trades if t.setup.direction == 1]),
        "short":            dir_s([t for t in trades if t.setup.direction == -1]),
        "positive":         _mean(r_vals) > 0,
        "_r_values":        r_vals,
        "_by_year":         {yr: rs for yr, rs in by_year.items()},
    }


if __name__ == "__main__":
    main()
