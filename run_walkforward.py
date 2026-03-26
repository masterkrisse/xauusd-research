"""
Walk-forward validation for the Combined Prior-Day Fade strategy.

Design:
  Rolling windows — 2-year training, 6-month test, 6-month step.
  No parameters are optimised on the training window.  The training window
  exists solely to initialise the 10-session regime filter: by providing
  prior-session close prices before the test window opens, the regime at
  the very first test session is computed from training data only.

  All strategy logic and parameters are fixed throughout.

  Segments (test windows):
    Seg 01: 2020-01 → 2020-06    (train: 2018-01 → 2019-12)
    Seg 02: 2020-07 → 2020-12    (train: 2018-07 → 2020-06)
    Seg 03: 2021-01 → 2021-06    (train: 2019-01 → 2020-12)
    Seg 04: 2021-07 → 2021-12    (train: 2019-07 → 2021-06)
    Seg 05: 2022-01 → 2022-06    (train: 2020-01 → 2021-12)
    Seg 06: 2022-07 → 2022-12    (train: 2020-07 → 2022-06)
    Seg 07: 2023-01 → 2023-06    (train: 2021-01 → 2022-12)
    Seg 08: 2023-07 → 2023-12    (train: 2021-07 → 2023-06)
    Seg 09: 2024-01 → 2024-06    (train: 2022-01 → 2023-12)
    Seg 10: 2024-07 → 2024-12    (train: 2022-07 → 2024-06)

  Execution:
    For each segment, the engine receives data covering (train_start, test_end].
    The engine uses ALL that data, but only trades whose entry_timestamp falls
    inside [test_start, test_end] are counted in the segment result.
    This is the correct walk-forward protocol: training data initialises the
    regime filter; test data is evaluated blind.

Usage:
    uv run python run_walkforward.py [results.json]

    Requires both IS and OOS data files to already exist:
      data/xauusd_15m_2018-01-01_2021-12-31.csv
      data/xauusd_15m_2022-01-01_2024-12-31.csv
"""

import json
import logging
import math
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from run_backtest import load_ohlcv
from src.strategies.asian_range_breakout.execution import (
    EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP, TradeResult,
)
from src.strategies.asian_range_breakout.results import (
    _approx_sharpe, _direction_summary, _max_drawdown_pct, _mean,
)
from src.strategies.combined_fade.engine import run_combined_fade_backtest
from src.strategies.prior_day_fade.params import PDFadeParams

logging.basicConfig(
    level=logging.WARNING,          # suppress engine noise during WF run
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Strategy parameters (unchanged from the combined backtest run) ─────────────
PARAMS = PDFadeParams(
    candle_minutes=15,
    signal_offset_hours=14.0,
    signal_window_end_hours=23.0,
    time_exit_hours=23.5,
    min_pdr_pct=0.0030,
    max_pdr_pct=0.0200,
    min_overshoot_pct=0.0002,
    spread_price=0.30,
    slippage_price=0.20,
    stop_buffer_floor_pct=0.0003,
    risk_pct=0.01,
    initial_equity=100_000.0,
)


# ── Walk-forward segment definitions ─────────────────────────────────────────
# (train_start, train_end, test_start, test_end)  — all inclusive
WF_SEGMENTS = [
    ("2018-01-01", "2019-12-31", "2020-01-01", "2020-06-30"),
    ("2018-07-01", "2020-06-30", "2020-07-01", "2020-12-31"),
    ("2019-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),
    ("2019-07-01", "2021-06-30", "2021-07-01", "2021-12-31"),
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
    ("2020-07-01", "2022-06-30", "2022-07-01", "2022-12-31"),
    ("2021-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-07-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2022-07-01", "2024-06-30", "2024-07-01", "2024-12-31"),
]


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "results/walkforward.json"

    # ── Load and combine all available data ───────────────────────────────────
    print("Loading data...", flush=True)
    df_is  = load_ohlcv("data/xauusd_15m_2018-01-01_2021-12-31.csv")
    df_oos = load_ohlcv("data/xauusd_15m_2022-01-01_2024-12-31.csv")
    df_all = pd.concat([df_is, df_oos]).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    print(f"Full dataset: {df_all.index[0].date()} → {df_all.index[-1].date()} "
          f"({len(df_all)} bars)\n", flush=True)

    # ── Run all segments ──────────────────────────────────────────────────────
    segment_results = []
    all_test_trades: List[TradeResult] = []

    for seg_idx, (tr_start, tr_end, te_start, te_end) in enumerate(WF_SEGMENTS, 1):
        seg = _run_segment(
            df_all=df_all,
            seg_idx=seg_idx,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start,  test_end=te_end,
        )
        segment_results.append(seg)
        all_test_trades.extend(seg["_trades"])
        _print_segment(seg)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = _aggregate(segment_results, all_test_trades)
    _print_aggregate(agg, segment_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "strategy": "XAUUSD_Combined_PD_Fade_RegimeFiltered",
        "version":  "v1.0_walkforward",
        "wf_design": {
            "train_window": "1 year (rolling)",
            "test_window":  "6 months",
            "step":         "6 months",
            "segments":     12,
            "full_period":  "2019-01 to 2024-12",
            "regime_filter": "10-session trend sign (no optimisation)",
        },
        "segments": [_clean_seg(s) for s in segment_results],
        "aggregate": agg,
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean_json(out), indent=2))
    print(f"\nResults saved → {output_path}")


# ── Segment runner ────────────────────────────────────────────────────────────

def _run_segment(
    df_all: pd.DataFrame,
    seg_idx: int,
    train_start: str, train_end: str,
    test_start: str,  test_end: str,
) -> dict:
    """
    Run one walk-forward segment.  Returns a metrics dict plus raw trades.
    """
    ts_start = pd.Timestamp(train_start, tz="UTC")
    te_end_ts = pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    test_end_ts   = pd.Timestamp(test_end,   tz="UTC") + pd.Timedelta(days=1)

    # Feed train+test data to the engine — test window sees no future data
    df_window = df_all[(df_all.index >= ts_start) & (df_all.index < te_end_ts)]

    # Silence engine logs during WF
    all_trades = run_combined_fade_backtest(df_window, PARAMS)

    # Only count trades entering in the test window
    test_trades = [
        t for t in all_trades
        if test_start_ts <= t.setup.entry_timestamp < test_end_ts
    ]

    metrics = _compute_segment_metrics(
        seg_idx=seg_idx,
        test_trades=test_trades,
        train_start=train_start, train_end=train_end,
        test_start=test_start,   test_end=test_end,
    )
    metrics["_trades"] = test_trades   # internal — stripped before JSON
    return metrics


def _compute_segment_metrics(
    seg_idx: int,
    test_trades: List[TradeResult],
    train_start: str, train_end: str,
    test_start:  str, test_end:   str,
) -> dict:
    n = len(test_trades)
    longs  = [t for t in test_trades if t.setup.direction ==  1]
    shorts = [t for t in test_trades if t.setup.direction == -1]

    def _dir_metrics(trades):
        if not trades:
            return {"trades": 0, "win_rate": None, "expectancy_r": None,
                    "avg_win_r": None, "avg_loss_r": None}
        r = [t.realized_r for t in trades]
        w = [x for x in r if x > 0]
        l = [x for x in r if x <= 0]
        return {
            "trades":      len(trades),
            "win_rate":    round(len(w) / len(trades), 4),
            "expectancy_r": round(_mean(r), 4),
            "avg_win_r":   round(_mean(w), 4),
            "avg_loss_r":  round(_mean(l), 4),
        }

    if n == 0:
        return {
            "segment": seg_idx,
            "train": f"{train_start} → {train_end}",
            "test":  f"{test_start} → {test_end}",
            "trades": 0, "win_rate": None, "expectancy_r": None,
            "profit_factor": None, "return_pct": None,
            "max_drawdown_pct": None, "sharpe": None,
            "tp": 0, "sl": 0, "sl_gap": 0, "time_exit": 0,
            "long": _dir_metrics([]), "short": _dir_metrics([]),
            "positive": False,
        }

    r_vals = [t.realized_r for t in test_trades]
    wins   = [r for r in r_vals if r > 0]
    losses = [r for r in r_vals if r <= 0]
    sl_l   = abs(sum(losses))
    pf     = (sum(wins) / sl_l) if sl_l > 0 else float("inf")

    # Notional equity curve for this segment (starting fresh at 100k for return calc)
    seg_equity = [PARAMS.initial_equity]
    eq = PARAMS.initial_equity
    for t in test_trades:
        eq += t.net_pnl
        seg_equity.append(eq)

    return {
        "segment":          seg_idx,
        "train":            f"{train_start} → {train_end}",
        "test":             f"{test_start} → {test_end}",
        "trades":           n,
        "win_rate":         round(len(wins) / n, 4),
        "expectancy_r":     round(_mean(r_vals), 4),
        "profit_factor":    round(pf, 4) if not math.isinf(pf) else "inf",
        "return_pct":       round((seg_equity[-1] / seg_equity[0] - 1) * 100, 3),
        "max_drawdown_pct": round(_max_drawdown_pct(seg_equity), 4),
        "sharpe":           round(_approx_sharpe(r_vals), 3),
        "tp":               sum(1 for t in test_trades if t.exit_reason == EXIT_TP),
        "sl":               sum(1 for t in test_trades if t.exit_reason == EXIT_SL),
        "sl_gap":           sum(1 for t in test_trades if t.exit_reason == EXIT_SL_GAP),
        "time_exit":        sum(1 for t in test_trades if t.exit_reason == EXIT_TIME),
        "long":             _dir_metrics(longs),
        "short":            _dir_metrics(shorts),
        "positive":         _mean(r_vals) > 0 if r_vals else False,
    }


# ── Aggregate ─────────────────────────────────────────────────────────────────

def _aggregate(segment_results: list, all_test_trades: List[TradeResult]) -> dict:
    segs_with_trades = [s for s in segment_results if s["trades"] > 0]
    all_r = [t.realized_r for t in all_test_trades]
    n = len(all_test_trades)

    if n == 0:
        return {"error": "no trades"}

    wins   = [r for r in all_r if r > 0]
    losses = [r for r in all_r if r <= 0]
    sl_l   = abs(sum(losses))
    pf     = (sum(wins) / sl_l) if sl_l > 0 else float("inf")

    # Full equity curve across all segments in chronological order
    all_test_trades_sorted = sorted(all_test_trades, key=lambda t: t.setup.entry_timestamp)
    equity_curve = [PARAMS.initial_equity]
    eq = PARAMS.initial_equity
    for t in all_test_trades_sorted:
        eq += t.net_pnl
        equity_curve.append(eq)

    exp_by_seg = [s["expectancy_r"] for s in segs_with_trades if s["expectancy_r"] is not None]
    pos_segs = sum(1 for s in segs_with_trades if s.get("positive", False))
    neg_segs = len(segs_with_trades) - pos_segs

    # Yearly summary
    by_year: dict[int, list] = {}
    for t in all_test_trades_sorted:
        yr = t.setup.entry_timestamp.year
        by_year.setdefault(yr, []).append(t.realized_r)

    yearly = {}
    for yr in sorted(by_year):
        rv = by_year[yr]
        w  = [r for r in rv if r > 0]
        yearly[str(yr)] = {
            "trades":       len(rv),
            "win_rate":     round(len(w) / len(rv), 4),
            "expectancy_r": round(_mean(rv), 4),
        }

    return {
        "total_test_trades":  n,
        "segments_with_trades": len(segs_with_trades),
        "positive_segments":  pos_segs,
        "negative_segments":  neg_segs,
        "pct_positive_segs":  round(pos_segs / len(segs_with_trades), 4) if segs_with_trades else None,
        "combined_expectancy_r": round(_mean(all_r), 4),
        "combined_win_rate":     round(len(wins) / n, 4),
        "combined_profit_factor": round(pf, 4) if not math.isinf(pf) else "inf",
        "combined_avg_win_r":    round(_mean(wins), 4),
        "combined_avg_loss_r":   round(_mean(losses), 4),
        "combined_return_pct":   round((equity_curve[-1] / equity_curve[0] - 1) * 100, 3),
        "combined_max_drawdown": round(_max_drawdown_pct(equity_curve), 4),
        "combined_sharpe":       round(_approx_sharpe(all_r), 3),
        "segment_expectancy_distribution": {
            "mean":   round(_mean(exp_by_seg), 4),
            "min":    round(min(exp_by_seg), 4),
            "max":    round(max(exp_by_seg), 4),
            "values": [round(e, 4) for e in exp_by_seg],
        },
        "yearly": yearly,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_segment(s: dict) -> None:
    sign = "+" if s.get("positive") else "-"
    n = s["trades"]
    exp = s["expectancy_r"]
    pf  = s["profit_factor"]
    ret = s["return_pct"]
    dd  = s["max_drawdown_pct"]
    sh  = s["sharpe"]

    l = s["long"];  s_ = s["short"]
    l_str = f"L:{l['trades']}@{l['expectancy_r']:+.3f}R" if l["trades"] else "L:0"
    s_str = f"S:{s_['trades']}@{s_['expectancy_r']:+.3f}R" if s_["trades"] else "S:0"

    if n == 0:
        print(f"  Seg {s['segment']:02d} [{s['test']}]  NO TRADES")
        return

    exp_str  = f"{exp:+.4f}R" if exp is not None else "  N/A  "
    pf_str   = f"{pf:.3f}" if isinstance(pf, float) else str(pf)
    ret_str  = f"{ret:+.1f}%" if ret is not None else "N/A"
    dd_str   = f"{dd:.1%}" if dd is not None else "N/A"
    sh_str   = f"{sh:+.2f}" if sh is not None else "N/A"
    tp_sl    = f"TP={s['tp']} SL={s['sl']}(+{s['sl_gap']}g) TIME={s['time_exit']}"

    print(f"  Seg {s['segment']:02d} [{s['test']}]  {sign}  "
          f"n={n:>3}  exp={exp_str}  PF={pf_str}  ret={ret_str}  "
          f"DD={dd_str}  Sharpe={sh_str}  |  {l_str}  {s_str}  |  {tp_sl}")


def _print_aggregate(agg: dict, segs: list) -> None:
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD AGGREGATE  (12 segments, 2019-01 → 2024-12)")
    print(f"{'='*80}")
    print(f"  Segments:         {agg['segments_with_trades']} with trades  "
          f"({agg['positive_segments']} positive / {agg['negative_segments']} negative = "
          f"{agg['pct_positive_segs']:.0%} hit rate)")
    print(f"  Total trades:     {agg['total_test_trades']}")
    print(f"  Combined exp:     {agg['combined_expectancy_r']:+.4f}R")
    print(f"  Combined WR:      {agg['combined_win_rate']:.1%}")
    print(f"  Profit factor:    {agg['combined_profit_factor']}")
    print(f"  Avg win/loss:     {agg['combined_avg_win_r']:+.4f}R / {agg['combined_avg_loss_r']:+.4f}R")
    print(f"  Combined return:  {agg['combined_return_pct']:+.1f}%")
    print(f"  Max drawdown:     {agg['combined_max_drawdown']:.1%}")
    print(f"  Sharpe (approx):  {agg['combined_sharpe']:+.3f}")
    print()
    d = agg["segment_expectancy_distribution"]
    print(f"  Segment expectancy:  mean={d['mean']:+.4f}R  "
          f"min={d['min']:+.4f}R  max={d['max']:+.4f}R")
    print(f"  Values: {d['values']}")
    print()
    print(f"  Yearly summary (across all test windows):")
    print(f"  {'Year':<6} {'Trades':>6} {'WinRate':>8} {'Expectancy':>11}")
    for yr, y in sorted(agg["yearly"].items()):
        print(f"  {yr:<6} {y['trades']:>6} {y['win_rate']:>8.1%} {y['expectancy_r']:>+11.4f}R")


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _clean_seg(s: dict) -> dict:
    """Remove internal _trades key before serialisation."""
    return {k: v for k, v in s.items() if k != "_trades"}


def _clean_json(obj):
    if isinstance(obj, float):
        if math.isinf(obj): return "Infinity"
        if math.isnan(obj): return "NaN"
    if isinstance(obj, dict):  return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_clean_json(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
